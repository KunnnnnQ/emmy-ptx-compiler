"""Cross-thread combines: WarpShuffle (registers) and TreeHalve (shared memory)."""

from __future__ import annotations

from ptxc.index import Lin
from ptxc.lower.base import FloatV


class CombineMixin:
    def combine(self, program, states) -> None:
        """Run a combine program (Assigns) that updates `states` in place."""
        saved = self.inplace
        self.inplace = frozenset(states)
        for a in program:
            self.s_Assign(a)
        self.inplace = saved

    def s_WarpShuffle(self, s) -> None:
        """XOR butterfly: at each step every lane folds in the state of the lane `step` away."""
        mask = self.ints.fresh()
        self.b.ins("activemask.b32", mask)
        step = int(s.length) // 2
        while step > 0:
            self.push()
            for st, sb in zip(s.state, s.state_b, strict=True):
                other = self.b.reg("f32")
                self.b.ins("shfl.sync.bfly.b32", other, self.f32(st), str(step), "31", mask)
                self.define(sb, FloatV(other))
            self.combine(s.combine_states, s.state)
            self.pop()
            step //= 2

    def s_TreeHalve(self, s) -> None:
        """Halving tree over shared slots: slot t folds in slot t+half, a barrier, repeat;
        then every thread reads the root."""
        t = self.as_int(self.lookup(s.tid_var))
        bufs = [self.buffer(name) for name in s.bufs]
        if s.inner is not None:  # segmented: each tree level halves the segment index
            inner_var, scale = s.inner
            inner = self.as_int(self.lookup(inner_var))
            slot, root = t.scale(scale) + inner, inner
        else:
            scale, slot, root = 1, t, Lin()
        sync = ("bar.sync", "0") if s.barrier_id == 0 else ("bar.sync", str(s.barrier_id), str(s.barrier_count))
        half = int(s.length) // 2
        while half > 0:
            active = self.ints.compare("<", t, Lin(const=half))
            skip = self.b.label("tree_skip")
            if active is not True:
                self.b.ins("bra", skip, guard=f"@!{active}")
            if active is not False:
                self.push()
                for buf, st, sb in zip(bufs, s.state, s.state_b, strict=True):
                    mine, theirs = self.b.reg("f32"), self.b.reg("f32")
                    self.load_into(buf, slot, [mine])
                    self.load_into(buf, slot + half * scale, [theirs])
                    self.define(st, FloatV(mine))  # shadows the thread's own state
                    self.define(sb, FloatV(theirs))
                self.combine(s.combine_states, s.state)
                for buf, st in zip(bufs, s.state, strict=True):
                    self.b.ins("st.shared.f32", self.address(buf, slot), self.f32(st))
                self.pop()
            if active is not True:
                self.b.place(skip)
            self.b.ins(*sync)
            half //= 2
        # every thread picks up the reduced state
        for buf, st in zip(bufs, s.state, strict=True):
            reg = self.lookup(st).reg
            self.load_into(buf, root, [reg])
            self.mutated(reg)
