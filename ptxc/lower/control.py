"""Launch structure and control flow: Tile, Sync, Loop / StridedLoop and Cond."""

from __future__ import annotations

import math

from ptxc.index import Lin
from ptxc.kernel import static_int
from ptxc.lower.base import FloatV, IntV
from ptxc.lower.walk import has_barrier, head, identity, size

UNROLL_BUDGET = 4096  # max statements a loop the IR marks `unroll` may expand to
SMALL_LOOP_TRIPS = 16  # unmarked loops with at most this many trips are unrolled too ...
SMALL_LOOP_BUDGET = 256  # ... if that stays within this many statements


def unroll_factor(body_size: int) -> int:
    """Copies per trip of a partially unrolled loop: more for small bodies."""
    if body_size <= 4:
        return 8
    if body_size <= 16:
        return 4
    if body_size <= 64:
        return 2
    return 1


class ControlMixin:
    # -- the thread grid ---------------------------------------------------
    def s_Tile(self, t) -> None:
        extents = [static_int(a.extent) for a in t.axes]
        n = math.prod(extents)
        cells = t.block_threads or self.spec.block[0]
        self.b.comment(head(t))
        # The launch geometry is part of the contract, so blockDim is a constant here.
        if t.aux_threads:
            gid = self.ctaid().scale(t.block_threads) + self.ints.mod(self.tid(), t.block_threads)
        else:
            gid = self.ctaid().scale(self.spec.block[0]) + self.tid()
        # Threads past the iteration space (p00: 64 cells, 256 threads) must not run.
        if t.aux_threads or n % cells != 0 or self.spec.threads_launched > n:
            past = self.ints.compare(">=", gid, Lin(const=n))
            if past is True:
                return
            if past is not False:
                self.used_exit = True
                self.b.ins("bra", self.exit_label, guard=f"@{past}")
        self.push()
        for axis, value in zip(t.axes, self._decode_axes(t, extents, gid), strict=True):
            self.define(axis.name, IntV(value))
        self.body(t.body)
        self.pop()

    def _decode_axes(self, t, extents: list[int], gid: Lin) -> list[Lin]:
        """axis_i = (gid / stride_i) % extent_i, row-major; optional grouped rasterization."""
        strides = [math.prod(extents[i + 1 :]) for i in range(len(extents))]
        vals: list[Lin | None] = [None] * len(extents)
        raster = _raster_plan(t, extents)
        if raster is not None:
            mi, ni, gi, oi, g = raster
            ea, eb = extents[gi], extents[oi]
            sub = self.ints.div(gid, strides[ni])
            if mi != 0:
                sub = self.ints.mod(sub, ea * eb)
            within = self.ints.mod(sub, g * eb)
            if ea % g == 0:
                a_val = self.ints.div(sub, g * eb).scale(g) + self.ints.mod(sub, g)
                b_val = self.ints.div(within, g)
            else:
                grp = self.ints.div(sub, g * eb)
                group_size = self.ints.minimum(Lin(const=g), Lin(const=ea) - grp.scale(g))
                a_val = grp.scale(g) + self.ints.divmod_dyn(within, group_size, "mod")
                b_val = self.ints.divmod_dyn(within, group_size, "div")
            vals[gi], vals[oi] = a_val, b_val
        for i, e in enumerate(extents):
            if vals[i] is not None:
                continue
            if e == 1:
                vals[i] = Lin()
                continue
            v = self.ints.div(gid, strides[i])
            vals[i] = v if i == 0 else self.ints.mod(v, e)
        return vals

    def s_Sync(self, s) -> None:
        if s.warp:
            self.b.ins("bar.warp.sync", "-1")
        elif s.barrier_id == 0:
            self.b.ins("bar.sync", "0")
        else:
            self.b.ins("bar.sync", str(s.barrier_id), str(s.count))

    # -- loops -------------------------------------------------------------
    def seed_accums(self, loop) -> None:
        """Seed the loop's own Accums with their identity, in the enclosing scope."""
        if not loop.seed:
            return
        seen = set()
        for s in loop.body:
            if type(s).__name__ == "Accum" and s.name not in seen:
                seen.add(s.name)
                self.define(s.name, FloatV(self.const_f32(identity(s.op))))

    def s_Loop(self, s) -> None:
        self.b.comment(head(s))
        self.seed_accums(s)
        extent = static_int(s.axis.extent)
        if extent <= 0:
            return
        self._counted_loop(s.axis.name, Lin(), Lin(const=extent), 1, s.body, unroll=s.unroll)

    def s_StridedLoop(self, s) -> None:
        self.b.comment(head(s))
        self.seed_accums(s)
        start = self.as_int(self.expr(s.start))
        step = self.as_int(self.expr(s.step))
        end = self.as_int(self.expr(s.end)) if s.end is not None else Lin(const=static_int(s.axis.extent))
        if not step.is_const or step.const <= 0:
            raise NotImplementedError("StridedLoop with a non-constant step")
        self._counted_loop(s.axis.name, start, end, step.const, s.body, unroll=s.unroll)

    def _counted_loop(self, var: str, start: Lin, end: Lin, step: int, body, *, unroll: bool) -> None:
        """for (var = start; var < end; var += step) body

        Three shapes, cheapest first:
        * fully unrolled, when the trip count is small and known from the value ranges:
          `var` becomes `start + k*step`, an affine form, so it folds into every address;
          trailing copies that only some threads run get a predicate;
        * partially unrolled by U (a main loop doing U copies per trip, then a remainder
          loop): the U loads of one trip are independent and can be in flight together;
        * a plain bottom-tested loop.
        A body with a barrier only gets the shapes in which every thread does the same.
        """
        body_size = size(body)
        uniform_only = has_barrier(body)
        start_rng = start.range(self.ints.ranges)
        if start_rng is not None and end.is_const:
            min_trips = max(0, -(-(end.const - start_rng[1]) // step))
            max_trips = max(0, -(-(end.const - start_rng[0]) // step))
            if max_trips == 0:
                return
            budget = UNROLL_BUDGET if unroll else SMALL_LOOP_BUDGET
            can_guard = not uniform_only or min_trips == max_trips
            small = max_trips <= SMALL_LOOP_TRIPS or unroll
            if can_guard and (max_trips == 1 or (small and max_trips * body_size <= budget)):
                self._unrolled_copies(var, start, end, step, body, min_trips, max_trips)
                return
        if self.ints.compare("<", start, end) is False:
            return
        factor = 1 if uniform_only or not end.is_const else unroll_factor(body_size)
        lo = start_rng[0] if start_rng is not None else None
        i = self.ints.fresh()  # the counter; it only has a range inside the bodies below
        self.b.ins("mov.u32", i, self.ints.operand(start))
        entry = start  # what the next entry test compares: `start` itself still has its range
        if factor > 1:
            entry = self._main_loop(var, i, start, end, step, body, factor, lo)
        end_rng = end.range(self.ints.ranges)
        self._plain_loop(var, i, entry, end, step, body, body_hi=end_rng[1] - 1 if end_rng else None, lo=lo)

    def _main_loop(self, var: str, i: str, start: Lin, end: Lin, step: int, body, factor: int, lo) -> Lin:
        """U copies per trip while the last of them is still in range. Returns what the
        remainder loop's entry test compares."""
        last = end.const - (factor - 1) * step
        enters = self.ints.compare("<", start, Lin(const=last))
        if enters is False:
            return start
        main_end = self.b.label("main_end")
        if enters is not True:
            self.b.ins("bra", main_end, guard=f"@!{enters}")
        loop_head = self.b.label("main")
        self.b.place(loop_head)
        self._counter_range(i, lo, last - 1)
        self.push()  # the trip's scope: copies share values here, never past the loop
        for k in range(factor):
            self.push(shares_values=True)
            self.define(var, IntV(Lin.of(i) + k * step))
            self.body(body)
            self.pop()
        self.pop()
        self.ints.ranges.pop(i, None)
        self.b.ins("add.s32", i, i, str(factor * step))
        again = self.b.reg("pred")
        self.b.ins("setp.lt.s32", again, i, str(last))
        self.b.ins("bra", loop_head, guard=f"@{again}")
        self.b.place(main_end)
        return Lin.of(i)

    def _counter_range(self, i: str, lo: int | None, hi: int | None) -> None:
        """Give loop counter `i` its in-body range (only while the body is emitted)."""
        if lo is not None and hi is not None:
            self.ints.ranges[i] = (lo, hi)

    def _unrolled_copies(self, var: str, start: Lin, end: Lin, step: int, body, min_trips: int, max_trips: int) -> None:
        for k in range(max_trips):
            value = start + k * step
            skip = None
            if k >= min_trips:  # only some threads get this far
                p = self.ints.compare("<", value, end)
                if p is False:
                    break
                if p is not True:
                    skip = self.b.label("tail_skip")
                    self.b.ins("bra", skip, guard=f"@!{p}")
            # an unguarded copy is straight-line code, so later copies may reuse its values
            self.push(shares_values=skip is None)
            self.define(var, IntV(value))
            self.body(body)
            self.pop()
            if skip:
                self.b.place(skip)

    def _plain_loop(self, var: str, i: str, entry: Lin, end: Lin, step: int, body, *, body_hi, lo) -> None:
        """Bottom-tested loop over counter `i`, which already holds its first value (`entry`)."""
        done = self.b.label("loop_end")
        enters = self.ints.compare("<", entry, end)
        if enters is False:
            return
        if enters is not True:
            self.b.ins("bra", done, guard=f"@!{enters}")
        loop_head = self.b.label("loop")
        self.b.place(loop_head)
        self._counter_range(i, lo, body_hi)
        self.push()
        self.define(var, IntV(Lin.of(i)))
        self.body(body)
        self.pop()
        self.ints.ranges.pop(i, None)
        self.b.ins("add.s32", i, i, str(step))
        again = self.b.reg("pred")
        self.b.ins("setp.lt.s32", again, i, self.ints.operand(end))
        self.b.ins("bra", loop_head, guard=f"@{again}")
        self.b.place(done)

    # -- branches ----------------------------------------------------------
    def s_Cond(self, s) -> None:
        self.b.comment(f"if ({s.cond.pretty()})")
        p = self.pred(self.expr(s.cond))
        if isinstance(p, bool):
            self.push()
            self.body(s.body if p else s.else_body)
            self.pop()
            return
        else_label, end_label = self.b.label("else"), self.b.label("endif")
        self.b.ins("bra", else_label if s.else_body else end_label, guard=f"@!{p}")
        self.push()
        self.body(s.body)
        self.pop()
        if s.else_body:
            self.b.ins("bra.uni", end_label)
            self.b.place(else_label)
            self.push()
            self.body(s.else_body)
            self.pop()
        self.b.place(end_label)


def _raster_plan(t, extents: list[int]):
    """(m index, n index, grouped index, other index, group) for a grouped rasterization, or None."""
    names = [a.name for a in t.axes]
    if not (t.raster_group and t.raster_axes and all(a in names for a in t.raster_axes)):
        return None
    mi, ni = names.index(t.raster_axes[0]), names.index(t.raster_axes[1])
    if ni != mi + 1:
        return None
    gi, oi = (mi, ni) if t.raster_orient == "m" else (ni, mi)
    g = min(t.raster_group, extents[gi])
    if extents[gi] <= 1 or g <= 1:
        return None
    return mi, ni, gi, oi, g
