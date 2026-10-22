"""Scalar compute: Assign, Accum (with fma fusion), Let, Init and Select."""

from __future__ import annotations

from ptxc import ops
from ptxc.lower.base import FloatV, IntV
from ptxc.lower.walk import identity

# bool masks travel as f32 0.0 / 1.0, Emmy's own convention (`where` tests nonzero)
_FLOAT_DTYPES = ("f32", "f16", "bf16", "bool")


class ComputeMixin:
    def s_Assign(self, s) -> None:
        if s.dtype is not None and s.dtype.name not in _FLOAT_DTYPES:
            raise NotImplementedError(f"Assign {s.name} at dtype {s.dtype.name}")
        args = [self.f32(a) for a in s.args]
        if s.name in self.inplace:
            dst = self.lookup(s.name).reg
            if s.op.name in ("copy", "pad") and args[0] == dst:
                return
        else:
            dst = self.b.reg("f32")
        ops.emit_op(self.b, s.op.name, args, dst)
        if s.name in self.inplace:
            self.mutated(dst)
        else:
            self.define(s.name, FloatV(dst))
            if ops.canonical(s.op.name) == "multiply":
                self.scope.cse[("product", dst)] = (args[0], args[1], self.writes)

    def _fusable_product(self, value: str) -> tuple[str, str] | None:
        """The factors of `value` if it is a multiply from this same scope whose factors
        have not been overwritten since, so `acc + value` may become one fma."""
        entry = self.scope.cse.get(("product", value))
        if entry is None:
            return None
        a, b, at = entry
        if self.last_write.get(a, 0) > at or self.last_write.get(b, 0) > at:
            return None
        return a, b

    def s_Accum(self, s) -> None:
        acc = self.scope.lookup(s.name)
        if acc is None:  # no seeding loop: start from the identity here
            acc = FloatV(self.const_f32(identity(s.op)))
            self.define(s.name, acc)
        base = acc.reg if s.base is None else self.f32(s.base)
        value = self.f32(s.value)
        factors = self._fusable_product(value) if ops.canonical(s.op.name) == "add" else None
        if factors is not None:
            # acc = base + a * b as one fused multiply-add (the separate mul becomes dead code)
            self.b.ins("fma.rn.f32", acc.reg, factors[0], factors[1], base)
        else:
            ops.emit_op(self.b, s.op.name, [base, value], acc.reg)
        self.mutated(acc.reg)

    def s_Let(self, s) -> None:
        v = self.expr(s.value)
        if s.dtype is not None and s.dtype.name in ("f32", "f16", "bf16"):
            v = FloatV(self.as_f32(v))
        elif not isinstance(v, (IntV, FloatV)):
            v = FloatV(self.as_f32(v))  # a predicate bound to a name
        self.define(s.name, v)

    def s_Init(self, s) -> None:
        self.define(s.name, FloatV(self.const_f32(float(s.identity))))

    def s_Select(self, s) -> None:
        """name = the value of the first branch whose predicate holds (the last is the fallback):
        a chain of `selp` built from the back, with statically decided predicates folded."""
        preds = [self.pred(self.expr(br.select)) for br in s.branches]
        result = None
        for br, p in reversed(list(zip(s.branches, preds, strict=True))):
            if p is False and result is not None:
                continue
            value = self.f32(br.value)
            if result is None or p is True:
                result = value
                continue
            r = self.b.reg("f32")
            self.b.ins("selp.f32", r, value, result, p)
            result = r
        self.define(s.name, FloatV(result))
