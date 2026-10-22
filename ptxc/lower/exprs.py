"""Expressions: indices, predicates and loop bounds (Var, Literal, BinaryExpr, ...).

`expr()` evaluates to an IntV (an affine form), a FloatV, or a predicate, which
is a register name or a Python bool when the value ranges already decide it.
"""

from __future__ import annotations

from ptxc import ops
from ptxc.index import Lin
from ptxc.lower.base import FloatV, IntV
from ptxc.ptx import f32_imm

_FLOAT_CMP = {"<": "lt", "<=": "le", ">": "gt", ">=": "ge", "==": "eq", "!=": "ne"}
_FLOAT_ARITH = {"+": "add", "-": "subtract", "*": "multiply", "/": "divide"}


class ExprMixin:
    def expr(self, e):
        kind = type(e).__name__
        if kind == "Var":
            return self.lookup(e.name)
        if kind == "Literal":
            if e.dtype == "int" or (isinstance(e.value, int) and not isinstance(e.value, bool)):
                return IntV(Lin(const=int(e.value)))
            return FloatV(self.const_f32(float(e.value)))
        if kind == "Builtin":
            v = self.builtin(e.name)
            if v is None:
                raise NotImplementedError(f"builtin {e.name!r}")
            return v
        if kind == "BinaryExpr":
            return self.binary(e.op, self.expr(e.left), self.expr(e.right))
        if kind == "TernaryExpr":
            p = self.pred(self.expr(e.cond))
            if p is True:
                return self.expr(e.if_true)
            if p is False:
                return self.expr(e.if_false)
            a, c = self.expr(e.if_true), self.expr(e.if_false)
            if isinstance(a, IntV) and isinstance(c, IntV):
                return IntV(self.ints.select(p, a.lin, c.lin))
            r = self.b.reg("f32")
            self.b.ins("selp.f32", r, self.as_f32(a), self.as_f32(c), p)
            return FloatV(r)
        if kind == "CastExpr":
            v = self.expr(e.expr)
            if str(e.dtype) in ("int", "i32", "long long", "i64"):
                if isinstance(v, IntV):
                    return v
                r = self.ints.fresh()
                self.b.ins("cvt.rzi.s32.f32", r, self.as_f32(v))
                return IntV(Lin.of(r))
            return FloatV(self.as_f32(v))
        if kind == "FlatIndex":
            return IntV(self.flatten(self.buffer(e.buffer), e.index))
        if kind == "FuncCallExpr":
            args = [self.as_f32(self.expr(a)) for a in e.args]
            r = self.b.reg("f32")
            ops.emit_op(self.b, e.name, args, r)
            return FloatV(r)
        raise NotImplementedError(f"expression {kind} is not supported")

    def as_f32(self, v) -> str:
        if isinstance(v, FloatV):
            return v.reg
        if isinstance(v, IntV):
            return self.int_to_f32(v.lin)
        r = self.b.reg("f32")  # a predicate used as a number
        if isinstance(v, bool):
            self.b.ins("mov.f32", r, f32_imm(1.0 if v else 0.0))
        else:
            self.b.ins("selp.f32", r, f32_imm(1.0), f32_imm(0.0), v)
        return r

    def as_int(self, v) -> Lin:
        if isinstance(v, IntV):
            return v.lin
        raise TypeError(f"expected an integer expression, got {v!r}")

    def pred(self, v) -> str | bool:
        if isinstance(v, (bool, str)):
            return v
        if isinstance(v, IntV):
            return self.ints.compare("!=", v.lin, Lin())
        p = self.b.reg("pred")
        self.b.ins("setp.ne.f32", p, v.reg, f32_imm(0.0))
        return p

    def binary(self, op: str, a, c):
        if op in ("&&", "||"):
            return self._logical(op, self.pred(a), self.pred(c))
        if isinstance(a, IntV) and isinstance(c, IntV):
            return self._int_binary(op, a.lin, c.lin)
        fa, fc = self.as_f32(a), self.as_f32(c)
        if op in _FLOAT_CMP:
            p = self.b.reg("pred")
            self.b.ins(f"setp.{_FLOAT_CMP[op]}.f32", p, fa, fc)
            return p
        if op not in _FLOAT_ARITH:
            raise NotImplementedError(f"float operator {op!r}")
        r = self.b.reg("f32")
        ops.emit_op(self.b, _FLOAT_ARITH[op], [fa, fc], r)
        return FloatV(r)

    def _logical(self, op: str, pa, pc):
        if isinstance(pa, bool) or isinstance(pc, bool):
            const, other = (pa, pc) if isinstance(pa, bool) else (pc, pa)
            if op == "&&":
                return other if const else False
            return True if const else other
        p = self.b.reg("pred")
        self.b.ins("and.pred" if op == "&&" else "or.pred", p, pa, pc)
        return p

    def _int_binary(self, op: str, x: Lin, y: Lin):
        if op == "+":
            return IntV(x + y)
        if op == "-":
            return IntV(x - y)
        if op == "*":
            return IntV(self.ints.mul(x, y))
        if op in ("/", "//"):
            return IntV(self.ints.divmod_dyn(x, y, "div"))
        if op == "%":
            return IntV(self.ints.divmod_dyn(x, y, "mod"))
        if op in _FLOAT_CMP:
            return self.ints.compare(op, x, y)
        if op in ("^", "&", "|"):
            return IntV(self.ints.bitop(op, x, y))
        raise NotImplementedError(f"integer operator {op!r}")
