"""Elementwise f32 operations -> PTX instruction sequences.

`emit_op(b, name, args, dst)` writes `dst = name(*args)`. Names are Emmy's
ElementwiseImpl names (numpy spellings: add, multiply, maximum, exp, ...).

Precision policy: arithmetic is IEEE round-to-nearest (`.rn`), which also stops
ptxas from contracting a separate multiply and add into an fma, so pointwise
kernels match numpy bit for bit. Transcendentals use the hardware's fast
approximations (ex2/lg2/rsqrt.approx), the same choice Emmy's own fast-math
build makes; reductions and fused kernels carry a tolerance for it.
"""

from __future__ import annotations

import math

from ptxc.ptx import PtxBuilder, f32_imm

ONE = f32_imm(1.0)
ZERO = f32_imm(0.0)
LOG2E = f32_imm(math.log2(math.e))
LN2 = f32_imm(math.log(2.0))

_BINARY = {
    "add": "add.rn.f32",
    "subtract": "sub.rn.f32",
    "multiply": "mul.rn.f32",
    "divide": "div.rn.f32",
    "maximum": "max.f32",
    "minimum": "min.f32",
    "fmax": "max.f32",
    "fmin": "min.f32",
}
_UNARY = {
    "sqrt": "sqrt.rn.f32",
    "rsqrt": "rsqrt.approx.ftz.f32",
    "reciprocal": "rcp.rn.f32",
    "abs": "abs.f32",
    "fabs": "abs.f32",
    "negative": "neg.f32",
    "copy": "mov.f32",
    "pad": "mov.f32",
    "sin": "sin.approx.ftz.f32",
    "cos": "cos.approx.ftz.f32",
}
_COMPARE = {
    "equal": "eq",
    "not_equal": "ne",
    "greater": "gt",
    "less": "lt",
    "greater_equal": "ge",
    "less_equal": "le",
}

# Names Emmy uses for the same op in different places (reduce registry, intrinsics).
_ALIASES = {"sum": "add", "prod": "multiply", "amax": "maximum", "max": "maximum", "min": "minimum", "exp_fast": "exp"}

# Arity of every supported op, for argument-count checks.
ARITY = {
    **{k: 2 for k in _BINARY},
    **{k: 1 for k in _UNARY},
    **{k: 2 for k in _COMPARE},
    "exp": 1,
    "log": 1,
    "tanh": 1,
    "erf": 1,
    "relu": 1,
    "sigmoid": 1,
    "silu": 1,
    "gelu": 1,
    "gelu_tanh": 1,
    "softplus": 1,
    "square": 1,
    "pow": 2,
    "where": 3,
    "bitwise_and": 2,
    "bitwise_or": 2,
    "bitwise_not": 1,
}


def canonical(name: str) -> str:
    return _ALIASES.get(name, name)


def supports(name: str) -> bool:
    return canonical(name) in ARITY


def emit_op(b: PtxBuilder, name: str, args: list[str], dst: str) -> None:
    name = canonical(name)
    if name not in ARITY:
        raise NotImplementedError(f"elementwise op {name!r} has no PTX lowering")
    if len(args) != ARITY[name]:
        raise ValueError(f"op {name!r} takes {ARITY[name]} arguments, got {len(args)}")

    def tmp() -> str:
        return b.reg("f32")

    if name in _BINARY:
        b.ins(_BINARY[name], dst, args[0], args[1])
    elif name in _UNARY:
        b.ins(_UNARY[name], dst, args[0])
    elif name == "square":
        b.ins("mul.rn.f32", dst, args[0], args[0])
    elif name == "relu":
        b.ins("max.f32", dst, args[0], ZERO)
    elif name == "exp":
        _exp(b, args[0], dst)
    elif name == "log":
        t = tmp()
        b.ins("lg2.approx.ftz.f32", t, args[0])
        b.ins("mul.f32", dst, t, LN2)
    elif name == "pow":
        t = tmp()
        b.ins("lg2.approx.ftz.f32", t, args[0])
        b.ins("mul.f32", t, t, args[1])
        b.ins("ex2.approx.ftz.f32", dst, t)
    elif name == "sigmoid":
        _sigmoid(b, args[0], dst)
    elif name == "silu":
        s = tmp()
        _sigmoid(b, args[0], s)
        b.ins("mul.rn.f32", dst, args[0], s)
    elif name == "tanh":
        _tanh(b, args[0], dst)
    elif name == "gelu_tanh":
        # 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))
        x = args[0]
        t, u = tmp(), tmp()
        b.ins("mul.f32", t, x, x)
        b.ins("fma.rn.f32", t, t, f32_imm(0.044715), ONE)
        b.ins("mul.f32", t, t, x)
        b.ins("mul.f32", t, t, f32_imm(math.sqrt(2.0 / math.pi)))
        _tanh(b, t, u)
        b.ins("add.f32", u, u, ONE)
        b.ins("mul.f32", u, u, x)
        b.ins("mul.f32", dst, u, f32_imm(0.5))
    elif name == "erf":
        _erf(b, args[0], dst)
    elif name == "gelu":
        # 0.5 x (1 + erf(x / sqrt(2)))
        t = tmp()
        b.ins("mul.f32", t, args[0], f32_imm(1.0 / math.sqrt(2.0)))
        _erf(b, t, t)
        b.ins("add.f32", t, t, ONE)
        b.ins("mul.f32", t, t, args[0])
        b.ins("mul.f32", dst, t, f32_imm(0.5))
    elif name == "softplus":
        # max(x, 0) + log(1 + exp(-|x|)): never overflows
        t, m = tmp(), tmp()
        b.ins("abs.f32", t, args[0])
        b.ins("neg.f32", t, t)
        _exp(b, t, t)
        b.ins("add.f32", t, t, ONE)
        b.ins("lg2.approx.ftz.f32", t, t)
        b.ins("mul.f32", t, t, LN2)
        b.ins("max.f32", m, args[0], ZERO)
        b.ins("add.f32", dst, m, t)
    elif name in _COMPARE:
        p = b.reg("pred")
        b.ins(f"setp.{_COMPARE[name]}.f32", p, args[0], args[1])
        b.ins("selp.f32", dst, ONE, ZERO, p)
    elif name == "where":
        p = b.reg("pred")
        b.ins("setp.ne.f32", p, args[0], ZERO)
        b.ins("selp.f32", dst, args[1], args[2], p)
    elif name in ("bitwise_and", "bitwise_or"):
        # Emmy's f32 masks: logical and/or on "nonzero"
        p, q = b.reg("pred"), b.reg("pred")
        b.ins("setp.ne.f32", p, args[0], ZERO)
        b.ins("setp.ne.f32", q, args[1], ZERO)
        b.ins("and.pred" if name == "bitwise_and" else "or.pred", p, p, q)
        b.ins("selp.f32", dst, ONE, ZERO, p)
    elif name == "bitwise_not":
        p = b.reg("pred")
        b.ins("setp.eq.f32", p, args[0], ZERO)
        b.ins("selp.f32", dst, ONE, ZERO, p)
    else:  # pragma: no cover - ARITY and the branches above must agree
        raise NotImplementedError(name)


def _exp(b: PtxBuilder, x: str, dst: str) -> None:
    # e^x = 2^(x * log2 e)
    t = b.reg("f32")
    b.ins("mul.f32", t, x, LOG2E)
    b.ins("ex2.approx.ftz.f32", dst, t)


def _sigmoid(b: PtxBuilder, x: str, dst: str) -> None:
    # 1 / (1 + e^-x)
    t = b.reg("f32")
    b.ins("mul.f32", t, x, f32_imm(-math.log2(math.e)))
    b.ins("ex2.approx.ftz.f32", t, t)
    b.ins("add.f32", t, t, ONE)
    b.ins("rcp.rn.f32", dst, t)


def _tanh(b: PtxBuilder, x: str, dst: str) -> None:
    # tanh x = 1 - 2 / (e^(2x) + 1); saturates cleanly to +-1 at both ends.
    # (tanh.approx.f32 is only good to ~2^-11, too coarse for tight tolerances.)
    t = b.reg("f32")
    b.ins("mul.f32", t, x, f32_imm(2.0 * math.log2(math.e)))
    b.ins("ex2.approx.ftz.f32", t, t)
    b.ins("add.f32", t, t, ONE)
    b.ins("rcp.rn.f32", t, t)
    b.ins("fma.rn.f32", dst, t, f32_imm(-2.0), ONE)


def _erf(b: PtxBuilder, x: str, dst: str) -> None:
    # Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7):
    # erf|x| = 1 - poly(t) e^(-x^2),  t = 1 / (1 + p|x|); the sign of x is restored at the end.
    p_coef = 0.3275911
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    ax, t, poly, e = b.reg("f32"), b.reg("f32"), b.reg("f32"), b.reg("f32")
    b.ins("abs.f32", ax, x)
    b.ins("fma.rn.f32", t, ax, f32_imm(p_coef), ONE)
    b.ins("rcp.rn.f32", t, t)
    b.ins("fma.rn.f32", poly, t, f32_imm(a5), f32_imm(a4))
    for a in (a3, a2, a1):
        b.ins("fma.rn.f32", poly, poly, t, f32_imm(a))
    b.ins("mul.f32", poly, poly, t)
    b.ins("mul.f32", e, ax, ax)
    b.ins("neg.f32", e, e)
    _exp(b, e, e)
    b.ins("fma.rn.f32", e, poly, e, f32_imm(0.0))
    b.ins("sub.f32", e, ONE, e)
    b.ins("copysign.f32", dst, x, e)
