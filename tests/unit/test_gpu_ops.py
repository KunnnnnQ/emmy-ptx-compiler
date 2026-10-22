"""Numerical accuracy of every elementwise op lowering, on a local GPU (skipped without one).

Each op becomes a tiny kernel `out[i] = op(a[i], b[i], c[i])` built with ptxc's own
PtxBuilder and ops.emit_op, run through the CUDA driver, and compared with float64 numpy.
"""

import math

import numpy as np
import pytest

from ptxc import ops
from ptxc.ptx import PtxBuilder

try:
    from cudadrv import Gpu

    GPU = Gpu()
except Exception:  # no driver, no device, or a CPU-only CI runner
    GPU = None

pytestmark = pytest.mark.skipif(GPU is None, reason="no CUDA device")

N = 4096
RNG = np.random.default_rng(0)
A = (RNG.standard_normal(N) * 3).astype(np.float32)
B = (RNG.standard_normal(N) * 3).astype(np.float32)
C = (RNG.standard_normal(N) * 3).astype(np.float32)


def _erf(x):
    return np.vectorize(math.erf)(x)


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


# op -> (numpy reference on float64 inputs, input transform keeping it in its domain, rtol, atol)
CASES = {
    "add": (lambda a, b: a + b, None, 0, 0),
    "subtract": (lambda a, b: a - b, None, 0, 0),
    "multiply": (lambda a, b: a * b, None, 0, 0),
    "divide": (lambda a, b: a / b, None, 0, 0),
    "maximum": (np.maximum, None, 0, 0),
    "minimum": (np.minimum, None, 0, 0),
    "sqrt": (np.sqrt, np.abs, 0, 0),
    "reciprocal": (lambda a: 1 / a, None, 0, 0),
    "abs": (np.abs, None, 0, 0),
    "negative": (np.negative, None, 0, 0),
    "square": (lambda a: a * a, None, 0, 0),
    "relu": (lambda a: np.maximum(a, 0), None, 0, 0),
    "copy": (lambda a: a, None, 0, 0),
    "rsqrt": (lambda a: 1 / np.sqrt(a), lambda a: np.abs(a) + 0.1, 2e-6, 0),
    "exp": (np.exp, None, 2e-6, 1e-30),
    "log": (np.log, lambda a: np.abs(a) + 0.1, 2e-6, 2e-7),
    "pow": (np.power, lambda a: np.abs(a) + 0.1, 1e-5, 1e-30),
    "sigmoid": (_sigmoid, None, 2e-6, 1e-7),
    "silu": (lambda a: a * _sigmoid(a), None, 2e-6, 1e-7),
    "tanh": (np.tanh, None, 2e-6, 2e-7),
    "erf": (_erf, None, 1e-6, 3e-7),
    "gelu": (lambda a: 0.5 * a * (1 + _erf(a / math.sqrt(2))), None, 1e-6, 1e-6),
    "gelu_tanh": (
        lambda a: 0.5 * a * (1 + np.tanh(math.sqrt(2 / math.pi) * (a + 0.044715 * a**3))),
        None,
        1e-5,
        1e-6,
    ),
    "softplus": (lambda a: np.log1p(np.exp(a)), None, 2e-6, 2e-7),
    "sin": (np.sin, None, 0, 2e-6),
    "cos": (np.cos, None, 0, 2e-6),
    "greater": (lambda a, b: (a > b).astype(np.float64), None, 0, 0),
    "less_equal": (lambda a, b: (a <= b).astype(np.float64), None, 0, 0),
    "where": (lambda c, a, b: np.where(c != 0, a, b), None, 0, 0),
}


def _kernel(op: str, arity: int) -> str:
    b = PtxBuilder()
    tid, ctaid, i = b.reg("s32"), b.reg("s32"), b.reg("s32")
    b.ins("mov.u32", tid, "%tid.x")
    b.ins("mov.u32", ctaid, "%ctaid.x")
    b.ins("mad.lo.s32", i, ctaid, "256", tid)
    off = b.reg("u64")
    b.ins("mul.wide.s32", off, i, "4")
    regs = []
    for name in ("a", "b", "c", "out"):
        raw, ptr, addr = b.reg("u64"), b.reg("u64"), b.reg("u64")
        b.ins("ld.param.u64", raw, f"[param_{name}]")
        b.ins("cvta.to.global.u64", ptr, raw)
        b.ins("add.s64", addr, ptr, off)
        regs.append(addr)
    args = []
    for addr in regs[:arity]:
        v = b.reg("f32")
        b.ins("ld.global.f32", v, f"[{addr}]")
        args.append(v)
    out = b.reg("f32")
    ops.emit_op(b, op, args, out)
    b.ins("st.global.f32", f"[{regs[3]}]", out)
    return b.render("k", [f".param .u64 param_{n}" for n in ("a", "b", "c", "out")], target="sm_89", maxntid=None)


@pytest.mark.parametrize("op", sorted(CASES))
def test_op_accuracy(op):
    ref, domain, rtol, atol = CASES[op]
    arity = ops.ARITY[op]
    inputs = [A, B, C][:arity]
    if domain is not None:
        inputs = [domain(x).astype(np.float32) for x in inputs]
    if op == "where":
        inputs[0] = (inputs[0] > 0).astype(np.float32)
    fn = GPU.function(GPU.load(_kernel(op, arity)), "k")
    bufs = [GPU.upload(x) for x in inputs] + [GPU.zeros(4)] * (3 - arity)
    out = GPU.zeros(N * 4)
    GPU.launch(fn, (N // 256,), (256,), 0, [*bufs, out])
    got = GPU.download(out, np.float32, (N,)).astype(np.float64)
    GPU.free_all()
    want = ref(*[x.astype(np.float64) for x in inputs])
    want32 = want.astype(np.float32).astype(np.float64)  # what a correctly rounded f32 op returns
    if rtol == 0 and atol == 0:
        np.testing.assert_array_equal(got, want32, err_msg=f"{op} should be exact")
    else:
        np.testing.assert_allclose(got, want, rtol=rtol, atol=atol, err_msg=op)
