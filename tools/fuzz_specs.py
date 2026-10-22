"""Random problem specs across the five categories, for fuzzing the compiler.

`build(seed, count)` returns specs in the task generator's format (id, code, ref,
rtol, atol, knobs). Every spec is a torch expression written as a lambda applied to
`torch.randn` inputs (a named intermediate would become a graph input) plus the same
computation in float64 numpy as the reference.
"""

from __future__ import annotations

import math
import random

import numpy as np

f32 = np.float32


def _d(x):
    return x.astype(np.float64)


def _erf(x):
    return np.vectorize(math.erf)(x)


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# pointwise expression trees: (torch text, numpy function of the input arrays)
# ---------------------------------------------------------------------------
UNARY = [
    ("neg", "(-{a})", lambda a: -a),
    ("abs", "{a}.abs()", np.abs),
    ("relu", "torch.relu({a})", lambda a: np.maximum(a, 0)),
    ("sigmoid", "torch.sigmoid({a})", _sigmoid),
    ("tanh", "torch.tanh({a})", np.tanh),
    ("exp", "torch.exp(torch.clamp({a}, -10.0, 10.0))", lambda a: np.exp(np.clip(a, -10, 10))),
    ("sqrt", "torch.sqrt({a}.abs())", lambda a: np.sqrt(np.abs(a))),
    ("rsqrt", "torch.rsqrt({a}.abs() + 1.0)", lambda a: 1 / np.sqrt(np.abs(a) + 1)),
    ("log", "torch.log({a}.abs() + 1.0)", lambda a: np.log(np.abs(a) + 1)),
    ("square", "torch.square({a})", lambda a: a * a),
    ("silu", "torch.nn.functional.silu({a})", lambda a: a * _sigmoid(a)),
    ("gelu", "torch.nn.functional.gelu({a})", lambda a: 0.5 * a * (1 + _erf(a / math.sqrt(2)))),
    ("erf", "torch.erf({a})", _erf),
    ("sin", "torch.sin({a})", np.sin),
    ("cos", "torch.cos({a})", np.cos),
    ("clamp", "torch.clamp({a}, -0.5, 0.5)", lambda a: np.clip(a, -0.5, 0.5)),
    ("leaky", "torch.where({a} > 0, {a}, {a} * 0.1)", lambda a: np.where(a > 0, a, a * 0.1)),
    ("scale", "({a} * 1.5)", lambda a: a * 1.5),
    ("shift", "({a} + 0.25)", lambda a: a + 0.25),
]
BINARY = [
    ("add", "({a} + {b})", lambda a, b: a + b),
    ("sub", "({a} - {b})", lambda a, b: a - b),
    ("mul", "({a} * {b})", lambda a, b: a * b),
    ("div", "({a} / ({b}.abs() + 1.0))", lambda a, b: a / (np.abs(b) + 1)),
    ("max", "torch.maximum({a}, {b})", np.maximum),
    ("min", "torch.minimum({a}, {b})", np.minimum),
]
DIMS = [1, 3, 7, 16, 31, 64, 100, 128, 255, 256, 1000, 1024, 4096]


def _shape(rng: random.Random, max_numel: int = 1 << 20) -> tuple[int, ...]:
    while True:
        shape = tuple(rng.choice(DIMS) for _ in range(rng.choice([1, 1, 2, 2, 3])))
        if math.prod(shape) <= max_numel and math.prod(shape) >= 2:
            return shape


def _tree(rng: random.Random, leaves: list[str], depth: int):
    """A random expression over leaf names: (torch text, fn(env) -> array)."""
    if depth == 0 or rng.random() < 0.25:
        name = rng.choice(leaves)
        return name, lambda env, n=name: env[n]
    if rng.random() < 0.55:
        _, text, fn = rng.choice(UNARY)
        a_txt, a_fn = _tree(rng, leaves, depth - 1)
        return text.format(a=a_txt), lambda env: fn(a_fn(env))
    _, text, fn = rng.choice(BINARY)
    a_txt, a_fn = _tree(rng, leaves, depth - 1)
    b_txt, b_fn = _tree(rng, leaves, depth - 1)
    return text.format(a=a_txt, b=b_txt), lambda env: fn(a_fn(env), b_fn(env))


def _lambda_spec(pid, args: list[tuple[str, tuple]], body: str, ref, rtol, atol, knobs=""):
    params = ", ".join(n for n, _ in args)
    inputs = ", ".join(f"torch.randn({', '.join(map(str, s))})" for _, s in args)
    spec = dict(id=pid, code=f"(lambda {params}: {body})({inputs})", ref=ref, rtol=rtol, atol=atol)
    if knobs:
        spec["knobs"] = knobs
    return spec


def pointwise(rng: random.Random, pid: str) -> dict:
    shape = _shape(rng)
    nin = rng.choice([1, 1, 2, 2, 3])
    names = [f"x{i}" for i in range(nin)]
    shapes = {n: shape for n in names}
    if nin > 1 and len(shape) > 1 and rng.random() < 0.3:  # broadcast a trailing-dim vector
        shapes[names[-1]] = shape[-1:]
    while True:
        text, fn = _tree(rng, names, rng.choice([1, 2, 3]))
        if all(n in text for n in names):  # an unused input would not be traced
            break
    ref = lambda *xs: [fn(dict(zip(names, map(_d, xs), strict=True))).astype(f32)]
    return _lambda_spec(pid, [(n, shapes[n]) for n in names], text, ref, 1e-4, 1e-4)


REDUCE_KNOBS = [
    "WORK=t128,REDUCE=coop",
    "WORK=t128,REDUCE=coop",
    "WORK=t32,REDUCE=coop",
    "WORK=t64,REDUCE=coop",
    "WORK=t256,REDUCE=coop",
    "WORK=t512,REDUCE=coop",
    "WORK=t128,REDUCE=coop-t",
    "WORK=t128,REDUCE=coop/r2",
    "WORK=t128,REDUCE=coop/r4",
    "WORK=t128,REDUCE=g2a/coop",
    "WORK=t128,REDUCE=g4k/coop",
    "",
]
PRE = [
    ("{x}", lambda x: x),
    ("({x} * {x})", lambda x: x * x),
    ("{x}.abs()", np.abs),
    ("torch.exp(torch.tanh({x}))", lambda x: np.exp(np.tanh(x))),
]


def reduction(rng: random.Random, pid: str) -> dict:
    shape = _shape(rng, 1 << 18)
    axis = rng.randrange(len(shape))
    op = rng.choice(["sum", "mean", "amax"])
    pre_txt, pre_fn = rng.choice(PRE)
    body = f"{pre_txt.format(x='x')}.{op}({axis if len(shape) > 1 else ''})"
    np_op = {"sum": np.sum, "mean": np.mean, "amax": np.max}[op]
    n = shape[axis]
    ref = lambda x: [np.asarray(np_op(pre_fn(_d(x)), axis=axis if len(shape) > 1 else None)).astype(f32)]
    return _lambda_spec(pid, [("x", shape)], body, ref, 1e-4, 1e-4 * max(1.0, math.sqrt(n)), rng.choice(REDUCE_KNOBS))


MM_DIMS = [1, 5, 16, 33, 64, 96, 128, 200, 256, 512]


def matmul(rng: random.Random, pid: str) -> dict:
    while True:
        m, k, n = (rng.choice(MM_DIMS) for _ in range(3))
        if m * k * n <= 1 << 25 and m * n >= 2:
            break
    form = rng.choice(["mm", "mm", "bmm", "linear", "linear_bias", "relu", "add_vec"])
    atol = 1e-3 * max(1.0, math.sqrt(k) / 8)
    if form == "bmm":
        b = rng.choice([2, 3, 4])
        return _lambda_spec(
            pid,
            [("a", (b, m, k)), ("c", (b, k, n))],
            "torch.bmm(a, c)",
            lambda a, c: [(_d(a) @ _d(c)).astype(f32)],
            1e-3,
            atol,
        )
    if form == "linear":
        return _lambda_spec(
            pid,
            [("x", (m, k)), ("w", (n, k))],
            "torch.nn.functional.linear(x, w)",
            lambda x, w: [(_d(x) @ _d(w).T).astype(f32)],
            1e-3,
            atol,
        )
    if form == "linear_bias":
        return _lambda_spec(
            pid,
            [("x", (m, k)), ("w", (n, k)), ("b", (n,))],
            "torch.nn.functional.linear(x, w, b)",
            lambda x, w, b: [(_d(x) @ _d(w).T + _d(b)).astype(f32)],
            1e-3,
            atol,
        )
    if form == "relu":
        return _lambda_spec(
            pid,
            [("a", (m, k)), ("c", (k, n))],
            "torch.relu(a @ c)",
            lambda a, c: [np.maximum(_d(a) @ _d(c), 0).astype(f32)],
            1e-3,
            atol,
        )
    if form == "add_vec":
        return _lambda_spec(
            pid,
            [("a", (m, k)), ("c", (k, n)), ("v", (n,))],
            "a @ c + v",
            lambda a, c, v: [(_d(a) @ _d(c) + _d(v)).astype(f32)],
            1e-3,
            atol,
        )
    return _lambda_spec(
        pid, [("a", (m, k)), ("c", (k, n))], "a @ c", lambda a, c: [(_d(a) @ _d(c)).astype(f32)], 1e-3, atol
    )


ROW_COLS = [32, 64, 100, 128, 256, 300, 512, 1000, 1024, 2048, 4096]
ROW_ROWS = [1, 3, 16, 64, 128]


def fused(rng: random.Random, pid: str) -> dict:
    cols, rows = rng.choice(ROW_COLS), rng.choice(ROW_ROWS)
    shape = (rows, cols) if rng.random() < 0.7 else (2, rows, cols)
    kind = rng.choice(["softmax", "softmax_t", "rms_pow", "rms_w", "l2", "layernorm", "center", "sumnorm"])
    x = [("x", shape)]
    if kind == "softmax":
        return _lambda_spec(pid, x, "torch.softmax(x, dim=-1)", lambda x: [_softmax(_d(x)).astype(f32)], 1e-3, 1e-5)
    if kind == "softmax_t":
        t = rng.choice([0.125, 0.5, 2.0])
        return _lambda_spec(
            pid, x, f"torch.softmax(x * {t}, dim=-1)", lambda x: [_softmax(_d(x) * t).astype(f32)], 1e-3, 1e-5
        )
    if kind == "rms_pow":
        return _lambda_spec(
            pid,
            x,
            "torch.mul(x, torch.pow(torch.mean(torch.pow(x, 2), -1, keepdim=True) + 1e-6, -0.5))",
            lambda x: [(_d(x) / np.sqrt((_d(x) ** 2).mean(-1, keepdims=True) + 1e-6)).astype(f32)],
            1e-3,
            1e-4,
        )
    if kind == "rms_w":
        return _lambda_spec(
            pid,
            [*x, ("w", (cols,))],
            "x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-5) * w",
            lambda x, w: [(_d(x) / np.sqrt((_d(x) ** 2).mean(-1, keepdims=True) + 1e-5) * _d(w)).astype(f32)],
            1e-3,
            1e-4,
        )
    if kind == "l2":
        return _lambda_spec(
            pid,
            x,
            "x / torch.sqrt((x * x).sum(-1, keepdim=True))",
            lambda x: [(_d(x) / np.sqrt((_d(x) ** 2).sum(-1, keepdims=True))).astype(f32)],
            1e-3,
            1e-4,
        )
    if kind == "layernorm":
        return _lambda_spec(
            pid,
            x,
            "(lambda d: d / torch.sqrt((d * d).mean(-1, keepdim=True) + 1e-5))(x - x.mean(-1, keepdim=True))",
            lambda x: [
                ((_d(x) - _d(x).mean(-1, keepdims=True)) / np.sqrt(_d(x).var(-1, keepdims=True) + 1e-5)).astype(f32)
            ],
            1e-3,
            1e-4,
        )
    if kind == "center":
        return _lambda_spec(
            pid,
            x,
            "x - x.mean(-1, keepdim=True)",
            lambda x: [(_d(x) - _d(x).mean(-1, keepdims=True)).astype(f32)],
            1e-3,
            1e-4,
        )
    return _lambda_spec(
        pid,
        x,
        "x.abs() / x.abs().sum(-1, keepdim=True)",
        lambda x: [(np.abs(_d(x)) / np.abs(_d(x)).sum(-1, keepdims=True)).astype(f32)],
        1e-3,
        1e-5,
    )


def attention(rng: random.Random, pid: str) -> dict:
    lq, lk = rng.choice([1, 8, 16, 33, 64, 128]), rng.choice([16, 48, 64, 100, 128, 256])
    d, dv = rng.choice([8, 16, 32, 64]), rng.choice([8, 16, 32, 64])
    scale = round(1 / math.sqrt(d), 6)
    lead = (rng.choice([2, 4]),) if rng.random() < 0.3 else ()
    args = [("q", (*lead, lq, d)), ("kt", (*lead, d, lk)), ("v", (*lead, lk, dv))]
    ref = lambda q, kt, v: [(_softmax(_d(q) @ _d(kt) * scale) @ _d(v)).astype(f32)]
    return _lambda_spec(pid, args, f"torch.softmax(q @ kt * {scale}, dim=-1) @ v", ref, 1e-3, 1e-4)


CATEGORIES = [("p", pointwise, 30), ("r", reduction, 25), ("m", matmul, 15), ("f", fused, 20), ("a", attention, 10)]


def build(seed: int, count: int) -> list[dict]:
    rng = random.Random(seed)
    kinds = [c for c, _, w in CATEGORIES for _ in range(w)]
    makers = {c: fn for c, fn, _ in CATEGORIES}
    specs = []
    for i in range(count):
        c = rng.choice(kinds)
        specs.append(makers[c](rng, f"z{c}{seed:03d}_{i:04d}"))
    return specs
