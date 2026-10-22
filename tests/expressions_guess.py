"""A guess at the hidden set: the workloads Emmy itself is built and tested for.

Sources: the expressions in Emmy's own docs and e2e tests (RMSNorm in its
`pow(mean(pow(x, 2)) + eps, -0.5)` spelling, `sigmoid * tanh` reduced, batch-1
`F.linear`, `torch.matmul(1x1024, 1024x3072)`), the LLM-inference ops Emmy's
recipes serve (SwiGLU, residual + RMSNorm, affine layer norm, 4-D softmax,
multi-head attention), and schedule knobs the open set never varies (WORK widths
t32..t512, REDUCE coop-t / r2 / g4a / g2k), plus fp16 buffers.

Generate with the task's generator, like tests/expressions_extra.py.
"""

import numpy as np

f32, f16 = np.float32, np.float16


def _d(x):
    return x.astype(np.float64)


def _softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _red(work="t128", reduce="coop"):
    return f"WORK={work},REDUCE={reduce}"


def _attn(q, kt, v, scale):
    return [(_softmax(_d(q) @ _d(kt) * scale) @ _d(v)).astype(f32)]


OPEN = [
    # ---- pointwise --------------------------------------------------------
    dict(
        id="gp01_swiglu",
        code="x = torch.randn(64, 256); y = torch.randn(64, 256); torch.nn.functional.silu(x) * y",
        ref=lambda x, y: [(_d(x) * _sigmoid(_d(x)) * _d(y)).astype(f32)],
        rtol=1e-5,
        atol=1e-6,
    ),
    dict(
        id="gp02_gelu_tanh",
        code="torch.nn.functional.gelu(torch.randn(4096), approximate='tanh')",
        ref=lambda x: [(0.5 * _d(x) * (1 + np.tanh(np.sqrt(2 / np.pi) * (_d(x) + 0.044715 * _d(x) ** 3)))).astype(f32)],
        rtol=1e-4,
        atol=1e-5,
    ),
    dict(
        id="gp03_sig_tanh",
        code="x = torch.randn(4, 64); torch.sigmoid(x) * torch.tanh(x)",
        ref=lambda x: [(_sigmoid(_d(x)) * np.tanh(_d(x))).astype(f32)],
        rtol=1e-5,
        atol=1e-6,
    ),
    dict(
        id="gp04_residual_scale",
        code="x = torch.randn(2, 16, 64); r = torch.randn(2, 16, 64); w = torch.randn(64); (x + r) * w",
        ref=lambda x, r, w: [(x + r) * w],
        rtol=0,
        atol=0,
    ),
    dict(
        id="gp05_f16_add",
        code="torch.randn(1024, dtype=torch.float16) + torch.randn(1024, dtype=torch.float16)",
        ref=lambda a, b: [(a + b).astype(f32)],
        rtol=1e-3,
        atol=1e-3,
    ),
    dict(
        id="gp06_scalar_chain",
        code="x = torch.randn(4096); (x * 0.5 + 0.25) * 3.0 - 1.0",
        ref=lambda x: [(x * f32(0.5) + f32(0.25)) * f32(3.0) - f32(1.0)],
        rtol=1e-6,
        atol=1e-6,
    ),
    dict(
        id="gp07_pow",
        code="torch.pow(torch.randn(2048).abs() + 0.5, 1.5)",
        ref=lambda x: [((np.abs(_d(x)) + 0.5) ** 1.5).astype(f32)],
        rtol=1e-4,
        atol=1e-5,
    ),
    dict(
        id="gp08_rsqrt_eps",
        code="1.0 / torch.sqrt(torch.randn(1024).abs() + 1e-3)",
        ref=lambda x: [(1 / np.sqrt(np.abs(_d(x)) + 1e-3)).astype(f32)],
        rtol=1e-5,
        atol=1e-6,
    ),
    dict(id="gp09_large", code="torch.randn(1 << 20) * 2.0", ref=lambda x: [x * f32(2.0)], rtol=0, atol=0),
    dict(id="gp10_tiny", code="torch.randn(7) + torch.randn(7)", ref=lambda a, b: [a + b], rtol=0, atol=0),
    dict(
        id="gp11_f16_silu",
        code="x = torch.randn(2048, dtype=torch.float16); torch.nn.functional.silu(x)",
        ref=lambda x: [(_d(x) * _sigmoid(_d(x))).astype(f16).astype(f32)],
        rtol=2e-3,
        atol=2e-3,
    ),
    # ---- reductions: other block widths and reduce plans ------------------
    dict(
        id="gr01_t256",
        code="torch.randn(4096).sum()",
        ref=lambda x: [_d(x).sum().astype(f32)],
        rtol=1e-4,
        atol=1e-4,
        knobs=_red("t256"),
    ),
    dict(
        id="gr02_t64_rows",
        code="torch.randn(64, 512).sum(-1)",
        ref=lambda x: [_d(x).sum(-1).astype(f32)],
        rtol=1e-4,
        atol=1e-4,
        knobs=_red("t64"),
    ),
    dict(
        id="gr03_t32_rows",
        code="torch.randn(128, 128).sum(-1)",
        ref=lambda x: [_d(x).sum(-1).astype(f32)],
        rtol=1e-4,
        atol=1e-4,
        knobs=_red("t32"),
    ),
    dict(
        id="gr04_t512_mean",
        code="torch.randn(65536).mean()",
        ref=lambda x: [_d(x).mean().astype(f32)],
        rtol=1e-4,
        atol=1e-4,
        knobs=_red("t512"),
    ),
    dict(
        id="gr05_coop_t",
        code="torch.randn(64, 512).sum(-1)",
        ref=lambda x: [_d(x).sum(-1).astype(f32)],
        rtol=1e-4,
        atol=1e-4,
        knobs=_red("t128", "coop-t"),
    ),
    dict(
        id="gr06_coop_r2",
        code="torch.randn(8192).sum()",
        ref=lambda x: [_d(x).sum().astype(f32)],
        rtol=1e-4,
        atol=1e-4,
        knobs=_red("t128", "coop/r2"),
    ),
    dict(
        id="gr07_grid_atomic",
        code="torch.randn(65536).sum()",
        ref=lambda x: [_d(x).sum().astype(f32)],
        rtol=1e-3,
        atol=1e-3,
        knobs=_red("t128", "g4a/coop"),
    ),
    dict(
        id="gr08_grid_kernel",
        code="torch.randn(32768).sum()",
        ref=lambda x: [_d(x).sum().astype(f32)],
        rtol=1e-3,
        atol=1e-3,
        knobs=_red("t128", "g2k/coop"),
    ),
    dict(
        id="gr09_amax_4d",
        code="torch.randn(2, 4, 8, 256).amax(-1)",
        ref=lambda x: [x.max(-1)],
        rtol=0,
        atol=0,
        knobs=_red(),
    ),
    dict(
        id="gr10_mean_mid_axis",
        code="torch.randn(16, 64, 32).mean(1)",
        ref=lambda x: [_d(x).mean(1).astype(f32)],
        rtol=1e-4,
        atol=1e-4,
        knobs=_red(),
    ),
    dict(
        id="gr11_sig_tanh_sum",
        code="(lambda x: (torch.sigmoid(x) * torch.tanh(x)).sum(-1))(torch.randn(4, 64))",
        ref=lambda x: [(_sigmoid(_d(x)) * np.tanh(_d(x))).sum(-1).astype(f32)],
        rtol=1e-4,
        atol=1e-4,
    ),
    dict(
        id="gr12_amin_rows",
        code="torch.randn(64, 300).amin(-1)",
        ref=lambda x: [x.min(-1)],
        rtol=0,
        atol=0,
        knobs=_red(),
    ),
    dict(
        id="gr13_f16_rowsum",
        code="torch.randn(32, 256, dtype=torch.float16).sum(-1)",
        ref=lambda x: [_d(x).sum(-1).astype(f32)],
        rtol=2e-3,
        atol=2e-2,
        knobs=_red(),
    ),
    # ---- matmul -----------------------------------------------------------
    dict(
        id="gm01_linear_b1",
        code="x = torch.randn(1, 64); w = torch.randn(32, 64); torch.nn.functional.linear(x, w)",
        ref=lambda x, w: [(_d(x) @ _d(w).T).astype(f32)],
        rtol=1e-3,
        atol=1e-3,
    ),
    dict(
        id="gm02_linear_b5",
        code="x = torch.randn(5, 96); w = torch.randn(72, 96); torch.nn.functional.linear(x, w)",
        ref=lambda x, w: [(_d(x) @ _d(w).T).astype(f32)],
        rtol=1e-3,
        atol=1e-3,
    ),
    dict(
        id="gm03_matvec_wide",
        code="torch.matmul(torch.randn(1, 1024), torch.randn(1024, 3072))",
        ref=lambda a, b: [(_d(a) @ _d(b)).astype(f32)],
        rtol=1e-3,
        atol=1e-3,
    ),
    dict(
        id="gm04_bmm",
        code="torch.bmm(torch.randn(4, 64, 32), torch.randn(4, 32, 48))",
        ref=lambda a, b: [(_d(a) @ _d(b)).astype(f32)],
        rtol=1e-3,
        atol=1e-3,
    ),
    dict(
        id="gm05_linear_deep",
        code="x = torch.randn(1, 4096); w = torch.randn(512, 4096); torch.nn.functional.linear(x, w)",
        ref=lambda x, w: [(_d(x) @ _d(w).T).astype(f32)],
        rtol=1e-3,
        atol=2e-3,
    ),
    dict(
        id="gm06_linear_bias",
        code="x = torch.randn(16, 128); w = torch.randn(64, 128); b = torch.randn(64); "
        "torch.nn.functional.linear(x, w, b)",
        ref=lambda x, w, b: [(_d(x) @ _d(w).T + _d(b)).astype(f32)],
        rtol=1e-3,
        atol=1e-3,
    ),
    dict(
        id="gm07_matmul_relu",
        code="torch.relu(torch.randn(128, 64) @ torch.randn(64, 128))",
        ref=lambda a, b: [np.maximum(_d(a) @ _d(b), 0).astype(f32)],
        rtol=1e-3,
        atol=1e-3,
    ),
    # Not here: an fp16 matmul, which Emmy schedules onto tensor cores (RegFragment / mma.sync),
    # a statement family the task excludes from every problem.
    # ---- fused ------------------------------------------------------------
    dict(
        id="gf01_rmsnorm_pow",
        code="x = torch.randn(2,64,256); "
        "torch.mul(x, torch.pow(torch.mean(torch.pow(x,2),-1,keepdim=True)+1e-6, -0.5))",
        ref=lambda x: [(_d(x) / np.sqrt((_d(x) ** 2).mean(-1, keepdims=True) + 1e-6)).astype(f32)],
        rtol=1e-3,
        atol=1e-4,
    ),
    dict(
        id="gf02_softmax_4d",
        code="torch.softmax(torch.randn(2, 4, 16, 128), dim=-1)",
        ref=lambda x: [_softmax(_d(x)).astype(f32)],
        rtol=1e-3,
        atol=1e-5,
    ),
    dict(
        id="gf03_layernorm_affine",
        # a lambda, not `d = x - m`: the tracer turns every named tensor into a graph input
        code="(lambda x, w, b: (lambda d: d / torch.sqrt((d * d).mean(-1, keepdim=True) + 1e-5) * w + b)"
        "(x - x.mean(-1, keepdim=True)))(torch.randn(32, 256), torch.randn(256), torch.randn(256))",
        ref=lambda x, w, b: [
            ((_d(x) - _d(x).mean(-1, keepdims=True)) / np.sqrt(_d(x).var(-1, keepdims=True) + 1e-5) * w + b).astype(f32)
        ],
        rtol=1e-3,
        atol=1e-4,
    ),
    dict(
        id="gf04_residual_rmsnorm",
        code="x = torch.randn(32, 512); r = torch.randn(32, 512); w = torch.randn(512); h = x + r; "
        "h * torch.rsqrt((h * h).mean(-1, keepdim=True) + 1e-6) * w",
        ref=lambda x, r, w: [
            ((_d(x) + _d(r)) / np.sqrt(((_d(x) + _d(r)) ** 2).mean(-1, keepdims=True) + 1e-6) * w).astype(f32)
        ],
        rtol=1e-3,
        atol=1e-4,
    ),
    dict(
        id="gf05_softmax_wide",
        code="torch.softmax(torch.randn(8, 4096), dim=-1)",
        ref=lambda x: [_softmax(_d(x)).astype(f32)],
        rtol=1e-3,
        atol=1e-6,
    ),
    dict(
        id="gf06_logsumexp",
        code="torch.logsumexp(torch.randn(64, 256), dim=-1)",
        ref=lambda x: [np.log(np.exp(_d(x)).sum(-1)).astype(f32)],
        rtol=1e-3,
        atol=1e-4,
    ),
    dict(
        id="gf07_normalize_sum",
        code="(lambda x: x / x.sum(-1, keepdim=True))(torch.randn(64, 128).abs())",
        ref=lambda x: [(np.abs(_d(x)) / np.abs(_d(x)).sum(-1, keepdims=True)).astype(f32)],
        rtol=1e-3,
        atol=1e-5,
    ),
    dict(
        id="gf08_softmax_f16",
        code="torch.softmax(torch.randn(64, 256, dtype=torch.float16), dim=-1)",
        ref=lambda x: [_softmax(_d(x)).astype(f16).astype(f32)],
        rtol=5e-3,
        atol=1e-4,
    ),
    # ---- attention-like ---------------------------------------------------
    dict(
        id="ga01_batched",
        code="q = torch.randn(2, 64, 32); kt = torch.randn(2, 32, 64); v = torch.randn(2, 64, 32); "
        "torch.softmax(q @ kt / 5.656854, dim=-1) @ v",
        ref=lambda q, kt, v: _attn(q, kt, v, 1 / 5.656854),
        rtol=1e-3,
        atol=1e-4,
    ),
    dict(
        id="ga02_heads",
        code="q = torch.randn(4, 32, 16); kt = torch.randn(4, 16, 48); v = torch.randn(4, 48, 16); "
        "torch.softmax(q @ kt * 0.25, dim=-1) @ v",
        ref=lambda q, kt, v: _attn(q, kt, v, 0.25),
        rtol=1e-3,
        atol=1e-4,
    ),
    dict(
        id="ga03_long_kv",
        code="q = torch.randn(16, 64); kt = torch.randn(64, 512); v = torch.randn(512, 64); "
        "torch.softmax(q @ kt / 8.0, dim=-1) @ v",
        ref=lambda q, kt, v: _attn(q, kt, v, 1 / 8.0),
        rtol=1e-3,
        atol=1e-4,
    ),
    dict(
        id="ga04_tall_q",
        code="q = torch.randn(256, 16); kt = torch.randn(16, 64); v = torch.randn(64, 16); "
        "torch.softmax(q @ kt / 4.0, dim=-1) @ v",
        ref=lambda q, kt, v: _attn(q, kt, v, 1 / 4.0),
        rtol=1e-3,
        atol=1e-4,
    ),
]
