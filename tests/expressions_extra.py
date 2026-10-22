"""Extra problems beyond the open set, to test that the compiler generalizes.

Same format as the task repository's tools/expressions_open.py; generate with

    python <task>/tools/generate.py tests/expressions_extra.py --emmy <emmy> --out tests/extra

Ids are prefixed by category like the open set: p (pointwise), r (reductions),
m (matmul), f (fused), a (attention-like). Reductions are pinned to one block
of 128 cooperating threads, as in the open set, unless the id says "unpinned".
"""

import numpy as np

REDUCE_KNOBS = "WORK=t128,REDUCE=coop"
f32 = np.float32


def _d(x):
    return x.astype(np.float64)


def _softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def _erf(x):
    from math import erf

    return np.vectorize(erf)(x)


OPEN = [
    # ---- pointwise --------------------------------------------------------
    dict(id="px01_sub_2d", code="x = torch.randn(64, 128); y = torch.randn(64, 128); x - y",
         ref=lambda x, y: [x - y], rtol=0, atol=0),
    dict(id="px02_exp", code="torch.exp(torch.randn(4096))",
         ref=lambda x: [np.exp(_d(x)).astype(f32)], rtol=1e-5, atol=1e-6),
    dict(id="px03_sigmoid", code="torch.sigmoid(torch.randn(2048))",
         ref=lambda x: [(1 / (1 + np.exp(-_d(x)))).astype(f32)], rtol=1e-5, atol=1e-6),
    dict(id="px04_tanh", code="torch.tanh(torch.randn(2048))",
         ref=lambda x: [np.tanh(_d(x)).astype(f32)], rtol=1e-5, atol=1e-6),
    dict(id="px05_silu", code="x = torch.randn(4096); x * torch.sigmoid(x)",
         ref=lambda x: [(_d(x) / (1 + np.exp(-_d(x)))).astype(f32)], rtol=1e-5, atol=1e-6),
    dict(id="px06_div", code="a = torch.randn(1024); b = torch.randn(1024); a / (b.abs() + 1.0)",
         ref=lambda a, b: [a / (np.abs(b) + f32(1.0))], rtol=1e-6, atol=1e-6),
    dict(id="px07_bias_bcast", code="x = torch.randn(32, 256); b = torch.randn(256); x + b",
         ref=lambda x, b: [x + b], rtol=0, atol=0),
    dict(id="px08_odd_mul", code="torch.randn(1000) * torch.randn(1000)",
         ref=lambda a, b: [a * b], rtol=0, atol=0),
    dict(id="px09_odd_add_const", code="torch.randn(999) + 1.5",
         ref=lambda x: [x + f32(1.5)], rtol=0, atol=0),
    dict(id="px10_maximum", code="torch.maximum(torch.randn(2048), torch.randn(2048))",
         ref=lambda a, b: [np.maximum(a, b)], rtol=0, atol=0),
    dict(id="px11_sqrt_abs", code="torch.sqrt(torch.randn(2048).abs())",
         ref=lambda x: [np.sqrt(np.abs(x))], rtol=1e-6, atol=1e-6),
    dict(id="px12_square_shift", code="x = torch.randn(4096); (x - 0.5) * (x - 0.5)",
         ref=lambda x: [(x - f32(0.5)) * (x - f32(0.5))], rtol=1e-6, atol=1e-6),
    dict(id="px13_leaky_where", code="x = torch.randn(1024); torch.where(x > 0, x, 0.1 * x)",
         ref=lambda x: [np.where(x > 0, x, f32(0.1) * x)], rtol=1e-6, atol=1e-6),
    dict(id="px14_rsqrt", code="torch.rsqrt(torch.randn(1024).abs() + 1.0)",
         ref=lambda x: [(1 / np.sqrt(np.abs(_d(x)) + 1.0)).astype(f32)], rtol=1e-5, atol=1e-6),
    dict(id="px15_log", code="torch.log(torch.randn(1024).abs() + 1.0)",
         ref=lambda x: [np.log(np.abs(_d(x)) + 1.0).astype(f32)], rtol=1e-5, atol=1e-6),
    dict(id="px16_gelu", code="torch.nn.functional.gelu(torch.randn(2048))",
         ref=lambda x: [(0.5 * _d(x) * (1 + _erf(_d(x) / np.sqrt(2.0)))).astype(f32)], rtol=1e-4, atol=1e-5),
    dict(id="px17_3d_mul", code="torch.randn(4, 16, 64) * torch.randn(4, 16, 64)",
         ref=lambda a, b: [a * b], rtol=0, atol=0),
    dict(id="px18_fma3", code="a = torch.randn(512); b = torch.randn(512); c = torch.randn(512); a * b + c",
         ref=lambda a, b, c: [a * b + c], rtol=1e-6, atol=1e-6),
    dict(id="px19_clamp", code="torch.clamp(torch.randn(2048), -0.5, 0.5)",
         ref=lambda x: [np.clip(x, f32(-0.5), f32(0.5))], rtol=0, atol=0),
    dict(id="px20_neg_abs", code="x = torch.randn(3000); -x.abs()",
         ref=lambda x: [-np.abs(x)], rtol=0, atol=0),

    # ---- reductions -------------------------------------------------------
    dict(id="rx01_sum_2048", code="torch.randn(2048).sum()",
         ref=lambda x: [_d(x).sum().astype(f32)], rtol=1e-4, atol=1e-4, knobs=REDUCE_KNOBS),
    dict(id="rx02_sum_odd", code="torch.randn(1000).sum()",
         ref=lambda x: [_d(x).sum().astype(f32)], rtol=1e-4, atol=1e-4, knobs=REDUCE_KNOBS),
    dict(id="rx03_max", code="torch.randn(4096).max()",
         ref=lambda x: [x.max()], rtol=0, atol=0, knobs=REDUCE_KNOBS),
    dict(id="rx04_row_mean", code="torch.randn(32, 1024).mean(-1)",
         ref=lambda x: [_d(x).mean(-1).astype(f32)], rtol=1e-4, atol=1e-4, knobs=REDUCE_KNOBS),
    dict(id="rx05_row_amax", code="torch.randn(128, 256).amax(-1)",
         ref=lambda x: [x.max(-1)], rtol=0, atol=0, knobs=REDUCE_KNOBS),
    dict(id="rx06_row_sum_odd", code="torch.randn(16, 300).sum(-1)",
         ref=lambda x: [_d(x).sum(-1).astype(f32)], rtol=1e-4, atol=1e-4, knobs=REDUCE_KNOBS),
    dict(id="rx07_sum_squares", code="x = torch.randn(8192); (x * x).sum()",
         ref=lambda x: [(_d(x) ** 2).sum().astype(f32)], rtol=1e-4, atol=1e-3, knobs=REDUCE_KNOBS),
    dict(id="rx08_col_sum", code="torch.randn(256, 64).sum(0)",
         ref=lambda x: [_d(x).sum(0).astype(f32)], rtol=1e-4, atol=1e-4, knobs=REDUCE_KNOBS),
    dict(id="rx09_min", code="torch.randn(2048).min()",
         ref=lambda x: [x.min()], rtol=0, atol=0, knobs=REDUCE_KNOBS),
    dict(id="rx10_sum_unpinned", code="torch.randn(65536).sum()",
         ref=lambda x: [_d(x).sum().astype(f32)], rtol=1e-3, atol=1e-3),
    dict(id="rx11_row_sum_short", code="torch.randn(64, 32).sum(-1)",
         ref=lambda x: [_d(x).sum(-1).astype(f32)], rtol=1e-4, atol=1e-4, knobs=REDUCE_KNOBS),
    dict(id="rx12_row_sum_unpinned", code="torch.randn(256, 512).sum(-1)",
         ref=lambda x: [_d(x).sum(-1).astype(f32)], rtol=1e-4, atol=1e-4),

    # ---- matmul -----------------------------------------------------------
    dict(id="mx01_256x128x256", code="torch.randn(256, 128) @ torch.randn(128, 256)",
         ref=lambda a, b: [(_d(a) @ _d(b)).astype(f32)], rtol=1e-3, atol=1e-3),
    dict(id="mx02_64sq", code="torch.randn(64, 64) @ torch.randn(64, 64)",
         ref=lambda a, b: [(_d(a) @ _d(b)).astype(f32)], rtol=1e-3, atol=1e-3),
    dict(id="mx03_odd", code="torch.randn(96, 80) @ torch.randn(80, 48)",
         ref=lambda a, b: [(_d(a) @ _d(b)).astype(f32)], rtol=1e-3, atol=1e-3),
    dict(id="mx04_tall", code="torch.randn(1024, 64) @ torch.randn(64, 32)",
         ref=lambda a, b: [(_d(a) @ _d(b)).astype(f32)], rtol=1e-3, atol=1e-3),
    dict(id="mx05_deep_k", code="torch.randn(32, 1024) @ torch.randn(1024, 128)",
         ref=lambda a, b: [(_d(a) @ _d(b)).astype(f32)], rtol=1e-3, atol=1e-3),
    dict(id="mx06_bias", code="a = torch.randn(128, 128); b = torch.randn(128, 128); c = torch.randn(128); a @ b + c",
         ref=lambda a, b, c: [(_d(a) @ _d(b) + _d(c)).astype(f32)], rtol=1e-3, atol=1e-3),
    dict(id="mx07_1024sq", code="torch.randn(1024, 1024) @ torch.randn(1024, 1024)",
         ref=lambda a, b: [(_d(a) @ _d(b)).astype(f32)], rtol=1e-3, atol=1e-3),

    # ---- fused ------------------------------------------------------------
    dict(id="fx01_layernorm",
         code="x = torch.randn(64, 512); m = x.mean(-1, keepdim=True); "
              "(x - m) / torch.sqrt(((x - m) * (x - m)).mean(-1, keepdim=True) + 1e-5)",
         ref=lambda x: [((_d(x) - _d(x).mean(-1, keepdims=True))
                         / np.sqrt(_d(x).var(-1, keepdims=True) + 1e-5)).astype(f32)],
         rtol=1e-3, atol=1e-4),
    dict(id="fx02_softmax_1000", code="torch.softmax(torch.randn(32, 1000), dim=-1)",
         ref=lambda x: [_softmax(_d(x)).astype(f32)], rtol=1e-3, atol=1e-5),
    dict(id="fx03_rmsnorm_2048",
         code="x = torch.randn(16, 2048); w = torch.randn(2048); "
              "x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-6) * w",
         ref=lambda x, w: [(_d(x) / np.sqrt((_d(x) ** 2).mean(-1, keepdims=True) + 1e-6) * w).astype(f32)],
         rtol=1e-3, atol=1e-4),
    dict(id="fx04_log_softmax", code="torch.log_softmax(torch.randn(64, 256), dim=-1)",
         ref=lambda x: [np.log(_softmax(_d(x))).astype(f32)], rtol=1e-3, atol=1e-4),
    dict(id="fx05_center", code="x = torch.randn(64, 256); x - x.mean(-1, keepdim=True)",
         ref=lambda x: [(_d(x) - _d(x).mean(-1, keepdims=True)).astype(f32)], rtol=1e-3, atol=1e-5),
    dict(id="fx06_softmax_narrow", code="torch.softmax(torch.randn(128, 64), dim=-1)",
         ref=lambda x: [_softmax(_d(x)).astype(f32)], rtol=1e-3, atol=1e-5),
    dict(id="fx07_l2norm_eps",
         code="x = torch.randn(32, 768); x / (torch.sqrt((x * x).sum(-1, keepdim=True)) + 1e-6)",
         ref=lambda x: [(_d(x) / (np.sqrt((_d(x) ** 2).sum(-1, keepdims=True)) + 1e-6)).astype(f32)],
         rtol=1e-3, atol=1e-4),
    dict(id="fx08_softmax_temp", code="x = torch.randn(64, 512); torch.softmax(x * 0.5, dim=-1)",
         ref=lambda x: [_softmax(_d(x) * 0.5).astype(f32)], rtol=1e-3, atol=1e-5),

    # ---- attention-like ---------------------------------------------------
    dict(id="ax01_small",
         code="q = torch.randn(32, 16); kt = torch.randn(16, 32); v = torch.randn(32, 16); "
              "torch.softmax(q @ kt / 4.0, dim=-1) @ v",
         ref=lambda q, kt, v: [(_softmax(_d(q) @ _d(kt) / 4.0) @ _d(v)).astype(f32)], rtol=1e-3, atol=1e-4),
    dict(id="ax02_256",
         code="q = torch.randn(256, 64); kt = torch.randn(64, 256); v = torch.randn(256, 64); "
              "torch.softmax(q @ kt / 8.0, dim=-1) @ v",
         ref=lambda q, kt, v: [(_softmax(_d(q) @ _d(kt) / 8.0) @ _d(v)).astype(f32)], rtol=1e-3, atol=1e-4),
    dict(id="ax03_rect",
         code="q = torch.randn(64, 32); kt = torch.randn(32, 128); v = torch.randn(128, 64); "
              "torch.softmax(q @ kt * 0.1767767, dim=-1) @ v",
         ref=lambda q, kt, v: [(_softmax(_d(q) @ _d(kt) * 0.1767767) @ _d(v)).astype(f32)], rtol=1e-3, atol=1e-4),
    dict(id="ax04_noscale",
         code="q = torch.randn(64, 64); kt = torch.randn(64, 64); v = torch.randn(64, 64); "
              "torch.softmax(q @ kt, dim=-1) @ v",
         ref=lambda q, kt, v: [(_softmax(_d(q) @ _d(kt)) @ _d(v)).astype(f32)], rtol=1e-3, atol=1e-4),
]
