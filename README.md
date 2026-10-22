# ptxc: Emmy Kernel IR → PTX

[![ci](https://github.com/KunnnnnQ/emmy-ptx-compiler/actions/workflows/ci.yml/badge.svg)](https://github.com/KunnnnnQ/emmy-ptx-compiler/actions/workflows/ci.yml)

A compiler from [Emmy](https://github.com/cloudrift-ai/emmy)'s Kernel IR straight to PTX, written for the
[Emmy Hackathon 2026](https://riftstack.ai/hackathon). It replaces Emmy's `Kernel IR → CUDA C++ → nvcc`
path with a direct `Kernel IR → PTX` translation that `ptxas` can assemble.

```bash
pip install emmy-ml==0.3.11
python compile.py path/to/kernel.ir.json > out.ptx
```

Emmy is used only to **load** the IR (`Graph.from_dict`). Every statement and expression is lowered to
PTX by the code in `ptxc/`; nothing goes through Emmy's renderer, `nvcc` or any other compiler.

## Results

| set | problems | passing |
|---|---|---|
| open set (task repository) | 15: pointwise, reductions, matmul, fused, attention | **15 / 15** |
| extra set (`tests/extra`, generated with the task's own generator) | 47 across the same five categories | **47 / 47** |
| hidden-set guess (`tests/guess`, Emmy's own test workloads + schedule variants) | 40 | **40 / 40** |
| fuzzing (`tools/fuzz.py`, random specs over all five categories, 4 seeds) | 600 | **600 / 600** |

Across all of these the largest error uses 13% of a problem's tolerance (an fp16 row sum, bounded by fp16
rounding of the output); every other problem stays under 10%.

**Speed.** `tools/bench.py` times each kernel on the runner next to a baseline: the problem's reference CUDA
(Emmy's own renderer output) compiled by NVRTC with Emmy's flags. Every one of the 61 kernels with a
baseline (NVRTC rejects one reference) runs at parity or faster, within the runner's timing noise; the
largest gains come from loop unrolling (below):

| kernel | ptxc | Emmy CUDA + NVRTC |
|---|---|---|
| a01 attention 64×64 | 47.7 µs | 72.3 µs |
| a02 attention 128×32 | 27.3 µs | 31.4 µs |
| attention 64×64, no scale (extra) | 281 µs | 377 µs |
| softmax 32×1000 (extra) | 2.9 µs | 4.7 µs |
| matmul 96×80×48 (extra) | 3.1 µs | 5.0 µs |
| r02 sum of 65536 | 8.8 µs | 8.9 µs |
| m01 matmul 512³ | 15.0 µs | 15.0 µs |

The extra set varies what the hidden set varies: shapes (odd lengths such as 999 and 1000, 2-D and 3-D,
rectangular and 1024² matmuls, 1000-wide softmax), and ops (exp, log, sigmoid, tanh, gelu/erf, silu,
rsqrt, sqrt, where, clamp, maximum/minimum, column sums, max-reductions, split-K atomics, layer-norm-style
centering, temperature softmax, attention of several sizes).

The open set pins every reduction to one schedule (`WORK=t128,REDUCE=coop`), so the guess set varies what
it holds fixed, to catch a compiler that only fits the open problems: block widths 32–512 (tree lengths
2–16), the transposed cooperative reduce (segmented `TreeHalve`), register-split and grid-split reduces
(atomic, and a two-kernel chain), fp16 buffers, rasterized grids, masked attention tails (`Select`), and
the workloads in Emmy's own tests and recipes: batch-1 `F.linear`, a 1×1024·1024×3072 matvec, `bmm`,
SwiGLU, the `pow(mean(pow(x, 2)) + eps, -0.5)` RMSNorm, affine layer norm, 4-D and fp16 softmax,
batched and multi-head attention.

## How it works

```
compile.py
  └─ ptxc/kernel.py        load IR + meta.json → KernelSpec (entry, params, buffer shapes, launch)
  └─ ptxc/lower/           walk the statement tree, one method per IR statement, by family:
       base.py               values, lexical scopes, dispatch, special registers
       exprs.py              indices, predicates, loop bounds
       memory.py             buffers, addresses, shared memory, Load / Write / cp.async
       compute.py            Assign, Accum (+ fma fusion), Let, Init, Select
       control.py            Tile, Sync, loops (unrolling), Cond
       combine.py            WarpShuffle, TreeHalve
  └─ ptxc/index.py         integer indices as affine forms + value ranges
  └─ ptxc/ops.py           elementwise f32 ops → PTX instruction sequences
  └─ ptxc/ptx.py           virtual registers, labels, final PTX text
```

### Values and scopes (`lower/base.py`)

Each IR name maps to a value in a chain of lexical scopes:

* `IntV`: an integer kept symbolic as an affine form (axis indices, loop counters, integer `Let`s);
* `FloatV`: an f32 virtual register (loads, computed values, accumulators).

Registers are fresh and never reused; `ptxas` does the real register allocation. The scopes encode the
IR's semantics:

* a `Loop`/`StridedLoop` seeds each `Accum` in its immediate body to the op's identity, just before the
  loop and in the enclosing scope, so the reduced value is visible after it; `Accum` updates that register
  in place (`acc = op(base or acc, value)`);
* names defined inside a loop or branch are local to it;
* a combine program (`WarpShuffle`, `TreeHalve`) runs with the carried state names marked in-place, so its
  final `Assign`s overwrite the state registers, exactly like Emmy's sequential merge semantics.

### Integer indices (`index.py`)

Every index is an affine form `Σ coeff·reg + const` for as long as possible, and every register carries a
value range (`%tid.x ∈ [0, 127]`, a decoded axis `∈ [0, extent−1]`, ...). Two things follow:

* **constants become immediate offsets**: `x[a0*4 + 3]` is `ld.global.f32 %f, [%rd+12]`, and an
  unrolled body of four loads shares one address register;
* **`/` and `%` mostly disappear**: the tile decode `a0 = gid / 128, a1 = gid % 128` with
  `gid = ctaid*128 + tid` and `tid < 128` simplifies to `a0 = ctaid, a1 = tid` at compile time. What
  remains is a shift/mask for powers of two, `div/rem.u32` otherwise (Emmy guarantees non-negative
  indices). Comparisons the ranges already decide (a bounds guard that can never fire, a loop that every
  thread is known to enter, a tree level every thread takes part in) are folded away too.

Materialized forms are cached per scope (common-subexpression reuse that never leaks out of a branch).

### Statements → PTX

| IR | PTX |
|---|---|
| `Tile` | `gid = ctaid.x·blockDim + tid.x` (blockDim from `meta.json`), per-axis decode, bounds guard when the launch has more threads than cells |
| `Load` / `Write` | `ld/st.global[.nc][.v2/.v4].f32` with `[reg+imm]` addressing; read-only inputs use the non-coherent path; `atomic` → `red.global.add.f32` |
| `Assign` / `Accum` | `ops.py`; `acc += a*b` becomes one `fma.rn.f32` when it is safe |
| `Loop` / `StridedLoop` | see *Loops* below |
| `Cond` | `setp` + predicated branch, or nothing when ranges decide it |
| `Smem` | module-level `.shared` arrays, or one `.extern .shared` pool above 48 KiB (matches `meta.json`'s dynamic size) |
| `Sync` | `bar.sync 0` / named `bar.sync id, count` / `bar.warp.sync` |
| `WarpShuffle` | `shfl.sync.bfly.b32` ladder under `activemask`, then the combine program |
| `TreeHalve` | unrolled halving over shared memory with `bar.sync` between levels, then a broadcast load |
| `CpAsyncCopy/Commit/Wait` | `cp.async.cg/ca.shared.global`, `cp.async.commit_group`, `cp.async.wait_group` |
| `Let`, `Init`, `Select`, `TernaryExpr`, `CastExpr`, `FlatIndex` | constants, `selp` chains, `cvt` |

### Loops

The trip count of a strided loop usually follows from the value ranges: `for a1 in tid..1024:128` runs
exactly 8 times for every `tid ∈ [0, 127]`. `_counted_loop` picks the cheapest shape:

* **fully unrolled** when the count is small: the loop variable becomes the affine form
  `start + k·step`, so `x[a0, a1]` in copy `k` is `[%rd + 512k]`, one base register for all copies.
  Copies only some threads reach (an odd length) get a predicate;
* **partially unrolled** (2–8 copies per trip by body size, then a remainder loop) when it is long:
  the copies' loads are independent, so many are in flight at once. This took the 65536-element sum from
  21 µs to 9 µs and the attention kernels below the CUDA baseline;
* a plain bottom-tested loop otherwise. Bodies containing a barrier only get shapes in which every
  thread does the same thing.

Unrolled copies are straight-line code, so a copy's scope stores the values it materializes (address
registers, shared index terms) in the enclosing trip scope, where the next copy reuses them; names stay
private to each copy.

### Numerics (`ops.py`)

Arithmetic is IEEE round-to-nearest (`add.rn`, `mul.rn`, `div.rn`, `sqrt.rn`, `rcp.rn`). The explicit
rounding mode also stops `ptxas` from contracting a multiply and an add into an fma, so pointwise kernels
match numpy bit for bit (several problems have zero tolerance). Transcendentals use the hardware
approximations Emmy's fast-math build uses (`ex2.approx`, `lg2.approx`, `rsqrt.approx`); `tanh` is built
from `ex2` because `tanh.approx` is only good to ~2⁻¹¹, and `erf` uses Abramowitz–Stegun 7.1.26 (error
below 1.5·10⁻⁷).

## Testing

Unit tests run in CI on every push (`.github/workflows/ci.yml`), together with `ruff`:

```bash
pip install -r requirements.txt pytest ruff numpy "nvidia-cuda-nvcc-cu12==12.9.*"
TASK_DIR=../emmy-hackathon-2026 pytest
```

* `tests/unit/test_index.py`: affine forms, C division semantics, the div/mod folds; the range-decided
  comparisons are checked against brute force over every small interval, and random forms against Python.
* `tests/unit/test_lowering.py`: every kernel of the open, extra and guess sets keeps the contract (entry
  name, `arg_order` parameters, ASCII) and assembles with `ptxas` for sm_89; the optimizations above are
  asserted on real kernels (no division in tile decodes, vector stores, fma, full and partial unrolling,
  no contraction in pointwise code).
* `tests/unit/test_gpu_ops.py`: every elementwise op as its own kernel on a local GPU against float64
  numpy; the exact ones must match bit for bit (skipped without a GPU, as in CI).

The end-to-end checks below need the runner or a GPU:

```bash
# every open problem, through the task repository's submit.py (the hidden-set pipeline)
python tools/check_all.py --task ../emmy-hackathon-2026

# the extra generated set, and the hidden-set guess
python tools/check_all.py --task ../emmy-hackathon-2026 --problems tests/extra
python tools/check_all.py --task ../emmy-hackathon-2026 --problems tests/guess
```

```bash
# fuzzing: random problems through the task's own generator (Linux/WSL, emmy-ml 0.3.11 + torch) ...
python tools/fuzz.py --task ../emmy-hackathon-2026 --emmy emmy --seed 5 --count 200 --out out/fuzz/s5 -j 12
# ... checked on a LOCAL GPU (any NVIDIA driver, no toolkit): same inputs, launch and tolerances as the runner
python tools/local_check.py out/fuzz/s5 --quiet
```

`tools/fuzz_specs.py` builds the random specs: pointwise expression trees (1–3 inputs, broadcasting,
19 unary and 6 binary ops including sin/cos/erf/gelu), reductions over a random axis with random schedule
knobs, matmul/`bmm`/`F.linear` with epilogues and odd sizes, row-wise fused kernels, and attention with
random lengths, head sizes and batch. `tools/local_check.py` runs each problem in its own process with a
ctypes binding of the CUDA driver API (`tools/cudadrv.py`) and reports, besides pass/fail, how much of the
tolerance the worst element uses. A fuzz seed cross-checked on the runner (RTX 5090) agreed with the local
GPU (RTX 4060) on all 191 problems.

```bash
# timing against the Emmy CUDA + NVRTC baseline (measurement only: pip install nvidia-cuda-nvrtc-cu12)
python tools/bench.py --task ../emmy-hackathon-2026 [--problems tests/extra]
```

`tests/expressions_extra.py` lists the extra problems; regenerate them with the task's generator and
emmy-ml 0.3.11 + CPU torch (`tools/setup_gen_env.sh` sets that up on Linux/WSL):

```bash
python ../emmy-hackathon-2026/tools/generate.py tests/expressions_extra.py --emmy emmy --out tests/extra
```

## Limits

* f32 compute throughout; f16 buffers are converted on load and store.
* `cp.async` with a swizzle, TMA, mbarriers and tensor-core (`mma`/`wgmma`) statements are rejected with
  a clear error: the task states that no problem uses them. (Emmy does pick tensor cores for an fp16
  matmul, so that one workload is out of scope.)
