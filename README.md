# ptxc: Emmy Kernel IR → PTX

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

## How it works

```
compile.py
  └─ ptxc/kernel.py   load IR + meta.json → KernelSpec (entry, params, buffer shapes, launch)
  └─ ptxc/lower.py    walk the statement tree, one method per IR statement
       ├─ ptxc/index.py   integer indices as affine forms + value ranges
       ├─ ptxc/ops.py     elementwise f32 ops → PTX instruction sequences
       └─ ptxc/ptx.py     virtual registers, labels, final PTX text
```

### Values and scopes (`lower.py`)

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
| `Let`, `Init`, `TernaryExpr`, `CastExpr`, `FlatIndex` | constants, `selp`, `cvt` |

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

```bash
# every open problem, through the task repository's submit.py (the hidden-set pipeline)
python tools/check_all.py --task ../emmy-hackathon-2026

# the extra generated set
python tools/check_all.py --task ../emmy-hackathon-2026 --problems tests/extra
```

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
  a clear error: the task states that no problem uses them.
