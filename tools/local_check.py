#!/usr/bin/env python3
"""Check compile.py on problems using the LOCAL GPU instead of the hosted runner.

Same pipeline as the task's submit.py (compile.py -> PTX -> launch under meta.json
-> compare with expected.npz), with the runner's input conventions, but no network
and no queue, so hundreds of problems take minutes. The runner stays the final word.

    python tools/local_check.py tests/extra tests/guess ../task/problems   # problem dirs or parents
    python tools/local_check.py -j 4 --quiet out/fuzz

Each problem runs in its own process: a kernel that faults or hangs cannot take the
next one down with it.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DTYPES = {"f32": "float32", "f16": "float16", "f64": "float64", "i32": "int32", "i64": "int64"}


def run_problem(pdir: Path) -> dict:
    """Compile and run one problem in this process. Returns a result record."""
    import ctypes

    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from cudadrv import CudaError, Gpu

    meta = json.loads((pdir / "meta.json").read_text())
    expected = dict(np.load(pdir / "expected.npz")) if (pdir / "expected.npz").exists() else {}
    gpu = Gpu()
    produced: dict[str, np.ndarray] = {}
    result = {"problem": pdir.name, "ok": True, "detail": []}
    for k in _dependency_order(meta["kernels"]):
        ir = pdir / f"{k['entry']}.ir.json"
        t0 = time.time()
        r = subprocess.run([sys.executable, str(ROOT / "compile.py"), str(ir)], capture_output=True, text=True)
        compile_s = time.time() - t0
        if r.returncode != 0:
            return {**result, "ok": False, "stage": "compile", "detail": r.stderr.strip()[-800:]}
        try:
            fn = gpu.function(gpu.load(r.stdout), k["entry"])
        except CudaError as exc:
            return {**result, "ok": False, "stage": "ptxas", "detail": str(exc)[-800:]}
        args, outputs = [], {}
        for a in k["args"]:
            dt = np.dtype(DTYPES[a["dtype"]])
            shape = tuple(a["shape"])
            if a.get("scalar"):
                args.append(ctypes.c_float(a["value"]) if dt.kind == "f" else ctypes.c_int64(a["value"]))
            elif a.get("output"):
                ptr = gpu.zeros(math.prod(shape) * dt.itemsize)
                outputs[a["name"]] = (ptr, dt, shape)
                args.append(ptr)
            elif a.get("intermediate"):
                args.append(gpu.upload(produced[a["name"]].astype(dt)))
            elif "data" in a:
                args.append(gpu.upload(np.asarray(a["data"], dtype=dt).reshape(shape)))
            else:
                data = np.random.default_rng(a["seed"]).standard_normal(shape).astype(dt)
                args.append(gpu.upload(data))
        try:
            gpu.launch(fn, k["grid"], k["block"], k["smem_bytes"], args)
        except CudaError as exc:
            return {**result, "ok": False, "stage": "run", "detail": str(exc)}
        for name, (ptr, dt, shape) in outputs.items():
            got = gpu.download(ptr, dt, shape)
            produced[name] = got
            if name in expected:
                ref = expected[name]
                got32 = got.astype(np.float32).reshape(ref.shape)
                err = float(np.max(np.abs(got32 - ref))) if got32.size else 0.0
                ok = bool(np.allclose(got32, ref, rtol=meta["rtol"], atol=meta["atol"]))
                # 1.0 means some element sits exactly on the tolerance; the closer, the riskier
                allowed = meta["atol"] + meta["rtol"] * np.abs(ref.astype(np.float64))
                diff = np.abs(got32.astype(np.float64) - ref)
                margin = float(
                    np.max(np.where(allowed > 0, diff / np.maximum(allowed, 1e-300), np.where(diff > 0, np.inf, 0.0)))
                )
                result["margin"] = max(result.get("margin", 0.0), margin)
                result["detail"].append(
                    f"{k['entry']}:{name} err={err:.3e} margin={margin:.3f} compile={compile_s:.2f}s"
                )
                if not ok:
                    result.update(ok=False, stage="wrong", tol=(meta["rtol"], meta["atol"]))
        gpu.free_all()
    return result


def _dependency_order(kernels: list[dict]) -> list[dict]:
    """Kernels so that each intermediate is produced before it is read. The task's generator
    lists multi-kernel problems in graph-dict order, which can put the consumer first."""
    done, ordered, pending = set(), [], list(kernels)
    while pending:
        ready = [k for k in pending if all(a["name"] in done for a in k["args"] if a.get("intermediate"))]
        if not ready:
            return ordered + pending  # a cycle or a missing producer: keep meta order and fail loudly
        for k in ready:
            ordered.append(k)
            pending.remove(k)
            done.update(a["name"] for a in k["args"] if a.get("output"))
    return ordered


def find_problems(paths: list[str]) -> list[Path]:
    out = []
    for p in map(Path, paths):
        if (p / "meta.json").exists():
            out.append(p)
        elif p.is_dir():
            out.extend(sorted(q for q in p.iterdir() if (q / "meta.json").exists()))
    return out


def check_in_subprocess(pdir: Path, timeout: float) -> dict:
    try:
        r = subprocess.run(
            [sys.executable, __file__, "--one", str(pdir)], capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return {"problem": pdir.name, "ok": False, "stage": "timeout", "detail": f"over {timeout}s"}
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {"problem": pdir.name, "ok": False, "stage": "crash", "detail": (r.stderr or r.stdout)[-800:]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="problem folders, or folders of problem folders")
    ap.add_argument("--one", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("-j", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--quiet", action="store_true", help="only print failures")
    ap.add_argument("--json", help="write all results to this file")
    args = ap.parse_args()

    if args.one:
        print(json.dumps(run_problem(Path(args.paths[0]))))
        return 0

    problems = find_problems(args.paths)
    with ThreadPoolExecutor(max_workers=args.j) as pool:
        results = list(pool.map(lambda p: check_in_subprocess(p, args.timeout), problems))
    failed = [r for r in results if not r["ok"]]
    for r in results:
        if not r["ok"] or not args.quiet:
            status = "PASS" if r["ok"] else f"FAIL[{r.get('stage')}]"
            detail = r["detail"] if isinstance(r["detail"], str) else "; ".join(r["detail"])
            print(f"{status:<14}{r['problem']:<34}{detail if not r['ok'] or not args.quiet else ''}")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1))
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
