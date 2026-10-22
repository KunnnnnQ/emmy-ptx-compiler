#!/usr/bin/env python3
"""Time our PTX against Emmy's own CUDA path on the runner. MEASUREMENT ONLY.

The baseline is the problem's reference CUDA (`*.ref.cu`, Emmy's renderer output)
compiled to PTX with NVRTC and Emmy's flags (fast math, sm_89). It is never part of
compile.py; it only answers "how fast is our PTX next to what nvcc-class tooling makes".

    pip install nvidia-cuda-nvrtc-cu12 requests
    python tools/bench.py --task ../task                    # open set
    python tools/bench.py --task ../task --problems tests/extra m a   # extra set, ids m* and a*
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
RUNNER = "http://runner.riftstack.ai:48701"


def _load_nvrtc():
    import nvidia.cuda_nvrtc as pkg

    base = Path(next(iter(pkg.__path__)))
    names = ["nvrtc64_120_0.dll"] if sys.platform == "win32" else ["libnvrtc.so.12", "libnvrtc.so"]
    for d in (base / "bin", base / "lib"):
        for n in names:
            if (d / n).exists():
                if sys.platform == "win32":  # nvrtc loads nvrtc-builtins from the search path
                    os.add_dll_directory(str(d))
                    os.environ["PATH"] = str(d) + os.pathsep + os.environ["PATH"]
                return ctypes.CDLL(str(d / n))
    raise FileNotFoundError("NVRTC library not found")


def nvrtc_ptx(source: str, name: str) -> str:
    lib = _load_nvrtc()
    prog = ctypes.c_void_p()
    src = source.encode()
    if lib.nvrtcCreateProgram(ctypes.byref(prog), src, name.encode(), 0, None, None):
        raise RuntimeError("nvrtcCreateProgram failed")
    opts = [b"--gpu-architecture=compute_89", b"--use_fast_math", b"-std=c++17"]
    arr = (ctypes.c_char_p * len(opts))(*opts)
    rc = lib.nvrtcCompileProgram(prog, len(opts), arr)
    if rc:
        size = ctypes.c_size_t()
        lib.nvrtcGetProgramLogSize(prog, ctypes.byref(size))
        log = ctypes.create_string_buffer(size.value)
        lib.nvrtcGetProgramLog(prog, log)
        raise RuntimeError(log.value.decode(errors="replace")[:500])
    size = ctypes.c_size_t()
    lib.nvrtcGetPTXSize(prog, ctypes.byref(size))
    buf = ctypes.create_string_buffer(size.value)
    lib.nvrtcGetPTX(prog, buf)
    lib.nvrtcDestroyProgram(ctypes.byref(prog))
    return buf.value.decode()


def post(path: str, body: dict) -> dict:
    """POST to the runner, retrying while it is busy or briefly unavailable."""
    for attempt in range(30):
        resp = requests.post(f"{RUNNER}{path}", json=body, timeout=(10, 300))
        if resp.status_code == 422:
            return {"ok": False, "stderr": f"HTTP 422: {resp.text[:200]}"}
        if resp.status_code == 200:
            r = resp.json()
            if r.get("stage") != "busy":
                return r
        time.sleep(2 + attempt)
    return {"ok": False, "stderr": f"runner unavailable (last HTTP {resp.status_code})"}


def run(ptx: str, k: dict, iters: int) -> tuple[float | None, int | None, str]:
    c = post("/compile", {"ptx": ptx})
    if not c.get("ok"):
        return None, None, c.get("stderr", "")[:300]
    regs = c.get("registers", {}).get(k["entry"])
    body = {
        "cubin_id": c["cubin_id"],
        "entry": k["entry"],
        "grid": k["grid"],
        "block": k["block"],
        "smem_bytes": k["smem_bytes"],
        "zero_outputs": k["zero_outputs"],
        "args": k["args"],
        "warmup": 5,
        "iters": iters,
        "max_return": 1,
    }
    r = post("/run", body)
    if not r.get("ok"):
        return None, regs, r.get("stderr", "")[:300]
    return r["median_us"], regs, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("prefixes", nargs="*")
    ap.add_argument("--task", default=str(ROOT.parent / "task"))
    ap.add_argument("--problems", default=None)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()
    pdir = Path(args.problems) if args.problems else Path(args.task) / "problems"
    problems = sorted(p for p in pdir.iterdir() if (p / "meta.json").exists())
    if args.prefixes:
        problems = [p for p in problems if any(p.name.startswith(x) for x in args.prefixes)]

    print(f"{'problem':<24}{'ours us':>9}{'ref us':>9}{'ratio':>7}{'regs':>11}")
    for p in problems:
        meta = json.loads((p / "meta.json").read_text())
        if len(meta["kernels"]) != 1:
            print(f"{p.name:<24} (multi-kernel, skipped)")
            continue
        k = meta["kernels"][0]
        ir = p / f"{k['entry']}.ir.json"
        ours = subprocess.run([sys.executable, str(ROOT / "compile.py"), str(ir)], capture_output=True, text=True)
        if ours.returncode:
            print(f"{p.name:<24} compile failed: {ours.stderr.strip()[:100]}")
            continue
        t_ours, r_ours, err = run(ours.stdout, k, args.iters)
        try:
            ref = nvrtc_ptx((p / f"{k['entry']}.ref.cu").read_text(), k["entry"] + ".cu")
            t_ref, r_ref, err2 = run(ref, k, args.iters)
        except (RuntimeError, FileNotFoundError) as exc:
            t_ref, r_ref, err2 = None, None, str(exc)[:100]
        ratio = f"{t_ours / t_ref:6.2f}" if t_ours and t_ref else "     -"
        fmt = lambda t: f"{t:9.1f}" if t else "        -"
        print(f"{p.name:<24}{fmt(t_ours)}{fmt(t_ref)} {ratio}{r_ours!s:>5}/{r_ref!s:<5}  {err or err2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
