#!/usr/bin/env python3
"""Generate random problems with the task's own generator, in parallel.

Needs emmy-ml==0.3.11 + torch (tools/setup_gen_env.sh) and the task repository:

    python tools/fuzz.py --task ../task --seed 1 --count 200 --out out/fuzz/s1 -j 12
    python tools/local_check.py out/fuzz/s1 --quiet          # then check them on the local GPU

Specs come from tools/fuzz_specs.py. A spec emmy itself cannot trace is reported and
skipped; every packaged problem is a valid test, expected outputs from float64 numpy.
"""

from __future__ import annotations

import argparse
import importlib.util
import multiprocessing as mp
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fuzz_specs import build

SPECS: list[dict] = []
GEN = None
ARGS = None


def _load_generator(task: Path):
    spec = importlib.util.spec_from_file_location("task_generate", task / "tools" / "generate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _one(i: int) -> tuple[str, str, str]:
    spec = SPECS[i]
    out = Path(ARGS.out) / spec["id"]
    with tempfile.TemporaryDirectory() as td:
        dump = Path(td) / "dump"
        try:
            GEN.trace(ARGS.emmy, spec, dump, "sm_89")
            GEN.package(spec, dump, out)
            return spec["id"], "ok", ""
        except subprocess.CalledProcessError as e:
            tail = (e.stderr or "").strip().splitlines()[-1:] or ["?"]
            return spec["id"], "untraceable", tail[0][:120]
        except Exception as e:
            shutil.rmtree(out, ignore_errors=True)
            return spec["id"], "package-error", str(e)[:120]


def main() -> int:
    global SPECS, GEN, ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--emmy", default="emmy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("-j", type=int, default=8)
    ARGS = ap.parse_args()
    SPECS = build(ARGS.seed, ARGS.count)
    GEN = _load_generator(Path(ARGS.task))
    Path(ARGS.out).mkdir(parents=True, exist_ok=True)
    (Path(ARGS.out) / "specs.txt").write_text("\n".join(f"{s['id']}\t{s.get('knobs', '')}\t{s['code']}" for s in SPECS))

    t0 = time.time()
    with mp.get_context("fork").Pool(ARGS.j) as pool:
        rows = pool.map(_one, range(len(SPECS)), chunksize=1)
    counts = {}
    for pid, status, why in rows:
        counts[status] = counts.get(status, 0) + 1
        if status != "ok":
            print(f"  {status:<14}{pid:<16}{why}")
    print(f"\n{counts} in {time.time() - t0:.0f}s -> {ARGS.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
