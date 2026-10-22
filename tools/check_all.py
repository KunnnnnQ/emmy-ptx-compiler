#!/usr/bin/env python3
"""Run compile.py against every problem through the task repository's submit.py.

    python tools/check_all.py --task ../task            # all open problems
    python tools/check_all.py --task ../task p00 r0     # problems whose id starts with p00 or r0
    python tools/check_all.py --task ../task --problems tests/extra   # the extra generated set

This is exactly the hidden-set pipeline: compile, assemble on the runner, launch
under meta.json, compare with expected.npz.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_one(task: Path, problem: Path, attempts: int = 4) -> tuple[str, bool, str, float]:
    t0 = time.time()
    for attempt in range(attempts):
        r = subprocess.run(
            [sys.executable, str(task / "submit.py"), str(problem), "--compiler", str(ROOT / "compile.py")],
            capture_output=True,
            text=True,
        )
        out = (r.stdout + r.stderr).strip()
        # a busy runner answers 502 / drops the connection: that is not a verdict, so retry
        if r.returncode == 0 or not any(
            s in out for s in ("502 Server Error", "503 Server Error", "cannot reach the runner")
        ):
            break
        time.sleep(10 * (attempt + 1))
    return problem.name, r.returncode == 0, out, time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("prefixes", nargs="*", help="only problems whose id starts with one of these")
    ap.add_argument("--task", default=str(ROOT.parent / "task"), help="path to the emmy-hackathon-2026 checkout")
    ap.add_argument("--problems", default=None, help="directory of problem folders (default: <task>/problems)")
    ap.add_argument("-j", type=int, default=4, help="problems in flight at once")
    ap.add_argument("-v", action="store_true", help="print submit.py output for passing problems too")
    args = ap.parse_args()

    task = Path(args.task)
    pdir = Path(args.problems) if args.problems else task / "problems"
    problems = sorted((p for p in pdir.iterdir() if (p / "meta.json").exists()), key=lambda p: p.name)
    if args.prefixes:
        problems = [p for p in problems if any(p.name.startswith(x) for x in args.prefixes)]

    with ThreadPoolExecutor(max_workers=args.j) as pool:
        results = list(pool.map(lambda p: run_one(task, p), problems))

    passed = 0
    for problem, ok, out, secs in results:
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  {problem:<22} ({secs:.1f}s)")
        if not ok or args.v:
            print("      " + out.replace("\n", "\n      "))
    print(f"\n{passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
