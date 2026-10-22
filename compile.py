#!/usr/bin/env python3
"""Emmy Kernel IR -> PTX.

    python compile.py path/to/kernel.ir.json > out.ptx

Reads the Kernel IR (and the meta.json beside it for the parameter order and
launch geometry) and prints PTX on stdout. Diagnostics go to stderr; the exit
code is non-zero when the IR uses something the compiler cannot lower.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ptxc import compile_kernel, load_kernel


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python compile.py path/to/kernel.ir.json > out.ptx", file=sys.stderr)
        return 2
    try:
        ptx = compile_kernel(load_kernel(argv[1]))
    except (NotImplementedError, ValueError, KeyError, TypeError) as exc:
        print(f"ptxc: {argv[1]}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    sys.stdout.write(ptx)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
