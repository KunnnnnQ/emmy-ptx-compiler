import json
import os
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def problem_dirs() -> list[Path]:
    """Every problem folder the tests can see: the repo's generated sets, plus the open set
    when TASK_DIR points at a checkout of the task repository."""
    roots = [ROOT / "tests" / "extra", ROOT / "tests" / "guess"]
    if os.environ.get("TASK_DIR"):
        roots.append(Path(os.environ["TASK_DIR"]) / "problems")
    out = []
    for r in roots:
        if r.is_dir():
            out.extend(sorted(p for p in r.iterdir() if (p / "meta.json").exists()))
    return out


def kernels() -> list[tuple[Path, dict]]:
    out = []
    for p in problem_dirs():
        meta = json.loads((p / "meta.json").read_text())
        out.extend((p / f"{k['entry']}.ir.json", k) for k in meta["kernels"])
    return out


@pytest.fixture(scope="session")
def ptxas() -> str | None:
    """ptxas from PTXAS, the PATH, or the pip package nvidia-cuda-nvcc-cu12."""
    if os.environ.get("PTXAS"):
        return os.environ["PTXAS"]
    found = shutil.which("ptxas")
    if found:
        return found
    try:
        import nvidia.cuda_nvcc as pkg

        cand = Path(next(iter(pkg.__path__))) / "bin" / ("ptxas.exe" if os.name == "nt" else "ptxas")
        return str(cand) if cand.exists() else None
    except ImportError:
        return None
