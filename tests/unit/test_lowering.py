"""Whole-kernel lowering: the output contract, and the optimizations the README claims.

Needs emmy-ml==0.3.11 (to read the IR); no GPU. With ptxas available (PTXAS, the PATH,
or pip's nvidia-cuda-nvcc-cu12) every kernel is also assembled for sm_89.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import ROOT, kernels

from ptxc import compile_kernel, load_kernel
from ptxc.lower import Lowerer

KERNELS = kernels()
IDS = [f"{ir.parent.name}/{k['entry']}" for ir, k in KERNELS]


def compile_ir(ir: Path) -> str:
    return compile_kernel(load_kernel(ir))


def guess(name: str) -> Path:
    return next((ROOT / "tests" / "guess" / name).glob("*.ir.json"))


def extra(name: str) -> Path:
    return next((ROOT / "tests" / "extra" / name).glob("*.ir.json"))


# -- the contract, for every kernel ---------------------------------------------------
@pytest.mark.parametrize("ir,k", KERNELS, ids=IDS)
def test_contract(ir, k):
    ptx = compile_ir(ir)
    assert ptx.isascii()
    first = next(line for line in ptx.splitlines() if line.strip() and not line.startswith("//"))
    assert first == ".version 8.0"
    assert f".visible .entry {k['entry']}(" in ptx
    params = re.findall(r"\.param \.u64 (param_\w+)", ptx)
    assert params == [Lowerer.param_name(a) for a in k["arg_order"]]


@pytest.mark.parametrize("ir,k", KERNELS, ids=IDS)
def test_ptxas_accepts(ir, k, ptxas, tmp_path):
    if ptxas is None:
        pytest.skip("no ptxas (set PTXAS or pip install nvidia-cuda-nvcc-cu12)")
    src = tmp_path / "k.ptx"
    src.write_text(compile_ir(ir))
    r = subprocess.run(
        [ptxas, "-arch=sm_89", str(src), "-o", str(tmp_path / "k.cubin")], capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr


# -- the optimizations the README describes ---------------------------------------------
def test_tile_decode_has_no_division():
    ptx = compile_ir(extra("rx04_row_mean"))  # Tile[a0, a1_co] with 128 threads per row
    assert "div." not in ptx and "rem." not in ptx


def test_unrolled_body_uses_vector_stores_and_immediate_offsets():
    ptx = compile_ir(guess("gp01_swiglu"))
    assert "st.global.v4.f32" in ptx


def test_split_k_matmul_accumulates_atomically():
    ptx = compile_ir(guess("gm01_linear_b1"))
    assert "red.global.add.f32" in ptx


def test_dot_products_fuse_into_fma():
    ptx = compile_ir(extra("mx02_64sq"))
    assert "fma.rn.f32" in ptx


def test_long_strided_loop_is_partially_unrolled():
    ptx = compile_ir(guess("gr04_t512_mean"))  # 65536 elements over 512 threads: 128 trips each
    assert re.search(r"L_main_\d+:", ptx), "expected a main loop of unrolled copies"
    assert re.search(r"ld\.global\.nc\.f32\s+%f\d+, \[%rd\d+\+\d+\]", ptx), "expected immediate offsets"


def test_short_strided_loop_is_fully_unrolled():
    ptx = compile_ir(extra("rx04_row_mean"))  # 1024 / 128 = 8 trips for every thread
    loop = ptx.split("for a")[1].split("WarpShuffle")[0]
    assert "bra" not in loop


def test_pointwise_arithmetic_is_not_contracted():
    ptx = compile_ir(guess("gp06_scalar_chain"))  # zero-ulp problems need separate roundings
    assert "fma" not in ptx and "mul.rn.f32" in ptx and "add.rn.f32" in ptx


# -- errors --------------------------------------------------------------------------
def test_unknown_statement_is_a_clear_error():
    class Bogus:
        def pretty(self, indent=""):
            return ["Bogus"]

    lowerer = Lowerer(load_kernel(guess("gp10_tiny")))
    with pytest.raises(NotImplementedError, match="Bogus"):
        lowerer.stmt(Bogus())


def test_cli_prints_ptx_or_fails_cleanly(tmp_path):
    ok = subprocess.run(
        [sys.executable, str(ROOT / "compile.py"), str(guess("gp10_tiny"))], capture_output=True, text=True
    )
    assert ok.returncode == 0 and ok.stdout.startswith("//")
    bad = subprocess.run([sys.executable, str(ROOT / "compile.py")], capture_output=True, text=True)
    assert bad.returncode == 2 and bad.stdout == "" and "usage" in bad.stderr
