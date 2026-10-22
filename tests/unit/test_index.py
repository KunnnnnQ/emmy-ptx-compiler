"""Affine index forms, value ranges and their simplifications (no GPU, no emmy)."""

import itertools
import random

import pytest

from ptxc.index import IntGen, Lin, _decide, c_div, c_mod
from ptxc.ptx import PtxBuilder


def make_gen():
    cache: dict = {}
    b = PtxBuilder()
    return b, IntGen(b, cache.get, cache.__setitem__)


def reg_with_range(ints: IntGen, lo: int, hi: int) -> Lin:
    return Lin.of(ints.fresh((lo, hi)))


# -- C integer semantics ------------------------------------------------------
@pytest.mark.parametrize("a,b", [(7, 2), (-7, 2), (7, -2), (-7, -2), (0, 5), (6, 3)])
def test_c_div_mod_truncate_toward_zero(a, b):
    q = int(a / b)
    assert c_div(a, b) == q
    assert c_mod(a, b) == a - b * q


# -- Lin arithmetic -------------------------------------------------------------
def test_lin_arithmetic_and_cancellation():
    x, y = Lin.of("%r1"), Lin.of("%r2")
    e = x.scale(64) + y.scale(2) + 1 - x.scale(64)
    assert e.terms == {"%r2": 2} and e.const == 1


def test_lin_range():
    e = Lin({"%r1": 4, "%r2": -1}, 3)
    assert e.range({"%r1": (0, 10), "%r2": (0, 5)}) == (-2, 43)
    assert e.range({"%r1": (0, 10)}) is None  # unknown register


# -- div / mod simplification -----------------------------------------------------
def test_tile_decode_folds_to_block_and_thread_ids():
    # gid = ctaid * 128 + tid with tid < 128: gid / 128 == ctaid and gid % 128 == tid
    b, ints = make_gen()
    ctaid, tid = reg_with_range(ints, 0, 63), reg_with_range(ints, 0, 127)
    gid = ctaid.scale(128) + tid
    assert ints.div(gid, 128).terms == ctaid.terms
    assert ints.mod(gid, 128).terms == tid.terms
    assert b.body == []  # no instruction was needed


def test_div_by_power_of_two_is_a_shift_and_mod_a_mask():
    b, ints = make_gen()
    x = reg_with_range(ints, 0, 4095)
    ints.reg_of(ints.div(x, 64))
    ints.reg_of(ints.mod(x, 64))
    text = "\n".join(b.body)
    assert "shr.u32" in text and "and.b32" in text and "div." not in text


def test_div_by_other_constants_uses_unsigned_division():
    b, ints = make_gen()
    x = reg_with_range(ints, 0, 4095)
    ints.reg_of(ints.div(x, 3))
    ints.reg_of(ints.mod(x, 3))
    text = "\n".join(b.body)
    assert "div.u32" in text and "rem.u32" in text


def test_exact_division_of_multiples_needs_no_instruction():
    b, ints = make_gen()
    x = reg_with_range(ints, 0, 99)
    assert ints.div(x.scale(128) + 64, 64).terms == x.scale(2).terms
    assert ints.mod(x.scale(128) + 64, 64).const == 0
    assert b.body == []


def test_materialized_forms_are_reused():
    b, ints = make_gen()
    x, y = reg_with_range(ints, 0, 9), reg_with_range(ints, 0, 9)
    first = ints.reg_of(x.scale(4) + y + 3)
    n = len(b.body)
    assert ints.reg_of(x.scale(4) + y + 3) == first
    assert len(b.body) == n


# -- comparisons decided by ranges ---------------------------------------------------
@pytest.mark.parametrize("op", ["<", "<=", ">", ">=", "==", "!="])
def test_decide_agrees_with_brute_force(op):
    py = {
        "<": lambda d: d < 0,
        "<=": lambda d: d <= 0,
        ">": lambda d: d > 0,
        ">=": lambda d: d >= 0,
        "==": lambda d: d == 0,
        "!=": lambda d: d != 0,
    }[op]
    for lo, hi in itertools.combinations_with_replacement(range(-3, 4), 2):
        values = {py(d) for d in range(lo, hi + 1)}
        decided = _decide(op, lo, hi)
        if decided is None:
            assert values == {True, False}, (op, lo, hi)  # never undecided when it is decidable
        else:
            assert values == {decided}, (op, lo, hi)


def test_compare_emits_setp_only_when_ranges_do_not_decide():
    b, ints = make_gen()
    lane = reg_with_range(ints, 0, 31)
    assert ints.compare("<", lane, Lin(const=32)) is True
    assert ints.compare(">=", lane, Lin(const=32)) is False
    assert b.body == []
    p = ints.compare("==", lane, Lin())
    assert isinstance(p, str) and p.startswith("%p")
    assert "setp.eq.s32" in b.body[-1]


def test_random_affine_forms_evaluate_like_python():
    """Fold random forms with random register values; div/mod must match Python on non-negatives."""
    rng = random.Random(0)
    for _ in range(300):
        _, ints = make_gen()
        regs = {ints.fresh((0, hi)): rng.randrange(hi + 1) for hi in (7, 31, 127)}
        coeffs = {r: rng.choice([0, 1, 2, 4, 32, 96]) for r in regs}
        form = Lin(coeffs, rng.randrange(0, 50))
        m = rng.choice([1, 2, 3, 8, 32, 64, 100])
        value = sum(c * regs[r] for r, c in coeffs.items()) + form.const
        for kind, expect in (("div", value // m), ("mod", value % m)):
            out = ints.div(form, m) if kind == "div" else ints.mod(form, m)
            if all(r in regs for r in out.terms):  # folded symbolically: evaluate the form
                got = sum(c * regs[r] for r, c in out.terms.items()) + out.const
                assert got == expect, (form, kind, m)
