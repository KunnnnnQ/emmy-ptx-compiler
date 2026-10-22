"""Integer index arithmetic: affine forms, value ranges, and their PTX.

Every index the IR computes (axis decodes, loop counters, buffer offsets) is
kept as an affine form `sum(coeff * reg) + const` (`Lin`) for as long as
possible. Two things fall out of that:

* constants never reach a register: `x[a0 * 4 + 3]` becomes an address
  `[%rd + 12]`, the `+3` folded into the instruction's immediate offset;
* with value ranges per register (`tid.x` is in [0, 127] for a 128-thread
  block), `/` and `%` by a constant usually simplify away entirely:
  `(ctaid * 128 + tid) / 128` is `ctaid`, `% 128` is `tid`.

Emmy guarantees every index is non-negative, which is what makes `/` and `%`
by a power of two plain shifts and masks.
"""

from __future__ import annotations

from collections.abc import Callable

from ptxc.ptx import PtxBuilder

Range = tuple[int, int]

_CMP = {"<": "lt", "<=": "le", ">": "gt", ">=": "ge", "==": "eq", "!=": "ne"}


def _decide(op: str, lo: int, hi: int) -> bool | None:
    """The value of `d op 0` for every d in [lo, hi], or None if it depends on d."""
    true_if = {"<": hi < 0, "<=": hi <= 0, ">": lo > 0, ">=": lo >= 0, "==": lo == hi == 0, "!=": lo > 0 or hi < 0}
    false_if = {"<": lo >= 0, "<=": lo > 0, ">": hi <= 0, ">=": hi < 0, "==": lo > 0 or hi < 0, "!=": lo == hi == 0}
    if true_if[op]:
        return True
    if false_if[op]:
        return False
    return None


def c_div(a: int, b: int) -> int:
    """C integer division (truncates toward zero)."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


def c_mod(a: int, b: int) -> int:
    return a - b * c_div(a, b)


def is_pow2(n: int) -> bool:
    return n > 0 and n & (n - 1) == 0


class Lin:
    """An affine integer form: sum(coeff * register) + const."""

    __slots__ = ("terms", "const")

    def __init__(self, terms: dict[str, int] | None = None, const: int = 0) -> None:
        self.terms = {r: c for r, c in (terms or {}).items() if c}
        self.const = int(const)

    @classmethod
    def of(cls, reg: str) -> Lin:
        return cls({reg: 1})

    @property
    def is_const(self) -> bool:
        return not self.terms

    def key(self) -> tuple:
        return tuple(sorted(self.terms.items()))

    def var_part(self) -> Lin:
        return Lin(self.terms)

    def __add__(self, other: Lin | int) -> Lin:
        other = other if isinstance(other, Lin) else Lin(const=other)
        terms = dict(self.terms)
        for r, c in other.terms.items():
            terms[r] = terms.get(r, 0) + c
        return Lin(terms, self.const + other.const)

    def __sub__(self, other: Lin | int) -> Lin:
        other = other if isinstance(other, Lin) else Lin(const=other)
        return self + other.scale(-1)

    def scale(self, k: int) -> Lin:
        return Lin({r: c * k for r, c in self.terms.items()}, self.const * k)

    def range(self, ranges: dict[str, Range]) -> Range | None:
        lo = hi = self.const
        for r, c in self.terms.items():
            if r not in ranges:
                return None
            a, b = ranges[r][0] * c, ranges[r][1] * c
            lo, hi = lo + min(a, b), hi + max(a, b)
        return lo, hi

    def __repr__(self) -> str:
        parts = [f"{c}*{r}" for r, c in sorted(self.terms.items())]
        return " + ".join(parts + [str(self.const)])


class IntGen:
    """Turns Lin arithmetic into PTX, with value ranges and common-subexpression reuse.

    `cache_get` / `cache_put` are supplied by the lowering: they look up and store
    already-materialized forms in the current lexical scope, so a value computed
    inside a branch is never reused after it.
    """

    def __init__(self, b: PtxBuilder, cache_get: Callable, cache_put: Callable) -> None:
        self.b = b
        self.ranges: dict[str, Range] = {}
        self._get = cache_get
        self._put = cache_put

    # -- registers -----------------------------------------------------------
    def fresh(self, rng: Range | None = None) -> str:
        r = self.b.reg("s32")
        if rng is not None:
            self.ranges[r] = rng
        return r

    def operand(self, x: Lin) -> str:
        """A register or an immediate holding x."""
        return str(x.const) if x.is_const else self.reg_of(x)

    def reg_of(self, x: Lin) -> str:
        """A register holding x (constants are moved into one)."""
        if x.is_const:
            key = ("const", x.const)
            hit = self._get(key)
            if hit is None:
                hit = self.fresh((x.const, x.const))
                self.b.ins("mov.u32", hit, str(x.const))
                self._put(key, hit)
            return hit
        if x.const == 0 and len(x.terms) == 1 and next(iter(x.terms.values())) == 1:
            return next(iter(x.terms))
        key = ("lin", x.key(), x.const)
        hit = self._get(key)
        if hit is not None:
            return hit
        if x.const:
            base = self.reg_of(x.var_part())
            r = self.fresh(x.range(self.ranges))
            self.b.ins("add.s32", r, base, str(x.const))
        else:
            r = self._sum_terms(x)
        self._put(key, r)
        return r

    def _sum_terms(self, x: Lin) -> str:
        acc: str | None = None
        for reg, c in sorted(x.terms.items()):
            if acc is None:
                if c == 1:
                    acc = reg
                    continue
                acc_new = self.fresh()
                if is_pow2(c):
                    self.b.ins("shl.b32", acc_new, reg, str(c.bit_length() - 1))
                else:
                    self.b.ins("mul.lo.s32", acc_new, reg, str(c))
            else:
                acc_new = self.fresh()
                if c == 1:
                    self.b.ins("add.s32", acc_new, acc, reg)
                elif c == -1:
                    self.b.ins("sub.s32", acc_new, acc, reg)
                else:
                    self.b.ins("mad.lo.s32", acc_new, reg, str(c), acc)
            acc = acc_new
        rng = x.range(self.ranges)
        if rng is not None:
            self.ranges[acc] = rng
        return acc

    # -- arithmetic ----------------------------------------------------------
    def mul(self, a: Lin, b: Lin) -> Lin:
        if a.is_const:
            return b.scale(a.const)
        if b.is_const:
            return a.scale(b.const)
        ra, rb = self.reg_of(a), self.reg_of(b)
        rng = None
        A, B = a.range(self.ranges), b.range(self.ranges)
        if A and B:
            prods = [A[0] * B[0], A[0] * B[1], A[1] * B[0], A[1] * B[1]]
            rng = (min(prods), max(prods))
        r = self.fresh(rng)
        self.b.ins("mul.lo.s32", r, ra, rb)
        return Lin.of(r)

    def _split(self, a: Lin, m: int) -> tuple[Lin, Lin]:
        """a = m * q + rest, where q collects every multiple of m."""
        q_terms = {r: c // m for r, c in a.terms.items() if c % m == 0}
        rest_terms = {r: c for r, c in a.terms.items() if c % m != 0}
        return Lin(q_terms, a.const // m), Lin(rest_terms, a.const % m)

    def div(self, a: Lin, m: int) -> Lin:
        """a / m for a constant m > 0 and a >= 0."""
        if m == 1:
            return a
        if a.is_const:
            return Lin(const=c_div(a.const, m))
        q, rest = self._split(a, m)
        rng = rest.range(self.ranges)
        if rng is not None and rng[0] >= 0:
            if rng[1] < m:
                return q  # the remainder can never carry into the quotient
            return q + self._divmod_reg(rest, m, "div")
        return self._divmod_reg(a, m, "div")

    def mod(self, a: Lin, m: int) -> Lin:
        """a % m for a constant m > 0 and a >= 0."""
        if m == 1:
            return Lin()
        if a.is_const:
            return Lin(const=c_mod(a.const, m))
        _, rest = self._split(a, m)
        rng = rest.range(self.ranges)
        if rng is not None and rng[0] >= 0:
            if rng[1] < m:
                return rest
            return self._divmod_reg(rest, m, "mod")
        return self._divmod_reg(a, m, "mod")

    def _divmod_reg(self, a: Lin, m: int, kind: str) -> Lin:
        key = (kind, a.key(), a.const, m)
        hit = self._get(key)
        if hit is not None:
            return Lin.of(hit)
        src = self.reg_of(a)
        rng = a.range(self.ranges)
        if kind == "div":
            out_rng = (rng[0] // m, rng[1] // m) if rng and rng[0] >= 0 else None
            r = self.fresh(out_rng)
            if is_pow2(m):
                self.b.ins("shr.u32", r, src, str(m.bit_length() - 1))
            else:
                self.b.ins("div.u32", r, src, str(m))
        else:
            out_rng = (0, min(m - 1, rng[1])) if rng and rng[0] >= 0 else (0, m - 1)
            r = self.fresh(out_rng)
            if is_pow2(m):
                self.b.ins("and.b32", r, src, str(m - 1))
            else:
                self.b.ins("rem.u32", r, src, str(m))
        self._put(key, r)
        return Lin.of(r)

    def divmod_dyn(self, a: Lin, b: Lin, kind: str) -> Lin:
        """a / b or a % b where b is only known at run time."""
        if b.is_const:
            return self.div(a, b.const) if kind == "div" else self.mod(a, b.const)
        r = self.fresh()
        self.b.ins("div.s32" if kind == "div" else "rem.s32", r, self.operand(a), self.reg_of(b))
        return Lin.of(r)

    def minimum(self, a: Lin, b: Lin) -> Lin:
        if a.is_const and b.is_const:
            return Lin(const=min(a.const, b.const))
        r = self.fresh()
        self.b.ins("min.s32", r, self.reg_of(a), self.operand(b))
        return Lin.of(r)

    def bitop(self, op: str, a: Lin, b: Lin) -> Lin:
        if a.is_const and b.is_const:
            py = {"^": lambda x, y: x ^ y, "&": lambda x, y: x & y, "|": lambda x, y: x | y}[op]
            return Lin(const=py(a.const, b.const))
        r = self.fresh()
        opcode = {"^": "xor.b32", "&": "and.b32", "|": "or.b32"}[op]
        self.b.ins(opcode, r, self.reg_of(a), self.operand(b))
        return Lin.of(r)

    # -- predicates ----------------------------------------------------------
    def compare(self, op: str, a: Lin, b: Lin) -> str | bool:
        """A predicate register for `a op b`, or a Python bool when ranges decide it."""
        d = a - b
        rng = d.range(self.ranges)
        if rng is not None:
            decided = _decide(op, *rng)
            if decided is not None:
                return decided
        key = ("cmp", op, d.key(), d.const)
        hit = self._get(key)
        if hit is not None:
            return hit
        p = self.b.reg("pred")
        self.b.ins(f"setp.{_CMP[op]}.s32", p, self.reg_of(d.var_part()), str(-d.const))
        self._put(key, p)
        return p

    def select(self, pred: str, a: Lin, b: Lin) -> Lin:
        A, B = a.range(self.ranges), b.range(self.ranges)
        rng = (min(A[0], B[0]), max(A[1], B[1])) if A and B else None
        r = self.fresh(rng)
        self.b.ins("selp.b32", r, self.operand(a), self.operand(b), pred)
        return Lin.of(r)
