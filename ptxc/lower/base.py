"""The Lowerer's core: values, lexical scopes, statement dispatch and kernel-wide registers.

The Lowerer walks the body once, in order, with one method per statement kind
(`s_Tile`, `s_Load`, ...), spread over the mixins in this package by statement
family. It keeps a chain of lexical scopes mapping IR names to values:

* IntV   - an integer as an affine form (axis indices, loop counters, Let ints)
* FloatV - an f32 virtual register (loaded and computed values, accumulators)

IR semantics the scopes implement:

* a `Loop` / `StridedLoop` seeds every `Accum` in its immediate body to the op's
  identity right before the loop, in the enclosing scope, so the reduced value
  is visible after the loop; the `Accum` then updates that register in place;
* names defined inside a loop or branch body are local to it;
* in a combine program (`WarpShuffle` / `TreeHalve`), an `Assign` to one of the
  carried state names updates that state's register in place.
"""

from __future__ import annotations

from dataclasses import dataclass

from ptxc.index import IntGen, Lin
from ptxc.kernel import Buffer, KernelSpec
from ptxc.lower.walk import pretty
from ptxc.ptx import PtxBuilder, f32_imm

_SPECIAL_REGS = {
    "threadIdx.x": "%tid.x",
    "threadIdx.y": "%tid.y",
    "threadIdx.z": "%tid.z",
    "blockIdx.x": "%ctaid.x",
    "blockIdx.y": "%ctaid.y",
    "blockIdx.z": "%ctaid.z",
    "blockDim.x": "%ntid.x",
    "gridDim.x": "%nctaid.x",
}

# Block statements print their own header comment (their full pretty form would repeat the body).
_BLOCKS = ("Tile", "Loop", "StridedLoop", "Cond")


@dataclass(frozen=True)
class IntV:
    lin: Lin


@dataclass(frozen=True)
class FloatV:
    reg: str


class Scope:
    """Names defined in a block, plus the values materialized there (for reuse).

    A `shares_values` scope is a straight-line block (one copy of an unrolled loop
    body): its code dominates everything after it in the parent, so the values it
    materializes are stored in the parent and the next copy reuses them, while its
    names stay private to it.
    """

    def __init__(self, parent: Scope | None, shares_values: bool = False) -> None:
        self.parent = parent
        self.shares_values = shares_values
        self.names: dict[str, object] = {}
        self.cse: dict[tuple, object] = {}

    def value_home(self) -> Scope:
        s = self
        while s.shares_values and s.parent is not None:
            s = s.parent
        return s

    def lookup(self, name: str):
        s = self
        while s is not None:
            if name in s.names:
                return s.names[name]
            s = s.parent
        return None

    def cse_get(self, key: tuple):
        s = self
        while s is not None:
            if key in s.cse:
                return s.cse[key]
            s = s.parent
        return None


class LowererBase:
    def __init__(self, spec: KernelSpec) -> None:
        self.spec = spec
        self.b = PtxBuilder()
        self.scope = Scope(None)
        self.ints = IntGen(self.b, lambda k: self.scope.cse_get(k), self._cse_put)
        self.inplace: frozenset[str] = frozenset()  # state names a combine program updates in place
        self.buffers: dict[str, Buffer] = dict(spec.buffers)
        self.smem_base: dict[str, str] = {}  # smem name -> u32 register with its address
        self.smem_offset: dict[str, int] = {}  # dynamic-pool offsets
        self.dynamic_smem = False
        self.global_base: dict[str, str] = {}
        self.special: dict[str, str] = {}
        self.writes = 0  # counter of in-place register writes, for the fma fusion check
        self.last_write: dict[str, int] = {}
        self.exit_label = self.b.label("exit")
        self.used_exit = False

    def _cse_put(self, key: tuple, value) -> None:
        self.scope.value_home().cse[key] = value

    # -- driver ------------------------------------------------------------
    def run(self) -> str:
        self._plan_shared_memory(self.spec.body)
        self.body(self.spec.body)
        if self.used_exit:
            self.b.place(self.exit_label)
        params = [f".param .u64 {self.param_name(p)}" for p in self.spec.params]
        maxntid = self.spec.block if self.spec.block != (1, 1, 1) else None
        return self.b.render(self.spec.name, params, target=self.spec.target, maxntid=maxntid)

    @staticmethod
    def param_name(buf: str) -> str:
        return "param_" + "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in buf)

    def body(self, stmts) -> None:
        for s in stmts:
            self.stmt(s)

    def stmt(self, s) -> None:
        kind = type(s).__name__
        handler = getattr(self, f"s_{kind}", None)
        if handler is None:
            raise NotImplementedError(f"Kernel IR statement {kind} is not supported")
        if kind not in _BLOCKS:
            self.b.comment(pretty(s))
        handler(s)

    # -- scopes ------------------------------------------------------------
    def push(self, shares_values: bool = False) -> Scope:
        self.scope = Scope(self.scope, shares_values)
        return self.scope

    def pop(self) -> None:
        self.scope = self.scope.parent

    def define(self, name: str, value) -> None:
        self.scope.names[name] = value

    def lookup(self, name: str):
        v = self.scope.lookup(name)
        if v is None:
            v = self.builtin(name)
        if v is None:
            raise KeyError(f"name {name!r} is used before it is defined")
        return v

    def mutated(self, reg: str) -> None:
        """Record an in-place write to `reg` (an accumulator or carried state)."""
        self.writes += 1
        self.last_write[reg] = self.writes

    # -- kernel-wide registers ---------------------------------------------
    def sreg(self, name: str, rng: tuple[int, int] | None) -> str:
        """A special register (%tid.x, ...) copied once at kernel entry."""
        if name not in self.special:
            r = self.ints.fresh(rng)
            self.b.ins("mov.u32", r, name, prologue=True)
            self.special[name] = r
        return self.special[name]

    def tid(self) -> Lin:
        return Lin.of(self.sreg("%tid.x", (0, self.spec.block[0] - 1)))

    def ctaid(self) -> Lin:
        return Lin.of(self.sreg("%ctaid.x", (0, self.spec.grid[0] - 1)))

    def builtin(self, name: str):
        """`lane`, `warp` and the CUDA built-ins, which the IR references without defining."""
        if name in ("lane", "warp"):
            key = f"%{name}"
            if key not in self.special:
                block = self.spec.block[0]
                tid = self.sreg("%tid.x", (0, block - 1))
                if name == "lane":
                    r = self.ints.fresh((0, min(31, block - 1)))
                    self.b.ins("and.b32", r, tid, "31", prologue=True)
                else:
                    r = self.ints.fresh((0, (block - 1) >> 5))
                    self.b.ins("shr.u32", r, tid, "5", prologue=True)
                self.special[key] = r
            return IntV(Lin.of(self.special[key]))
        if name in _SPECIAL_REGS:
            return IntV(Lin.of(self.sreg(_SPECIAL_REGS[name], None)))
        return None

    # -- f32 values --------------------------------------------------------
    def f32(self, name: str) -> str:
        """The f32 register holding IR value `name`."""
        v = self.lookup(name)
        if isinstance(v, FloatV):
            return v.reg
        if isinstance(v, IntV):
            return self.int_to_f32(v.lin)
        raise TypeError(f"{name!r} is not a number")

    def int_to_f32(self, lin: Lin) -> str:
        r = self.b.reg("f32")
        if lin.is_const:
            self.b.ins("mov.f32", r, f32_imm(lin.const))
        else:
            self.b.ins("cvt.rn.f32.s32", r, self.ints.reg_of(lin))
        return r

    def const_f32(self, x: float) -> str:
        r = self.b.reg("f32")
        self.b.ins("mov.f32", r, f32_imm(x))
        return r
