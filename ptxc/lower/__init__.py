"""Lower a Kernel IR statement tree to PTX.

One mixin per statement family; `Lowerer` puts them together:

    base.py     values, lexical scopes, dispatch, special registers
    exprs.py    indices, predicates, loop bounds
    memory.py   buffers, addresses, shared memory, Load / Write / cp.async
    compute.py  Assign, Accum (+ fma fusion), Let, Init, Select
    control.py  Tile, Sync, loops (unrolling), Cond
    combine.py  WarpShuffle, TreeHalve
"""

from __future__ import annotations

from ptxc.kernel import KernelSpec
from ptxc.lower.base import FloatV, IntV, LowererBase, Scope
from ptxc.lower.combine import CombineMixin
from ptxc.lower.compute import ComputeMixin
from ptxc.lower.control import ControlMixin
from ptxc.lower.exprs import ExprMixin
from ptxc.lower.memory import MemoryMixin


class Lowerer(ControlMixin, CombineMixin, MemoryMixin, ComputeMixin, ExprMixin, LowererBase):
    """Kernel IR -> PTX for one kernel. `Lowerer(spec).run()` returns the PTX text."""


def compile_kernel(spec: KernelSpec) -> str:
    return Lowerer(spec).run()


__all__ = ["FloatV", "IntV", "Lowerer", "Scope", "compile_kernel"]
