"""Load one kernel: the Kernel IR (through Emmy's own classes) plus its meta.json.

Everything the lowering needs about the outside world ends up in a KernelSpec:
the entry name, the parameter order, every global buffer's shape and dtype,
and the launch geometry the kernel will run under.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from emmy.compiler.graph import Graph

# Element size in bytes of every dtype a buffer may have.
DTYPE_BYTES = {"f32": 4, "f16": 2, "bf16": 2, "f64": 8, "i32": 4, "i64": 8}


@dataclass
class Buffer:
    name: str
    shape: tuple[int, ...]
    dtype: str = "f32"
    space: str = "global"  # "global" or "shared"
    readonly: bool = False  # global input never written by this kernel -> ld.global.nc

    @property
    def esize(self) -> int:
        return DTYPE_BYTES[self.dtype]


@dataclass
class KernelSpec:
    name: str
    body: tuple  # the KernelOp's statement tree
    params: list[str]  # buffer names in launch argument order
    buffers: dict[str, Buffer]
    grid: tuple[int, int, int] = (1, 1, 1)
    block: tuple[int, int, int] = (1, 1, 1)
    smem_bytes: int = 0  # dynamic shared memory the launch provides
    target: str = "sm_89"
    zero_outputs: list[str] = field(default_factory=list)

    @property
    def threads_launched(self) -> int:
        return math.prod(self.grid) * math.prod(self.block)


def static_int(dim) -> int:
    """A Dim / Literal / int extent as a plain int (every problem is static)."""
    if isinstance(dim, int):
        return dim
    if hasattr(dim, "as_static"):
        return int(dim.as_static())
    if hasattr(dim, "expr") and hasattr(dim.expr, "value"):
        return int(dim.expr.value)
    raise ValueError(f"non-static extent {dim!r}")


def _find_kernel_node(graph: Graph, ir_path: Path):
    kernels = [n for n in graph.nodes.values() if type(n.op).__name__ == "KernelOp"]
    if not kernels:
        raise ValueError(f"{ir_path}: no KernelOp in the graph")
    if len(kernels) == 1:
        return kernels[0]
    stem = ir_path.name.removesuffix(".ir.json")
    for n in kernels:
        if n.op.name == stem:
            return n
    raise ValueError(f"{ir_path}: {len(kernels)} KernelOps and none is named {stem!r}")


def _find_meta_kernel(ir_path: Path, entry: str, node_id: str) -> dict | None:
    meta_path = ir_path.parent / "meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for k in meta.get("kernels", []):
        if k.get("entry") == entry or k.get("node") == node_id:
            return k
    return None


def load_kernel(ir_path: str | Path) -> KernelSpec:
    ir_path = Path(ir_path)
    graph = Graph.from_dict(json.loads(ir_path.read_text(encoding="utf-8")))
    node = _find_kernel_node(graph, ir_path)
    op = node.op
    name = op.name or node.id

    # Every node of the graph names a buffer: inputs, constants, this kernel's outputs.
    buffers: dict[str, Buffer] = {}
    for n in graph.nodes.values():
        for t in n.outputs:
            shape = tuple(static_int(d) for d in t.shape)
            buffers[t.name] = Buffer(t.name, shape, t.dtype.name)
    outputs = {t.name for t in node.outputs}
    for b in buffers.values():
        b.readonly = b.name not in outputs

    meta = _find_meta_kernel(ir_path, name, node.id)
    if meta is not None:
        params = list(meta["arg_order"])
        spec = KernelSpec(
            name=name,
            body=op.body,
            params=params,
            buffers=buffers,
            grid=tuple(meta["grid"]),
            block=tuple(meta["block"]),
            smem_bytes=int(meta.get("smem_bytes", 0)),
            zero_outputs=list(meta.get("zero_outputs", [])),
        )
        # meta.json also knows shapes/dtypes; trust it for anything the graph lacks.
        for a in meta.get("args", []):
            if a["name"] not in buffers:
                buffers[a["name"]] = Buffer(a["name"], tuple(a["shape"]), a["dtype"], readonly=not a.get("output"))
    else:
        # No meta.json: fall back to the node's inputs followed by its outputs.
        params = list(node.inputs) + [t.name for t in node.outputs]
        spec = KernelSpec(name=name, body=op.body, params=params, buffers=buffers)
    for p in spec.params:
        if p not in buffers:
            raise ValueError(f"parameter {p!r} is not a buffer of the graph")
    return spec
