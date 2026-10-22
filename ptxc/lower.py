"""Lower a Kernel IR statement tree to PTX.

The Lowerer walks the body once, in order, with one method per statement kind
(`s_Tile`, `s_Load`, ...). It keeps a chain of lexical scopes mapping IR names
to values:

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

import math
import sys
from dataclasses import dataclass

from ptxc import ops
from ptxc.index import IntGen, Lin, c_div, c_mod, is_pow2
from ptxc.kernel import DTYPE_BYTES, Buffer, KernelSpec, static_int
from ptxc.ptx import PtxBuilder, f32_imm

STATIC_SMEM_LIMIT = 48 * 1024
UNROLL_BUDGET = 4096  # max statements a loop the IR marks `unroll` may expand to
SMALL_LOOP_TRIPS = 16  # unmarked loops with at most this many trips are unrolled too ...
SMALL_LOOP_BUDGET = 256  # ... if that stays within this many statements

# C type spellings of Smem.dtype -> canonical dtype
_SMEM_DTYPES = {"float": "f32", "__half": "f16", "half": "f16", "double": "f64", "int": "i32"}

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


class Lowerer:
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

    # ======================================================================
    # driver
    # ======================================================================
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
        if kind not in ("Tile", "Loop", "StridedLoop", "Cond"):
            self.b.comment(_pretty(s))
        handler(s)

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

    # ======================================================================
    # values
    # ======================================================================
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

    # ======================================================================
    # expressions (indices, predicates, loop bounds)
    # ======================================================================
    def expr(self, e):
        """Evaluate an Expr to IntV / FloatV / a predicate (register name or bool)."""
        kind = type(e).__name__
        if kind == "Var":
            return self.lookup(e.name)
        if kind == "Literal":
            if e.dtype == "int" or (isinstance(e.value, int) and not isinstance(e.value, bool)):
                return IntV(Lin(const=int(e.value)))
            return FloatV(self.const_f32(float(e.value)))
        if kind == "Builtin":
            v = self.builtin(e.name)
            if v is None:
                raise NotImplementedError(f"builtin {e.name!r}")
            return v
        if kind == "BinaryExpr":
            return self.binary(e.op, self.expr(e.left), self.expr(e.right))
        if kind == "TernaryExpr":
            p = self.pred(self.expr(e.cond))
            if p is True:
                return self.expr(e.if_true)
            if p is False:
                return self.expr(e.if_false)
            a, c = self.expr(e.if_true), self.expr(e.if_false)
            if isinstance(a, IntV) and isinstance(c, IntV):
                return IntV(self.ints.select(p, a.lin, c.lin))
            r = self.b.reg("f32")
            self.b.ins("selp.f32", r, self.as_f32(a), self.as_f32(c), p)
            return FloatV(r)
        if kind == "CastExpr":
            v = self.expr(e.expr)
            if str(e.dtype) in ("int", "i32", "long long", "i64"):
                if isinstance(v, IntV):
                    return v
                r = self.ints.fresh()
                self.b.ins("cvt.rzi.s32.f32", r, self.as_f32(v))
                return IntV(Lin.of(r))
            return FloatV(self.as_f32(v))
        if kind == "FlatIndex":
            return IntV(self.flatten(self.buffer(e.buffer), e.index))
        if kind == "FuncCallExpr":
            args = [self.as_f32(self.expr(a)) for a in e.args]
            r = self.b.reg("f32")
            ops.emit_op(self.b, e.name, args, r)
            return FloatV(r)
        raise NotImplementedError(f"expression {kind} is not supported")

    def as_f32(self, v) -> str:
        if isinstance(v, FloatV):
            return v.reg
        if isinstance(v, IntV):
            return self.int_to_f32(v.lin)
        r = self.b.reg("f32")  # a predicate used as a number
        if isinstance(v, bool):
            self.b.ins("mov.f32", r, f32_imm(1.0 if v else 0.0))
        else:
            self.b.ins("selp.f32", r, f32_imm(1.0), f32_imm(0.0), v)
        return r

    def as_int(self, v) -> Lin:
        if isinstance(v, IntV):
            return v.lin
        raise TypeError(f"expected an integer expression, got {v!r}")

    def pred(self, v) -> str | bool:
        if isinstance(v, (bool, str)):
            return v
        if isinstance(v, IntV):
            return self.ints.compare("!=", v.lin, Lin())
        p = self.b.reg("pred")
        self.b.ins("setp.ne.f32", p, v.reg, f32_imm(0.0))
        return p

    def binary(self, op: str, a, c):
        if op in ("&&", "||"):
            pa, pc = self.pred(a), self.pred(c)
            if isinstance(pa, bool) or isinstance(pc, bool):
                const, other = (pa, pc) if isinstance(pa, bool) else (pc, pa)
                if op == "&&":
                    return other if const else False
                return True if const else other
            p = self.b.reg("pred")
            self.b.ins("and.pred" if op == "&&" else "or.pred", p, pa, pc)
            return p
        if isinstance(a, IntV) and isinstance(c, IntV):
            x, y = a.lin, c.lin
            if op == "+":
                return IntV(x + y)
            if op == "-":
                return IntV(x - y)
            if op == "*":
                return IntV(self.ints.mul(x, y))
            if op in ("/", "//"):
                return IntV(self.ints.divmod_dyn(x, y, "div"))
            if op == "%":
                return IntV(self.ints.divmod_dyn(x, y, "mod"))
            if op in ("<", "<=", ">", ">=", "==", "!="):
                return self.ints.compare(op, x, y)
            if op in ("^", "&", "|"):
                return IntV(self.ints.bitop(op, x, y))
            raise NotImplementedError(f"integer operator {op!r}")
        # floating-point expression
        fa, fc = self.as_f32(a), self.as_f32(c)
        if op in ("<", "<=", ">", ">=", "==", "!="):
            cmp = {"<": "lt", "<=": "le", ">": "gt", ">=": "ge", "==": "eq", "!=": "ne"}[op]
            p = self.b.reg("pred")
            self.b.ins(f"setp.{cmp}.f32", p, fa, fc)
            return p
        name = {"+": "add", "-": "subtract", "*": "multiply", "/": "divide"}.get(op)
        if name is None:
            raise NotImplementedError(f"float operator {op!r}")
        r = self.b.reg("f32")
        ops.emit_op(self.b, name, [fa, fc], r)
        return FloatV(r)

    # ======================================================================
    # buffers and addresses
    # ======================================================================
    def buffer(self, name: str) -> Buffer:
        if name not in self.buffers:
            raise KeyError(f"unknown buffer {name!r}")
        return self.buffers[name]

    def flatten(self, buf: Buffer, index) -> Lin:
        """Row-major element offset of `buf[index]`."""
        idx = [self.as_int(self.expr(e)) for e in index]
        shape = buf.shape
        if len(idx) != len(shape):
            if len(idx) == 1:  # already flat
                return idx[0]
            if not idx:
                return Lin()
            raise ValueError(f"{buf.name}: {len(idx)} indices for shape {shape}")
        flat = Lin()
        stride = 1
        for i, extent in zip(reversed(idx), reversed(shape)):
            flat = flat + i.scale(stride)
            stride *= extent
        return flat

    def global_ptr(self, buf: Buffer) -> str:
        if buf.name not in self.global_base:
            if buf.name not in self.spec.params:
                raise KeyError(f"buffer {buf.name!r} is not a kernel parameter")
            raw, ptr = self.b.reg("u64"), self.b.reg("u64")
            self.b.ins("ld.param.u64", raw, f"[{self.param_name(buf.name)}]", prologue=True)
            self.b.ins("cvta.to.global.u64", ptr, raw, prologue=True)
            self.global_base[buf.name] = ptr
        return self.global_base[buf.name]

    def address(self, buf: Buffer, flat: Lin) -> str:
        """`[reg+imm]` operand for element `flat` of `buf`; constants go to the immediate."""
        byte_off = flat.const * buf.esize
        var = flat.var_part()
        if buf.space == "shared":
            base = self.smem_base[buf.name]
            if var.is_const:
                reg = base
            else:
                key = ("saddr", buf.name, var.key())
                reg = self.scope.cse_get(key)
                if reg is None:
                    reg = self.ints.fresh()
                    self.b.ins("mad.lo.s32", reg, self.ints.reg_of(var), str(buf.esize), base)
                    self._cse_put(key, reg)
        else:
            base = self.global_ptr(buf)
            if var.is_const:
                reg = base
            else:
                key = ("gaddr", buf.name, var.key())
                reg = self.scope.cse_get(key)
                if reg is None:
                    # the 64-bit byte offset is shared by every buffer indexed the same way
                    wkey = ("wide", var.key(), buf.esize)
                    wide = self.scope.cse_get(wkey)
                    if wide is None:
                        wide = self.b.reg("u64")
                        self.b.ins("mul.wide.s32", wide, self.ints.reg_of(var), str(buf.esize))
                        self._cse_put(wkey, wide)
                    reg = self.b.reg("u64")
                    self.b.ins("add.s64", reg, base, wide)
                    self._cse_put(key, reg)
        return f"[{reg}+{byte_off}]" if byte_off else f"[{reg}]"

    # ======================================================================
    # shared memory
    # ======================================================================
    def _plan_shared_memory(self, body) -> None:
        """Static `.shared` arrays, or one dynamic pool when they exceed the static limit."""
        decls = list(_iter_stmts(body, "Smem"))
        total, offsets = 0, {}
        for s in decls:
            if s.name in offsets:
                continue
            esize = DTYPE_BYTES[_smem_dtype(s)]
            align = max(int(s.align or 0), esize)
            total = -(-total // align) * align
            offsets[s.name] = total
            total += math.prod(int(e) for e in s.extents) * esize
        self.dynamic_smem = total > STATIC_SMEM_LIMIT or (self.spec.smem_bytes > 0 and bool(decls))
        if self.dynamic_smem:
            self.smem_offset = offsets
            self.b.shared_decls.append(".extern .shared .align 16 .b8 smem_pool[];")
            if self.spec.smem_bytes and total > self.spec.smem_bytes:
                print(
                    f"ptxc: warning: shared memory needs {total} bytes, launch provides {self.spec.smem_bytes}",
                    file=sys.stderr,
                )

    def s_Smem(self, s) -> None:
        if s.name in self.smem_base:
            return
        dtype = _smem_dtype(s)
        buf = Buffer(s.name, tuple(int(e) for e in s.extents), dtype, space="shared")
        self.buffers[s.name] = buf
        r = self.ints.fresh()
        if self.dynamic_smem:
            self.b.ins("mov.u32", r, "smem_pool", prologue=True)
            if self.smem_offset[s.name]:
                self.b.ins("add.u32", r, r, str(self.smem_offset[s.name]), prologue=True)
        else:
            sym = "smem_" + "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in s.name)
            nbytes = math.prod(buf.shape) * buf.esize
            align = max(int(s.align or 0), buf.esize)
            self.b.shared_decls.append(f".shared .align {align} .b8 {sym}[{nbytes}];")
            self.b.ins("mov.u32", r, sym, prologue=True)
        self.smem_base[s.name] = r

    # ======================================================================
    # statements: launch structure
    # ======================================================================
    def s_Tile(self, t) -> None:
        extents = [static_int(a.extent) for a in t.axes]
        n = math.prod(extents)
        cells = t.block_threads or self.spec.block[0]
        self.b.comment(_head(t))
        # The launch geometry is part of the contract, so blockDim is a constant here.
        if t.aux_threads:
            gid = self.ctaid().scale(t.block_threads) + self.ints.mod(self.tid(), t.block_threads)
        else:
            gid = self.ctaid().scale(self.spec.block[0]) + self.tid()
        # Threads past the iteration space (p00: 64 cells, 256 threads) must not run.
        if t.aux_threads or n % cells != 0 or self.spec.threads_launched > n:
            past = self.ints.compare(">=", gid, Lin(const=n))
            if past is True:
                return
            if past is not False:
                self.used_exit = True
                self.b.ins("bra", self.exit_label, guard=f"@{past}")
        self.push()
        for axis, value in zip(t.axes, self._decode_axes(t, extents, gid)):
            self.define(axis.name, IntV(value))
        self.body(t.body)
        self.pop()

    def _decode_axes(self, t, extents: list[int], gid: Lin) -> list[Lin]:
        """axis_i = (gid / stride_i) % extent_i, row-major; optional grouped rasterization."""
        strides = [math.prod(extents[i + 1 :]) for i in range(len(extents))]
        vals: list[Lin | None] = [None] * len(extents)
        names = [a.name for a in t.axes]
        raster = None
        if t.raster_group and t.raster_axes and all(a in names for a in t.raster_axes):
            mi, ni = names.index(t.raster_axes[0]), names.index(t.raster_axes[1])
            if ni == mi + 1:
                gi, oi = (mi, ni) if t.raster_orient == "m" else (ni, mi)
                g = min(t.raster_group, extents[gi])
                if extents[gi] > 1 and g > 1:
                    raster = (mi, ni, gi, oi, g)
        if raster is not None:
            mi, ni, gi, oi, g = raster
            ea, eb = extents[gi], extents[oi]
            sub = self.ints.div(gid, strides[ni])
            if mi != 0:
                sub = self.ints.mod(sub, ea * eb)
            within = self.ints.mod(sub, g * eb)
            if ea % g == 0:
                a_val = self.ints.div(sub, g * eb).scale(g) + self.ints.mod(sub, g)
                b_val = self.ints.div(within, g)
            else:
                grp = self.ints.div(sub, g * eb)
                size = self.ints.minimum(Lin(const=g), Lin(const=ea) - grp.scale(g))
                a_val = grp.scale(g) + self.ints.divmod_dyn(within, size, "mod")
                b_val = self.ints.divmod_dyn(within, size, "div")
            vals[gi], vals[oi] = a_val, b_val
        for i, e in enumerate(extents):
            if vals[i] is not None:
                continue
            if e == 1:
                vals[i] = Lin()
                continue
            v = self.ints.div(gid, strides[i])
            vals[i] = v if i == 0 else self.ints.mod(v, e)
        return vals

    def s_Sync(self, s) -> None:
        if s.warp:
            self.b.ins("bar.warp.sync", "-1")
        elif s.barrier_id == 0:
            self.b.ins("bar.sync", "0")
        else:
            self.b.ins("bar.sync", str(s.barrier_id), str(s.count))

    # ======================================================================
    # statements: scalar compute
    # ======================================================================
    def s_Load(self, s) -> None:
        buf = self.buffer(s.input)
        flat = self.flatten(buf, s.index)
        regs = [self.b.reg("f32") for _ in s.names]
        self.load_into(buf, flat, regs)
        for name, r in zip(s.names, regs):
            self.define(name, FloatV(r))

    def load_into(self, buf: Buffer, flat: Lin, regs: list[str]) -> None:
        space = "shared" if buf.space == "shared" else ("global.nc" if buf.readonly else "global")
        for start, width in _vector_chunks(len(regs), buf.esize):
            addr = self.address(buf, flat + start)
            chunk = regs[start : start + width]
            if buf.dtype == "f32":
                vec = f".v{width}" if width > 1 else ""
                dst = "{" + ", ".join(chunk) + "}" if width > 1 else chunk[0]
                self.b.ins(f"ld.{space}{vec}.f32", dst, addr)
            elif buf.dtype == "f16":
                halves = [self.b.reg("b16") for _ in chunk]
                vec = f".v{width}" if width > 1 else ""
                dst = "{" + ", ".join(halves) + "}" if width > 1 else halves[0]
                self.b.ins(f"ld.{space}{vec}.b16", dst, addr)
                for h, r in zip(halves, chunk):
                    self.b.ins("cvt.f32.f16", r, h)
            else:
                raise NotImplementedError(f"loads of {buf.dtype} buffers")

    def s_Write(self, s) -> None:
        buf = self.buffer(s.output)
        flat = self.flatten(buf, s.index)
        regs = [self.f32(v) for v in s.values]
        space = "shared" if buf.space == "shared" else "global"
        if s.atomic:
            if buf.dtype != "f32":
                raise NotImplementedError(f"atomic writes to {buf.dtype}")
            for k, r in enumerate(regs):
                self.b.ins(f"red.{space}.add.f32", self.address(buf, flat + k), r)
            return
        for start, width in _vector_chunks(len(regs), buf.esize):
            addr = self.address(buf, flat + start)
            chunk = regs[start : start + width]
            vec = f".v{width}" if width > 1 else ""
            if buf.dtype == "f32":
                src = "{" + ", ".join(chunk) + "}" if width > 1 else chunk[0]
                self.b.ins(f"st.{space}{vec}.f32", addr, src)
            elif buf.dtype == "f16":
                halves = [self.b.reg("b16") for _ in chunk]
                for h, r in zip(halves, chunk):
                    self.b.ins("cvt.rn.f16.f32", h, r)
                src = "{" + ", ".join(halves) + "}" if width > 1 else halves[0]
                self.b.ins(f"st.{space}{vec}.b16", addr, src)
            else:
                raise NotImplementedError(f"writes to {buf.dtype} buffers")

    def s_Assign(self, s) -> None:
        # bool masks travel as f32 0.0 / 1.0, Emmy's own convention (`where` tests nonzero)
        if s.dtype is not None and s.dtype.name not in ("f32", "f16", "bf16", "bool"):
            raise NotImplementedError(f"Assign {s.name} at dtype {s.dtype.name}")
        args = [self.f32(a) for a in s.args]
        if s.name in self.inplace:
            dst = self.lookup(s.name).reg
            if s.op.name in ("copy", "pad") and args[0] == dst:
                return
        else:
            dst = self.b.reg("f32")
        ops.emit_op(self.b, s.op.name, args, dst)
        if s.name in self.inplace:
            self.mutated(dst)
        else:
            self.define(s.name, FloatV(dst))
            if ops.canonical(s.op.name) == "multiply":
                self.scope.cse[("product", dst)] = (args[0], args[1], self.writes)

    def mutated(self, reg: str) -> None:
        """Record an in-place write to `reg` (an accumulator or carried state)."""
        self.writes += 1
        self.last_write[reg] = self.writes

    def _fusable_product(self, value: str) -> tuple[str, str] | None:
        """The factors of `value` if it is a multiply from this same scope whose factors
        have not been overwritten since, so `acc + value` may become one fma."""
        entry = self.scope.cse.get(("product", value))
        if entry is None:
            return None
        a, b, at = entry
        if self.last_write.get(a, 0) > at or self.last_write.get(b, 0) > at:
            return None
        return a, b

    def s_Accum(self, s) -> None:
        acc = self.scope.lookup(s.name)
        if acc is None:  # no seeding loop: start from the identity here
            acc = FloatV(self.const_f32(_identity(s.op)))
            self.define(s.name, acc)
        base = acc.reg if s.base is None else self.f32(s.base)
        value = self.f32(s.value)
        factors = self._fusable_product(value) if ops.canonical(s.op.name) == "add" else None
        if factors is not None:
            # acc = base + a * b as one fused multiply-add (the separate mul becomes dead code)
            self.b.ins("fma.rn.f32", acc.reg, factors[0], factors[1], base)
        else:
            ops.emit_op(self.b, s.op.name, [base, value], acc.reg)
        self.mutated(acc.reg)

    def s_Let(self, s) -> None:
        v = self.expr(s.value)
        if s.dtype is not None and s.dtype.name in ("f32", "f16", "bf16"):
            v = FloatV(self.as_f32(v))
        elif not isinstance(v, (IntV, FloatV)):
            v = FloatV(self.as_f32(v))  # a predicate bound to a name
        self.define(s.name, v)

    def s_Init(self, s) -> None:
        self.define(s.name, FloatV(self.const_f32(float(s.identity))))

    # ======================================================================
    # statements: control flow
    # ======================================================================
    def seed_accums(self, loop) -> None:
        """Seed the loop's own Accums with their identity, in the enclosing scope."""
        if not loop.seed:
            return
        seen = set()
        for s in loop.body:
            if type(s).__name__ == "Accum" and s.name not in seen:
                seen.add(s.name)
                self.define(s.name, FloatV(self.const_f32(_identity(s.op))))

    def s_Loop(self, s) -> None:
        self.b.comment(_head(s))
        self.seed_accums(s)
        extent = static_int(s.axis.extent)
        if extent <= 0:
            return
        self._counted_loop(s.axis.name, Lin(), Lin(const=extent), 1, s.body, unroll=s.unroll)

    def s_StridedLoop(self, s) -> None:
        self.b.comment(_head(s))
        self.seed_accums(s)
        start = self.as_int(self.expr(s.start))
        step = self.as_int(self.expr(s.step))
        end = self.as_int(self.expr(s.end)) if s.end is not None else Lin(const=static_int(s.axis.extent))
        if not step.is_const or step.const <= 0:
            raise NotImplementedError("StridedLoop with a non-constant step")
        self._counted_loop(s.axis.name, start, end, step.const, s.body, unroll=s.unroll)

    def _counted_loop(self, var: str, start: Lin, end: Lin, step: int, body, *, unroll: bool) -> None:
        """for (var = start; var < end; var += step) body

        Three shapes, cheapest first:
        * fully unrolled, when the trip count is small and known from the value ranges:
          `var` becomes `start + k*step`, an affine form, so it folds into every address;
          trailing copies that only some threads run get a predicate;
        * partially unrolled by U (a main loop doing U copies per trip, then a remainder
          loop): the U loads of one trip are independent and can be in flight together;
        * a plain bottom-tested loop.
        A body with a barrier only gets the shapes in which every thread does the same.
        """
        size = _size(body)
        uniform_only = _has_barrier(body)
        start_rng = start.range(self.ints.ranges)
        if start_rng is not None and end.is_const:
            min_trips = max(0, -(-(end.const - start_rng[1]) // step))
            max_trips = max(0, -(-(end.const - start_rng[0]) // step))
            if max_trips == 0:
                return
            budget = UNROLL_BUDGET if unroll else SMALL_LOOP_BUDGET
            can_guard = not uniform_only or min_trips == max_trips
            if can_guard and (max_trips == 1 or (max_trips <= SMALL_LOOP_TRIPS or unroll) and max_trips * size <= budget):
                self._unrolled_copies(var, start, end, step, body, min_trips, max_trips)
                return
        if self.ints.compare("<", start, end) is False:
            return
        factor = 1 if uniform_only or not end.is_const else _unroll_factor(size)
        lo = start_rng[0] if start_rng is not None else None
        i = self.ints.fresh()  # the counter; it only has a range inside the bodies below
        self.b.ins("mov.u32", i, self.ints.operand(start))
        entry = start  # what the next entry test compares: `start` itself still has its range
        if factor > 1:
            # main loop: U copies per trip while the last of them is still in range
            last = end.const - (factor - 1) * step
            enters = self.ints.compare("<", start, Lin(const=last))
            if enters is not False:
                main_end = self.b.label("main_end")
                if enters is not True:
                    self.b.ins("bra", main_end, guard=f"@!{enters}")
                head = self.b.label("main")
                self.b.place(head)
                self._counter_range(i, lo, last - 1)
                self.push()  # the trip's scope: copies share values here, never past the loop
                for k in range(factor):
                    self.push(shares_values=True)
                    self.define(var, IntV(Lin.of(i) + k * step))
                    self.body(body)
                    self.pop()
                self.pop()
                self.ints.ranges.pop(i, None)
                self.b.ins("add.s32", i, i, str(factor * step))
                again = self.b.reg("pred")
                self.b.ins("setp.lt.s32", again, i, str(last))
                self.b.ins("bra", head, guard=f"@{again}")
                self.b.place(main_end)
                entry = Lin.of(i)
        end_rng = end.range(self.ints.ranges)
        self._plain_loop(var, i, entry, end, step, body, body_hi=end_rng[1] - 1 if end_rng else None, lo=lo)

    def _counter_range(self, i: str, lo: int | None, hi: int | None) -> None:
        """Give loop counter `i` its in-body range (only while the body is emitted)."""
        if lo is not None and hi is not None:
            self.ints.ranges[i] = (lo, hi)

    def _unrolled_copies(self, var: str, start: Lin, end: Lin, step: int, body, min_trips: int, max_trips: int) -> None:
        for k in range(max_trips):
            value = start + k * step
            skip = None
            if k >= min_trips:  # only some threads get this far
                p = self.ints.compare("<", value, end)
                if p is False:
                    break
                if p is not True:
                    skip = self.b.label("tail_skip")
                    self.b.ins("bra", skip, guard=f"@!{p}")
            # an unguarded copy is straight-line code, so later copies may reuse its values
            self.push(shares_values=skip is None)
            self.define(var, IntV(value))
            self.body(body)
            self.pop()
            if skip:
                self.b.place(skip)

    def _plain_loop(self, var: str, i: str, entry: Lin, end: Lin, step: int, body, *, body_hi, lo) -> None:
        """Bottom-tested loop over counter `i`, which already holds its first value (`entry`)."""
        done = self.b.label("loop_end")
        enters = self.ints.compare("<", entry, end)
        if enters is False:
            return
        if enters is not True:
            self.b.ins("bra", done, guard=f"@!{enters}")
        head = self.b.label("loop")
        self.b.place(head)
        self._counter_range(i, lo, body_hi)
        self.push()
        self.define(var, IntV(Lin.of(i)))
        self.body(body)
        self.pop()
        self.ints.ranges.pop(i, None)
        self.b.ins("add.s32", i, i, str(step))
        again = self.b.reg("pred")
        self.b.ins("setp.lt.s32", again, i, self.ints.operand(end))
        self.b.ins("bra", head, guard=f"@{again}")
        self.b.place(done)

    def s_Cond(self, s) -> None:
        self.b.comment(f"if ({s.cond.pretty()})")
        p = self.pred(self.expr(s.cond))
        if isinstance(p, bool):
            self.push()
            self.body(s.body if p else s.else_body)
            self.pop()
            return
        else_label, end_label = self.b.label("else"), self.b.label("endif")
        self.b.ins("bra", else_label if s.else_body else end_label, guard=f"@!{p}")
        self.push()
        self.body(s.body)
        self.pop()
        if s.else_body:
            self.b.ins("bra.uni", end_label)
            self.b.place(else_label)
            self.push()
            self.body(s.else_body)
            self.pop()
        self.b.place(end_label)

    # ======================================================================
    # statements: cross-thread combines
    # ======================================================================
    def combine(self, program, states) -> None:
        """Run a combine program (Assigns) that updates `states` in place."""
        saved = self.inplace
        self.inplace = frozenset(states)
        for a in program:
            self.s_Assign(a)
        self.inplace = saved

    def s_WarpShuffle(self, s) -> None:
        mask = self.ints.fresh()
        self.b.ins("activemask.b32", mask)
        step = int(s.length) // 2
        while step > 0:
            self.push()
            for st, sb in zip(s.state, s.state_b):
                other = self.b.reg("f32")
                self.b.ins("shfl.sync.bfly.b32", other, self.f32(st), str(step), "31", mask)
                self.define(sb, FloatV(other))
            self.combine(s.combine_states, s.state)
            self.pop()
            step //= 2

    def s_TreeHalve(self, s) -> None:
        t = self.as_int(self.lookup(s.tid_var))
        bufs = [self.buffer(name) for name in s.bufs]
        if s.inner is not None:
            inner_var, scale = s.inner
            inner = self.as_int(self.lookup(inner_var))
            slot, root = t.scale(scale) + inner, inner
        else:
            scale, slot, root = 1, t, Lin()
        sync = ("bar.sync", "0") if s.barrier_id == 0 else ("bar.sync", str(s.barrier_id), str(s.barrier_count))
        half = int(s.length) // 2
        while half > 0:
            active = self.ints.compare("<", t, Lin(const=half))
            skip = self.b.label("tree_skip")
            if active is not True:
                self.b.ins("bra", skip, guard=f"@!{active}")
            if active is not False:
                self.push()
                for buf, st, sb in zip(bufs, s.state, s.state_b):
                    mine, theirs = self.b.reg("f32"), self.b.reg("f32")
                    self.load_into(buf, slot, [mine])
                    self.load_into(buf, slot + half * scale, [theirs])
                    self.define(st, FloatV(mine))  # shadows the thread's own state
                    self.define(sb, FloatV(theirs))
                self.combine(s.combine_states, s.state)
                for buf, st in zip(bufs, s.state):
                    self.b.ins("st.shared.f32", self.address(buf, slot), self.f32(st))
                self.pop()
            if active is not True:
                self.b.place(skip)
            self.b.ins(*sync)
            half //= 2
        # every thread picks up the reduced state
        for buf, st in zip(bufs, s.state):
            reg = self.lookup(st).reg
            self.load_into(buf, root, [reg])
            self.mutated(reg)

    # ======================================================================
    # statements: asynchronous copies into shared memory
    # ======================================================================
    def s_CpAsyncCopy(self, s) -> None:
        if s.swizzle not in (None, "NONE"):
            raise NotImplementedError(f"cp.async with swizzle {s.swizzle}")
        smem = self.buffer(s.smem)
        dst = self.flatten(smem, s.smem_index)
        if s.lane_index is not None:
            row, col = (self.as_int(self.expr(e)) for e in s.lane_index)
            dst = dst + row.scale(smem.shape[-1]) + col
        src_buf = self.buffer(s.src)
        src = self.flatten(src_buf, s.src_index)
        n = int(s.nbytes)
        cache = "cg" if n == 16 else "ca"
        self.b.ins(f"cp.async.{cache}.shared.global", self.address(smem, dst), self.address(src_buf, src), str(n))

    def s_CpAsyncCommit(self, s) -> None:
        self.b.ins("cp.async.commit_group")

    def s_CpAsyncWait(self, s) -> None:
        self.b.ins("cp.async.wait_group", str(int(s.group)))


# ==========================================================================
# helpers
# ==========================================================================
def _identity(op) -> float:
    ident = op.identity
    if ident is None:
        raise ValueError(f"reduction op {op.name!r} has no identity")
    return float(ident)


def _smem_dtype(s) -> str:
    dt = _SMEM_DTYPES.get(str(s.dtype).strip())
    if dt is None:
        raise NotImplementedError(f"shared memory of type {s.dtype!r}")
    return dt


def _vector_chunks(n: int, esize: int):
    """Split n consecutive elements into legal vector accesses (v2/v4, at most 16 bytes)."""
    start = 0
    while start < n:
        left = n - start
        width = 1
        for w in (4, 2):
            if left >= w and w * esize <= 16:
                width = w
                break
        yield start, width
        start += width


def _iter_stmts(body, kind: str):
    for s in body:
        if type(s).__name__ == kind:
            yield s
        for field in ("body", "else_body"):
            inner = getattr(s, field, None)
            if inner:
                yield from _iter_stmts(inner, kind)


_BARRIERS = ("Sync", "WarpShuffle", "TreeHalve", "CpAsyncWait")


def _has_barrier(body) -> bool:
    return any(next(_iter_stmts(body, kind), None) is not None for kind in _BARRIERS)


def _unroll_factor(size: int) -> int:
    """Copies per trip of a partially unrolled loop: more for small bodies."""
    if size <= 4:
        return 8
    if size <= 16:
        return 4
    if size <= 64:
        return 2
    return 1


def _size(body) -> int:
    n = 0
    for s in body:
        n += 1
        for field in ("body", "else_body"):
            inner = getattr(s, field, None)
            if inner:
                n += _size(inner)
    return n


def _pretty(s) -> str:
    try:
        return "\n".join(s.pretty())
    except Exception:  # noqa: BLE001 - comments are best effort
        return type(s).__name__


def _head(s) -> str:
    lines = _pretty(s).splitlines()
    return lines[0] if lines else type(s).__name__


def compile_kernel(spec: KernelSpec) -> str:
    return Lowerer(spec).run()
