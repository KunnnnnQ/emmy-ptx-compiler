"""Memory: buffers, addresses, shared memory, Load / Write and cp.async copies."""

from __future__ import annotations

import math
import sys

from ptxc.index import Lin
from ptxc.kernel import DTYPE_BYTES, Buffer
from ptxc.lower.base import FloatV
from ptxc.lower.walk import iter_stmts

STATIC_SMEM_LIMIT = 48 * 1024

# C type spellings of Smem.dtype -> canonical dtype
_SMEM_DTYPES = {"float": "f32", "__half": "f16", "half": "f16", "double": "f64", "int": "i32"}


def smem_dtype(s) -> str:
    dt = _SMEM_DTYPES.get(str(s.dtype).strip())
    if dt is None:
        raise NotImplementedError(f"shared memory of type {s.dtype!r}")
    return dt


def vector_chunks(n: int, esize: int):
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


def _symbol(prefix: str, name: str) -> str:
    return prefix + "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)


class MemoryMixin:
    # -- buffers and addresses ---------------------------------------------
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
        for i, extent in zip(reversed(idx), reversed(shape), strict=True):
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
        if var.is_const:
            reg = self.smem_base[buf.name] if buf.space == "shared" else self.global_ptr(buf)
        elif buf.space == "shared":
            reg = self._shared_address(buf, var)
        else:
            reg = self._global_address(buf, var)
        return f"[{reg}+{byte_off}]" if byte_off else f"[{reg}]"

    def _shared_address(self, buf: Buffer, var: Lin) -> str:
        key = ("saddr", buf.name, var.key())
        reg = self.scope.cse_get(key)
        if reg is None:
            reg = self.ints.fresh()
            self.b.ins("mad.lo.s32", reg, self.ints.reg_of(var), str(buf.esize), self.smem_base[buf.name])
            self._cse_put(key, reg)
        return reg

    def _global_address(self, buf: Buffer, var: Lin) -> str:
        base = self.global_ptr(buf)
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
        return reg

    # -- shared memory -----------------------------------------------------
    def _plan_shared_memory(self, body) -> None:
        """Static `.shared` arrays, or one dynamic pool when they exceed the static limit."""
        decls = list(iter_stmts(body, "Smem"))
        total, offsets = 0, {}
        for s in decls:
            if s.name in offsets:
                continue
            esize = DTYPE_BYTES[smem_dtype(s)]
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
        buf = Buffer(s.name, tuple(int(e) for e in s.extents), smem_dtype(s), space="shared")
        self.buffers[s.name] = buf
        r = self.ints.fresh()
        if self.dynamic_smem:
            self.b.ins("mov.u32", r, "smem_pool", prologue=True)
            if self.smem_offset[s.name]:
                self.b.ins("add.u32", r, r, str(self.smem_offset[s.name]), prologue=True)
        else:
            sym = _symbol("smem_", s.name)
            nbytes = math.prod(buf.shape) * buf.esize
            align = max(int(s.align or 0), buf.esize)
            self.b.shared_decls.append(f".shared .align {align} .b8 {sym}[{nbytes}];")
            self.b.ins("mov.u32", r, sym, prologue=True)
        self.smem_base[s.name] = r

    # -- loads and stores --------------------------------------------------
    def s_Load(self, s) -> None:
        buf = self.buffer(s.input)
        flat = self.flatten(buf, s.index)
        regs = [self.b.reg("f32") for _ in s.names]
        self.load_into(buf, flat, regs)
        for name, r in zip(s.names, regs, strict=True):
            self.define(name, FloatV(r))

    def load_into(self, buf: Buffer, flat: Lin, regs: list[str]) -> None:
        """Load consecutive elements from `flat` on into f32 registers, vectorized where legal."""
        space = "shared" if buf.space == "shared" else ("global.nc" if buf.readonly else "global")
        for start, width in vector_chunks(len(regs), buf.esize):
            addr = self.address(buf, flat + start)
            chunk = regs[start : start + width]
            vec = f".v{width}" if width > 1 else ""
            if buf.dtype == "f32":
                dst = "{" + ", ".join(chunk) + "}" if width > 1 else chunk[0]
                self.b.ins(f"ld.{space}{vec}.f32", dst, addr)
            elif buf.dtype == "f16":
                halves = [self.b.reg("b16") for _ in chunk]
                dst = "{" + ", ".join(halves) + "}" if width > 1 else halves[0]
                self.b.ins(f"ld.{space}{vec}.b16", dst, addr)
                for h, r in zip(halves, chunk, strict=True):
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
        for start, width in vector_chunks(len(regs), buf.esize):
            addr = self.address(buf, flat + start)
            chunk = regs[start : start + width]
            vec = f".v{width}" if width > 1 else ""
            if buf.dtype == "f32":
                src = "{" + ", ".join(chunk) + "}" if width > 1 else chunk[0]
                self.b.ins(f"st.{space}{vec}.f32", addr, src)
            elif buf.dtype == "f16":
                halves = [self.b.reg("b16") for _ in chunk]
                for h, r in zip(halves, chunk, strict=True):
                    self.b.ins("cvt.rn.f16.f32", h, r)
                src = "{" + ", ".join(halves) + "}" if width > 1 else halves[0]
                self.b.ins(f"st.{space}{vec}.b16", addr, src)
            else:
                raise NotImplementedError(f"writes to {buf.dtype} buffers")

    # -- asynchronous copies into shared memory ----------------------------
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
        cache = "cg" if n == 16 else "ca"  # .cg (L2 only) exists for 16-byte copies alone
        self.b.ins(f"cp.async.{cache}.shared.global", self.address(smem, dst), self.address(src_buf, src), str(n))

    def s_CpAsyncCommit(self, s) -> None:
        self.b.ins("cp.async.commit_group")

    def s_CpAsyncWait(self, s) -> None:
        self.b.ins("cp.async.wait_group", str(int(s.group)))
