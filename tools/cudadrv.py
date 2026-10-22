"""A minimal CUDA driver API binding (ctypes + numpy) for running PTX on a local GPU.

The driver JIT-compiles PTX itself, so this needs an NVIDIA driver and no CUDA
toolkit. Works on Windows (nvcuda.dll) and Linux / WSL (libcuda.so.1). Testing only.
"""

from __future__ import annotations

import ctypes
import sys

import numpy as np

_lib = ctypes.WinDLL("nvcuda.dll") if sys.platform == "win32" else ctypes.CDLL("libcuda.so.1")

_JIT_INFO_LOG_BUFFER, _JIT_INFO_LOG_BUFFER_SIZE = 3, 4
_JIT_ERROR_LOG_BUFFER, _JIT_ERROR_LOG_BUFFER_SIZE = 5, 6
_FUNC_ATTR_MAX_DYNAMIC_SHARED = 8


class CudaError(RuntimeError):
    pass


def _check(code: int, what: str, log: str = "") -> None:
    if code == 0:
        return
    msg = ctypes.c_char_p()
    _lib.cuGetErrorString(code, ctypes.byref(msg))
    text = msg.value.decode() if msg.value else "unknown error"
    raise CudaError(f"{what}: {text} ({code})" + (f"\n{log}" if log else ""))


class Gpu:
    def __init__(self, ordinal: int = 0) -> None:
        _check(_lib.cuInit(0), "cuInit")
        self.dev = ctypes.c_int()
        _check(_lib.cuDeviceGet(ctypes.byref(self.dev), ordinal), "cuDeviceGet")
        self.ctx = ctypes.c_void_p()
        _check(_lib.cuDevicePrimaryCtxRetain(ctypes.byref(self.ctx), self.dev), "cuDevicePrimaryCtxRetain")
        _check(_lib.cuCtxSetCurrent(self.ctx), "cuCtxSetCurrent")
        self._allocs: list[ctypes.c_uint64] = []

    def name(self) -> str:
        buf = ctypes.create_string_buffer(256)
        _lib.cuDeviceGetName(buf, 256, self.dev)
        return buf.value.decode()

    def load(self, ptx: str) -> ctypes.c_void_p:
        """JIT the PTX; a failure raises with the driver's ptxas log."""
        size = 1 << 15
        info, err = ctypes.create_string_buffer(size), ctypes.create_string_buffer(size)
        opts = (ctypes.c_int * 4)(
            _JIT_INFO_LOG_BUFFER, _JIT_INFO_LOG_BUFFER_SIZE, _JIT_ERROR_LOG_BUFFER, _JIT_ERROR_LOG_BUFFER_SIZE
        )
        vals = (ctypes.c_void_p * 4)(
            ctypes.cast(info, ctypes.c_void_p),
            ctypes.c_void_p(size),
            ctypes.cast(err, ctypes.c_void_p),
            ctypes.c_void_p(size),
        )
        mod = ctypes.c_void_p()
        code = _lib.cuModuleLoadDataEx(ctypes.byref(mod), ptx.encode() + b"\0", 4, opts, vals)
        _check(code, "PTX JIT", err.value.decode(errors="replace"))
        return mod

    def function(self, module, name: str) -> ctypes.c_void_p:
        fn = ctypes.c_void_p()
        _check(_lib.cuModuleGetFunction(ctypes.byref(fn), module, name.encode()), f"entry {name!r}")
        return fn

    def upload(self, array: np.ndarray) -> ctypes.c_uint64:
        array = np.ascontiguousarray(array)
        ptr = self.zeros(array.nbytes)
        if array.nbytes:
            _check(
                _lib.cuMemcpyHtoD_v2(ptr, array.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(array.nbytes)),
                "cuMemcpyHtoD",
            )
        return ptr

    def zeros(self, nbytes: int) -> ctypes.c_uint64:
        ptr = ctypes.c_uint64()
        _check(_lib.cuMemAlloc_v2(ctypes.byref(ptr), ctypes.c_size_t(max(nbytes, 4))), "cuMemAlloc")
        _check(_lib.cuMemsetD8_v2(ptr, 0, ctypes.c_size_t(max(nbytes, 4))), "cuMemsetD8")
        self._allocs.append(ptr)
        return ptr

    def download(self, ptr: ctypes.c_uint64, dtype, shape) -> np.ndarray:
        out = np.empty(shape, dtype=dtype)
        if out.nbytes:
            _check(
                _lib.cuMemcpyDtoH_v2(out.ctypes.data_as(ctypes.c_void_p), ptr, ctypes.c_size_t(out.nbytes)),
                "cuMemcpyDtoH",
            )
        return out

    def launch(self, fn, grid, block, smem_bytes: int, args: list) -> None:
        if smem_bytes > 48 * 1024:
            _check(_lib.cuFuncSetAttribute(fn, _FUNC_ATTR_MAX_DYNAMIC_SHARED, smem_bytes), "dynamic shared memory size")
        holders = [a if isinstance(a, ctypes._SimpleCData) else ctypes.c_uint64(a.value) for a in args]
        params = (ctypes.c_void_p * len(holders))(*[ctypes.cast(ctypes.byref(h), ctypes.c_void_p) for h in holders])
        gx, gy, gz = ([*list(grid), 1, 1])[:3]
        bx, by, bz = ([*list(block), 1, 1])[:3]
        _check(_lib.cuLaunchKernel(fn, gx, gy, gz, bx, by, bz, smem_bytes, None, params, None), "cuLaunchKernel")
        _check(_lib.cuCtxSynchronize(), "kernel execution")

    def free_all(self) -> None:
        for p in self._allocs:
            _lib.cuMemFree_v2(p)
        self._allocs.clear()
