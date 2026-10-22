"""ptxc: a compiler from Emmy Kernel IR to PTX.

Pipeline:  kernel.load_kernel (IR json + meta.json -> KernelSpec)
        -> lower.Lowerer     (walk the statement tree, emit instructions)
        -> ptx.PtxBuilder    (virtual registers, labels, final text)
"""

from ptxc.kernel import load_kernel
from ptxc.lower import compile_kernel

__all__ = ["compile_kernel", "load_kernel"]
