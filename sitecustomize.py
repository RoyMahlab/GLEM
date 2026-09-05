"""Make the CUDA 11 runtime libraries visible to the dynamic loader.

`dgl-cu111` links against ``libcudart.so.11.0``, ``libcublas.so.11`` and
``libcusparse.so.11`` by SONAME, and expects to find them on the system
(conda's ``cudatoolkit`` used to provide them).  This venv instead gets them
from the ``nvidia-*-cu11`` wheels, which drop the libraries in private
directories the loader knows nothing about::

    site-packages/nvidia/cublas/lib/libcublas.so.11

Without help, ``import dgl`` fails with::

    OSError: libcublas.so.11: cannot open shared object file

``dlopen`` resolves a DT_NEEDED entry against objects that are *already*
loaded before it searches the filesystem, so loading each library here by
absolute path is enough to satisfy ``libdgl.so`` later on.  This is the same
trick ``torch._C._preload_cuda_deps`` uses.

Python imports ``sitecustomize`` automatically at interpreter startup (after
``.pth`` files are processed), so nothing needs to call into this module.

Note: torch's cu113 wheels statically link cuBLAS/cuSPARSE and ship their own
mangled ``libcudart``, so torch is unaffected either way -- this exists purely
for DGL.
"""

import ctypes
import glob
import os
import sys

# libcublas.so.11 depends on libcublasLt.so.11, so ordering matters.
_LIBS = (
    "cuda_runtime/lib/libcudart.so.11.0",
    "cublas/lib/libcublasLt.so.11",
    "cublas/lib/libcublas.so.11",
    "cusparse/lib/libcusparse.so.11",
    "curand/lib/libcurand.so.10",
)


def _nvidia_roots():
    """site-packages/nvidia directories on sys.path, nearest first."""
    for entry in sys.path:
        if not entry:
            continue
        root = os.path.join(entry, "nvidia")
        if os.path.isdir(root):
            yield root


def _preload_cuda11_deps():
    for root in _nvidia_roots():
        loaded = False
        for rel in _LIBS:
            for path in glob.glob(os.path.join(root, rel)):
                try:
                    # RTLD_GLOBAL so dependent libraries can resolve symbols too.
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                    loaded = True
                except OSError:
                    # A missing or unloadable library is not fatal here: the
                    # import that actually needs it will raise a far clearer
                    # error than a crash during interpreter startup would.
                    pass
        if loaded:
            return


try:
    _preload_cuda11_deps()
except Exception:  # never let interpreter startup fail because of this
    pass
