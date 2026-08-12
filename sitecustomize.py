"""Expose the CUDA 11 runtime libraries from the `nvidia-*-cu11` wheels to ld.so.

The `ct` conda environment got `libcudart.so.11.0`, `libcublas.so.11` and
`libcusparse.so.11` from the `cudatoolkit=11.3.1` conda package. Under uv those
libraries come from the `nvidia-cuda-runtime-cu11`, `nvidia-cublas-cu11` and
`nvidia-cusparse-cu11` wheels, which unpack into `site-packages/nvidia/*/lib` --
a directory the dynamic loader does not search. `dgl_cu111` links against them by
SONAME and loads its C extension with a bare `ctypes.CDLL("libdgl.so")`, so
`import dgl` would otherwise die with
"libcublas.so.11: cannot open shared object file".

Loading them here with RTLD_GLOBAL registers their SONAMEs with the loader, which
then reuses them for anything dlopen'd later in the process (the same trick
torch >= 1.13 uses in `torch/__init__.py::_preload_cuda_deps`). CPython imports
`sitecustomize` automatically during startup, so this applies to `uv run ...`,
`.venv/bin/python ...` and any subprocess spawned from them -- no
LD_LIBRARY_PATH needed.

`torch` itself does not need this: the cu113 wheels statically link cuBLAS/cuDNN.
"""

import ctypes
import glob
import os
import sysconfig

# Ordered: dependencies before dependents (libcublas needs libcublasLt).
_SONAMES = (
    "libcudart.so.11.0",
    "libcublasLt.so.11",
    "libcublas.so.11",
    "libcusparse.so.11",
)


def _preload_cuda11_deps():
    root = os.path.join(sysconfig.get_paths()["purelib"], "nvidia")
    if not os.path.isdir(root):
        return
    for soname in _SONAMES:
        for path in sorted(glob.glob(os.path.join(root, "*", "lib", soname))):
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                continue
            break


try:
    _preload_cuda11_deps()
except Exception:  # never let this break interpreter startup
    pass
