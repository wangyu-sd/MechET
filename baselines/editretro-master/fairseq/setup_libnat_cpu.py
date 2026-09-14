#!/usr/bin/env python3
"""Build only EditRetro's Fairseq CPU Levenshtein extension.

The legacy all-in-one setup requires Cython even when only ``libnat`` is
needed for a CPU smoke test.  This compatibility helper leaves the official
C++ implementation unchanged and writes the extension into ``fairseq/``.
"""
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


ROOT = Path(__file__).resolve().parent

setup(
    name="editretro-fairseq-libnat-cpu",
    version="0.0.0",
    ext_modules=[
        CppExtension(
            "fairseq.libnat",
            sources=[str(ROOT / "fairseq/clib/libnat/edit_dist.cpp")],
            extra_compile_args=["-O2"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
