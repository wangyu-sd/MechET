"""Append staged non-CUDA dependencies, preserving the base CUDA11 packages.

Matches the established EARHO CUDA11 runtime layering. The dependency archive
excludes torch, functorch, nvidia, triton, vllm, xformers, cupy and cuda packages.
"""
import os
import sys

fallback = os.environ.get('MECHET_VLLM_PUREPY_FALLBACK')
if fallback and fallback not in sys.path:
    sys.path.append(fallback)
