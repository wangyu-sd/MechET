"""Expose staged pure-Python vLLM dependencies after the CUDA 11 base env."""

import os
import sys

fallback = os.environ.get("MECHET_VLLM_PUREPY_FALLBACK")
if fallback and fallback not in sys.path:
    sys.path.append(fallback)
