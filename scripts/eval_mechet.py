#!/usr/bin/env python3
"""Legacy entrypoint: gold-data audit only (not model eval).

Prefer ``scripts/audit_mechet_gold.py``. See ``docs/EVAL.md``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    cmd = [sys.executable, str(REPO / "scripts/audit_mechet_gold.py"), *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd))
