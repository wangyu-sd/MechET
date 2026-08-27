"""Make the vendored RetroBridge source importable from the MechET monorepo."""

import sys
from pathlib import Path


RETROBRIDGE_ROOT = Path(__file__).resolve().parents[1]
if str(RETROBRIDGE_ROOT) not in sys.path:
    sys.path.insert(0, str(RETROBRIDGE_ROOT))
