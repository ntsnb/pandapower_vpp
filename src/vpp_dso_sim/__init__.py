"""DSO and multi-VPP simulation toolkit built on pandapower."""

from __future__ import annotations

import os
from pathlib import Path

if os.environ.get("PANDAPOWER_GRID_BUILDER_ENABLE_NUMBA_JIT") != "1":
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

_BOOT_CACHE_DIR = Path(os.environ.get("VPP_DSO_BOOT_CACHE_DIR", "/tmp/pandapower_vpp_cache"))
(_BOOT_CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_BOOT_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_BOOT_CACHE_DIR))

__version__ = "0.1.0"
