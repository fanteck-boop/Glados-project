# ------------------------------------------------------------------
# Config with relative-import fallback for standalone mode
# ------------------------------------------------------------------
from __future__ import annotations

# Relative imports (package mode) or absolute (standalone mode)
try:
    from . import config as cfg
except ImportError:
    import config as cfg  # type: ignore[no-redef]