"""Process-wide compatibility shims for legacy dependencies."""

import numpy as np

if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]
