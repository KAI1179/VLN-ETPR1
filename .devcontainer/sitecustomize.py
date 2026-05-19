"""Process-wide compatibility shims for legacy dependencies."""

import os

import numpy as np

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("MAGNUM_LOG", "quiet")

if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]
