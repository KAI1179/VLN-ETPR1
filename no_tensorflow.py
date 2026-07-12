"""Disable TensorFlow import paths used by optional logging dependencies."""

from __future__ import annotations

import os
import sys
import types


def configure_no_tensorflow() -> None:
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    sys.modules.setdefault(
        "tensorboard.compat.notf", types.ModuleType("tensorboard.compat.notf")
    )
