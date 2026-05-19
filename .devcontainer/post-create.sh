#!/usr/bin/env bash
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate etpr1-uv

# ===== Habitat Fix =====

HABITAT_LAB_DIR="/workspaces/ETP-R1/data/habitat-lab-0.1.7"
RL_REQUIREMENTS="${HABITAT_LAB_DIR}/habitat_baselines/rl/requirements.txt"

if [[ ! -d "${HABITAT_LAB_DIR}" ]]; then
    echo "Habitat-Lab not found at ${HABITAT_LAB_DIR}; skipping editable install."
    echo "Download habitat-lab 0.1.7 under data/ and rerun .devcontainer/post-create.sh."
    exit 0
fi

# Restore removed NumPy aliases for legacy Habitat-Lab/Habitat-Sim code.
cp /workspaces/ETP-R1/.devcontainer/sitecustomize.py "$CONDA_PREFIX/lib/python3.8/site-packages/sitecustomize.py"

# Patch habitat-lab requirements
if [[ -f "${RL_REQUIREMENTS}" ]]; then
    python - <<'PY'
from pathlib import Path

path = Path("/workspaces/ETP-R1/data/habitat-lab-0.1.7/habitat_baselines/rl/requirements.txt")
text = path.read_text()
old = "tensorflow==1.13.1"
new = 'tensorflow==1.13.1; python_version < "3.8"'
if old in text and new not in text:
    text = text.replace(
        "# full tensorflow required for tensorboard video support\n"
        "tensorflow==1.13.1",
        "# TensorFlow 1.13.1 has no Python 3.8 wheel\n"
        f"{new}",
    )
    path.write_text(text)
PY
fi

# Install habitat-lab
cd "${HABITAT_LAB_DIR}"
python setup.py develop --all --no-deps

# Verification
python -m pip check
python - <<'PY'
import habitat
import habitat_baselines
import numpy as np
from habitat_baselines.common.baseline_registry import baseline_registry

print("habitat", getattr(habitat, "__version__", "unknown"))
print("habitat_baselines import ok")
print("baseline_registry", type(baseline_registry).__name__)
assert np.float is float
PY

# ===== Tensorflow Fix =====
# See https://github.com/tensorflow/tensorflow/issues/57679#issuecomment-1249197802

# TensorRT Verification
python3 -c "import tensorrt; print(tensorrt.__version__); assert tensorrt.Builder(tensorrt.Logger())"

# Refresh activation hooks now that the TensorRT package is installed.
bash /workspaces/ETP-R1/scripts/install-conda-activation-hooks.sh

# Tensorflow Verification
conda activate etpr1-uv
python3 -c "import tensorflow as tf; assert tf.config.list_physical_devices('GPU'), 'Failed to setup Tensorflow properly, or maybe you do not have GPU'"
