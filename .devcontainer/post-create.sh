#!/usr/bin/env bash
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate etpr1-new

# ===== Habitat Fix =====

HABITAT_LAB_DIR="/workspaces/ETP-R1/data/habitat-lab-0.1.7"
RL_REQUIREMENTS="${HABITAT_LAB_DIR}/habitat_baselines/rl/requirements.txt"

if [[ ! -d "${HABITAT_LAB_DIR}" ]]; then
    echo "Habitat-Lab not found at ${HABITAT_LAB_DIR}; skipping editable install."
    echo "Download habitat-lab 0.1.7 under data/ and rerun .devcontainer/post-create.sh."
    exit 0
fi

# Patch habitat-lab calls to np.float
find data/habitat-lab-0.1.7 -type f -name '*.py' -exec sed -Ei 's/\bnp\.float\b/float/g' {} +

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

# Verification
python -m pip check
python - <<'PY'
import habitat
import habitat_baselines
from habitat_baselines.common.baseline_registry import baseline_registry

print("habitat", getattr(habitat, "__version__", "unknown"))
print("habitat_baselines import ok")
print("baseline_registry", type(baseline_registry).__name__)
PY

# Install habitat-lab
cd "${HABITAT_LAB_DIR}"
python setup.py develop --all --no-deps

# ===== Tensorflow Fix =====
# See https://github.com/tensorflow/tensorflow/issues/57679#issuecomment-1249197802

# TensorRT Verification
python3 -c "import tensorrt; print(tensorrt.__version__); assert tensorrt.Builder(tensorrt.Logger())"

# Patch LD_LIBRARY_PATH
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/python3.8/site-packages/tensorrt/' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# Tensorflow Verification
conda activate etpr1-new
python3 -c "import tensorflow as tf; assert tf.config.list_physical_devices('GPU'), 'Failed to setup Tensorflow properly, or maybe you do not have GPU'"
