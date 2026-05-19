#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "CONDA_PREFIX is not set. Activate the etpr1-uv conda environment first." >&2
    exit 1
fi

activate_dir="${CONDA_PREFIX}/etc/conda/activate.d"
deactivate_dir="${CONDA_PREFIX}/etc/conda/deactivate.d"
mkdir -p "${activate_dir}" "${deactivate_dir}"

cat > "${activate_dir}/etpr1-env-vars.sh" <<'EOF'
#!/usr/bin/env bash

export ETPR1_BACKUP_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
export ETPR1_BACKUP_GLOG_MINLOGLEVEL="${GLOG_minloglevel-}"
export ETPR1_BACKUP_MAGNUM_LOG="${MAGNUM_LOG-}"

export GLOG_minloglevel="${GLOG_minloglevel:-2}"
export MAGNUM_LOG="${MAGNUM_LOG:-quiet}"

case ":${LD_LIBRARY_PATH:-}:" in
    *":${CONDA_PREFIX}/lib:"*) ;;
    *) export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
esac

tensorrt_lib="${CONDA_PREFIX}/lib/python3.8/site-packages/tensorrt"
if [[ -d "${tensorrt_lib}" ]]; then
    case ":${LD_LIBRARY_PATH:-}:" in
        *":${tensorrt_lib}:"*) ;;
        *) export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}${tensorrt_lib}" ;;
    esac
fi
EOF

cat > "${deactivate_dir}/etpr1-env-vars.sh" <<'EOF'
#!/usr/bin/env bash

if [[ -n "${ETPR1_BACKUP_LD_LIBRARY_PATH+x}" ]]; then
    export LD_LIBRARY_PATH="${ETPR1_BACKUP_LD_LIBRARY_PATH}"
    unset ETPR1_BACKUP_LD_LIBRARY_PATH
fi

if [[ -n "${ETPR1_BACKUP_GLOG_MINLOGLEVEL+x}" ]]; then
    export GLOG_minloglevel="${ETPR1_BACKUP_GLOG_MINLOGLEVEL}"
    unset ETPR1_BACKUP_GLOG_MINLOGLEVEL
fi

if [[ -n "${ETPR1_BACKUP_MAGNUM_LOG+x}" ]]; then
    export MAGNUM_LOG="${ETPR1_BACKUP_MAGNUM_LOG}"
    unset ETPR1_BACKUP_MAGNUM_LOG
fi
EOF

chmod +x "${activate_dir}/etpr1-env-vars.sh" "${deactivate_dir}/etpr1-env-vars.sh"
