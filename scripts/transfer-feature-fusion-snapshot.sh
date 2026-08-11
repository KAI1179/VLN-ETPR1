#!/usr/bin/env bash
set -euo pipefail

repo_root="${REPO_ROOT:-${HOME}/ETP-R1}"
destination_host="${DESTINATION_HOST:-VIPL-xukai}"
destination="/data/xukai/etp-r1-snapshot"
stage_dir="${STAGE_DIR:-${repo_root}/.transfer-staging/feature-fusion-snapshot}"
archive_name="cognitive-map-caches.full.tar"
archive="${stage_dir}/${archive_name}"
checksums="${stage_dir}/SHA256SUMS"
readme="${repo_root}/docs/data/feature-fusion-snapshot.md"
pinned_commit="c442132aee771395baab443c3037c6d7c4764c4e"
checkpoint_hashes=(
  "127f76283af088495167b4d94080e8373c5c6defcb1ba75cbf972dc1e3db672d"
  "4103199d5113d9000c27e984258c345e8c08a4836b7af59ee5528d914332be21"
  "4e4ab7be1c2ee32fd53a993f7b190ae93fd313faa4b800d95a52952a689e4f3c"
)

required_paths=(
  "data/cognitive_maps/gt.legacy.r1p5.direction5.v1"
  "data/cognitive_maps/gt.legacy.r1p5.direction5.blurred.v1"
  "data/llm_navigation/llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree"
)

verify_remote_checkpoints() {
  local root="$1"
  local listing
  local expected
  local count

  listing="$(
    ssh "${destination_host}" \
      "find '${root}' -type f -name '*.pth' -exec sha256sum {} +"
  )"
  for expected in "${checkpoint_hashes[@]}"; do
    count="$(awk -v hash="${expected}" '$1 == hash { count++ } END { print count + 0 }' <<<"${listing}")"
    if test "${count}" -ne 1; then
      echo "expected exactly one checkpoint with SHA-256 ${expected} under ${destination_host}:${root}; found ${count}" >&2
      exit 1
    fi
  done
}

for command in git rsync sha256sum ssh tar; do
  command -v "${command}" >/dev/null || {
    echo "missing required command: ${command}" >&2
    exit 1
  }
done

git -C "${repo_root}" cat-file -e "${pinned_commit}^{commit}"
test -f "${readme}"
for path in "${required_paths[@]}"; do
  test -d "${repo_root}/${path}" || {
    echo "missing required cache namespace: ${repo_root}/${path}" >&2
    exit 1
  }
done

ssh "${destination_host}" "test -d '${destination}/checkpoints'" || {
  echo "missing checkpoint bundle: ${destination_host}:${destination}/checkpoints" >&2
  exit 1
}
verify_remote_checkpoints "${destination}/checkpoints"

mkdir -p "${stage_dir}"
if test -f "${archive}" && test -f "${checksums}"; then
  (cd "${stage_dir}" && sha256sum --check SHA256SUMS)
elif test -e "${archive}" || test -e "${checksums}"; then
  echo "incomplete staging output exists; inspect it before retrying: ${stage_dir}" >&2
  exit 1
else
  partial="${archive}.partial.$$"
  trap 'rm -f -- "${partial}"' EXIT
  tar -C "${repo_root}" -cf "${partial}" "${required_paths[@]}"
  mv -- "${partial}" "${archive}"
  trap - EXIT

  (
    cd "${stage_dir}"
    sha256sum "${archive_name}" >SHA256SUMS
  )
fi

ssh "${destination_host}" "mkdir -p '${destination}/artifacts'"

rsync -ah --partial --append-verify --info=progress2 \
  "${readme}" "${destination_host}:${destination}/CHECKOUTS.md"
rsync -ah --partial --append-verify --info=progress2 \
  "${archive}" "${checksums}" \
  "${destination_host}:${destination}/artifacts/"

ssh "${destination_host}" \
  "cd '${destination}/artifacts' && sha256sum --check SHA256SUMS"

echo "snapshot verified at ${destination_host}:${destination}"
echo "local staging retained at ${stage_dir}"
