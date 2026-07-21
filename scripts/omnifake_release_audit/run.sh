#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/input/workflow_58770161/workspace/test/cameramotion_det}"
STAGE="${STAGE:-preflight}"

# Large, disposable downloaded data stays in /tmp.
SET_ROOT="${SET_ROOT:-/tmp/omnifake/Omni-Fake-SET}"
OOD_ROOT="${OOD_ROOT:-/tmp/omnifake/Omni-Fake-OOD}"

# Audit summaries are small and reusable, so write them to persistent NAS.
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/res/omnifake_release_audit/v1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
AUDIT="${PROJECT_ROOT}/scripts/omnifake_release_audit/audit_release.py"

echo "=== Omni-Fake 发布内容与配对/掩码可用性审计 ==="
echo "stage=${STAGE}"
echo "set_root=${SET_ROOT}"
echo "ood_root=${OOD_ROOT}"
echo "out_dir=${OUT_DIR}"

preflight() {
  test -f "${AUDIT}" || { echo "Missing file: ${AUDIT}" >&2; exit 2; }
  "${PYTHON_BIN}" -c 'import pyarrow; print("pyarrow:", pyarrow.__version__)'
  command -v ffprobe >/dev/null || { echo "ffprobe is required" >&2; exit 2; }
  echo "Preflight passed. No media was read."
}

case "${STAGE}" in
  preflight)
    preflight
    ;;
  hub)
    # Run this on a machine that can access huggingface.co.
    mkdir -p "${OUT_DIR}"
    "${PYTHON_BIN}" "${AUDIT}" hub --out-dir "${OUT_DIR}" --fail-on-audit
    ;;
  local)
    preflight
    test -d "${SET_ROOT}" || { echo "Missing SET_ROOT: ${SET_ROOT}" >&2; exit 2; }
    test -d "${OOD_ROOT}" || { echo "Missing OOD_ROOT: ${OOD_ROOT}" >&2; exit 2; }
    mkdir -p "${OUT_DIR}"
    "${PYTHON_BIN}" "${AUDIT}" local \
      --set-root "${SET_ROOT}" \
      --ood-root "${OOD_ROOT}" \
      --out-dir "${OUT_DIR}" \
      --max-parquet-rows "${MAX_PARQUET_ROWS:-2000}" \
      --decode-samples "${DECODE_SAMPLES:-150}"
    ;;
  *)
    echo "Unknown STAGE=${STAGE}; expected preflight, hub, or local" >&2
    exit 2
    ;;
esac
