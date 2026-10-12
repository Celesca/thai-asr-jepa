#!/usr/bin/env bash
# ==============================================================================
# Script: 01_download_data.sh
# Purpose: Download Porjai Central Thai voice dataset from Hugging Face Hub.
#          If the dataset is already present or cached, skips downloading.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

TARGET_DIR="${PROJECT_ROOT}/data_cache/raw_porjai"
REPO_ID="CMKL/Porjai-Thai-voice-dataset-central"
FORCE_DOWNLOAD=false

# Parse command line options
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force|-f)
            FORCE_DOWNLOAD=true
            shift
            ;;
        --target-dir)
            TARGET_DIR="$2"
            shift 2
            ;;
        --repo-id)
            REPO_ID="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --force, -f        Force download even if data already exists"
            echo "  --target-dir DIR   Target directory for downloaded data (default: data_cache/raw_porjai)"
            echo "  --repo-id ID       Hugging Face repository ID (default: CMKL/Porjai-Thai-voice-dataset-central)"
            echo "  --help, -h         Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

echo "======================================================================"
echo " [Step 1] Download Porjai Thai Dataset"
echo " Repo ID:    ${REPO_ID}"
echo " Target Dir: ${TARGET_DIR}"
echo "======================================================================"

# Check if data already exists
DATA_EXISTS=false
if [ -d "${TARGET_DIR}" ]; then
    PARQUET_COUNT=$(find "${TARGET_DIR}" -type f -name "*.parquet" 2>/dev/null | wc -l || true)
    if [ "${PARQUET_COUNT}" -gt 0 ]; then
        DATA_EXISTS=true
        echo "[INFO] Found ${PARQUET_COUNT} existing parquet file(s) in: ${TARGET_DIR}"
    fi
fi

# Also check if processed manifests already exist
MANIFEST_FILE="${PROJECT_ROOT}/data_cache/porjai/manifests/train.jsonl"
if [ -f "${MANIFEST_FILE}" ] && [ "${DATA_EXISTS}" = false ]; then
    echo "[INFO] Processed dataset manifests already exist at: ${MANIFEST_FILE}"
    DATA_EXISTS=true
fi

if [ "${DATA_EXISTS}" = true ] && [ "${FORCE_DOWNLOAD}" = false ]; then
    echo "----------------------------------------------------------------------"
    echo "[INFO] Data already exists. Skipping download as requested."
    echo "[INFO] If you want to force re-download, run with --force flag:"
    echo "       bash bash/01_download_data.sh --force"
    echo "======================================================================"
    exit 0
fi

if [ "${FORCE_DOWNLOAD}" = true ]; then
    echo "[INFO] --force specified. Proceeding with download..."
else
    echo "[INFO] Data not found locally. Proceeding to download from Hugging Face Hub..."
fi

mkdir -p "${TARGET_DIR}"

# Download using huggingface_hub snapshot_download
uv run python - <<EOF
import os
import sys
from huggingface_hub import snapshot_download

repo_id = "${REPO_ID}"
target_dir = "${TARGET_DIR}"

print(f"--> Initiating snapshot download for '{repo_id}' into '{target_dir}'...")
try:
    snapshot_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=target_dir,
        resume_download=True,
    )
    print(f"--> Download successfully completed! Files stored at: {snapshot_path}")
except Exception as e:
    print(f"[ERROR] Failed to download dataset: {e}", file=sys.stderr)
    sys.exit(1)
EOF

echo "======================================================================"
echo "[SUCCESS] Step 1 finished. Data is ready in: ${TARGET_DIR}"
echo "======================================================================"
