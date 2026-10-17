#!/usr/bin/env bash
# ==============================================================================
# Script: 02_prepare_data.sh
# Purpose: Preprocess audio (fbank, WAVs), normalize Thai text, build
#          train/val/test manifests (JSONL), and create Thai character vocab.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG_FILE="${PROJECT_ROOT}/configs/data/porjai.yaml"
MAX_SAMPLES=0 # Default to 0 (process all available samples)
OUTPUT_DIR="${PROJECT_ROOT}/data_cache/porjai"
FORCE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --force|-f)
            FORCE=true
            shift
            ;;
        --smoke)
            MAX_SAMPLES=500
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config PATH       Path to data configuration YAML (default: configs/data/porjai.yaml)"
            echo "  --max-samples N     Number of samples to process (0 = full dataset, default: 0)"
            echo "  --smoke             Quick smoke-test preparation (sets --max-samples 500)"
            echo "  --output-dir DIR    Destination directory (default: data_cache/porjai)"
            echo "  --force, -f         Force re-preparation even if manifests already exist"
            echo "  --help, -h          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

echo "======================================================================"
echo " [Step 2] Prepare Porjai Thai Dataset Manifests"
echo " Config:      ${CONFIG_FILE}"
echo " Max Samples: ${MAX_SAMPLES} (0 = all)"
echo " Output Dir:  ${OUTPUT_DIR}"
echo "======================================================================"

MANIFEST_TRAIN="${OUTPUT_DIR}/manifests/train.jsonl"
VOCAB_FILE="${OUTPUT_DIR}/vocab.json"

if [ -f "${MANIFEST_TRAIN}" ] && [ -f "${VOCAB_FILE}" ] && [ "${FORCE}" = false ]; then
    echo "[INFO] Existing manifests and vocabulary found at:"
    echo "       - ${MANIFEST_TRAIN}"
    echo "       - ${VOCAB_FILE}"
    echo "[INFO] If you want to re-run and overwrite, use --force flag:"
    echo "       bash bash/02_prepare_data.sh --force"
    echo "----------------------------------------------------------------------"
    echo "[INFO] Skipping data preparation (already prepared)."
    echo "======================================================================"
    exit 0
fi

echo "[INFO] Running data preparation pipeline..."
uv run python -m thai_audio_jepa_asr.cli.prepare_data \
    --config "${CONFIG_FILE}" \
    --max-samples "${MAX_SAMPLES}" \
    --output-dir "${OUTPUT_DIR}"

echo "======================================================================"
echo "[SUCCESS] Step 2 finished. Manifests & vocab ready in: ${OUTPUT_DIR}"
echo "======================================================================"
