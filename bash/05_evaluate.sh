#!/usr/bin/env bash
# ==============================================================================
# Script: 05_evaluate.sh
# Purpose: Evaluate a trained Thai ASR checkpoint on val/test sets,
#          computing Character Error Rate (CER) and Word Error Rate (WER).
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CHECKPOINT=""
SPLIT="val"
CONFIG_FILE="${PROJECT_ROOT}/configs/experiment/evaluate.yaml"
OUTPUT_DIR=""
EXTRA_ARGS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint|-c)
            CHECKPOINT="$2"
            shift 2
            ;;
        --split|-s)
            SPLIT="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --output-dir|-o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --checkpoint, -c PATH  Path to checkpoint .pt file (auto-detects latest if omitted)"
            echo "  --split, -s SPLIT      Dataset split to evaluate: 'val' or 'test' (default: val)"
            echo "  --config PATH          Path to evaluation YAML (default: configs/experiment/evaluate.yaml)"
            echo "  --output-dir, -o DIR   Directory to save evaluation results"
            echo "  --help, -h             Show this help message"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# If checkpoint is not specified, auto-detect the latest best_checkpoint.pt or checkpoint_*.pt
if [ -z "${CHECKPOINT}" ]; then
    echo "[INFO] No checkpoint specified. Searching for latest trained checkpoint in outputs/runs/..."
    LATEST_CKPT=$(find "${PROJECT_ROOT}/outputs/runs" -type f -name "*.pt" -printf "%T@ %p\n" 2>/dev/null | sort -nr | head -n 1 | awk '{print $2}' || true)
    if [ -n "${LATEST_CKPT}" ] && [ -f "${LATEST_CKPT}" ]; then
        CHECKPOINT="${LATEST_CKPT}"
        echo "[INFO] Auto-detected latest checkpoint: ${CHECKPOINT}"
    else
        echo "[ERROR] No checkpoint found in outputs/runs/." >&2
        echo "[ERROR] Please specify a checkpoint manually with: $0 --checkpoint <path/to/checkpoint.pt>" >&2
        exit 1
    fi
fi

if [ ! -f "${CHECKPOINT}" ]; then
    echo "[ERROR] Checkpoint file does not exist: ${CHECKPOINT}" >&2
    exit 1
fi

CMD_ARGS=(
    "--config" "${CONFIG_FILE}"
    "--checkpoint" "${CHECKPOINT}"
    "--split" "${SPLIT}"
)

if [ -n "${OUTPUT_DIR}" ]; then
    CMD_ARGS+=("--output-dir" "${OUTPUT_DIR}")
fi

echo "======================================================================"
echo " [Step 5] Evaluate Thai ASR Model"
echo " Checkpoint: ${CHECKPOINT}"
echo " Split:      ${SPLIT}"
echo " Config:     ${CONFIG_FILE}"
echo "======================================================================"

uv run python -m thai_audio_jepa_asr.cli.evaluate \
    "${CMD_ARGS[@]}" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

echo "======================================================================"
echo "[SUCCESS] Step 5 finished. Evaluation metrics (CER/WER) generated."
echo "======================================================================"
