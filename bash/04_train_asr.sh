#!/usr/bin/env bash
# ==============================================================================
# Script: 04_train_asr.sh
# Purpose: Supervised CTC ASR Fine-Tuning with Audio-JEPA Encoder on Thai speech.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG_FILE="${PROJECT_ROOT}/configs/experiment/audio_jepa_ctc.yaml"
EXTRA_ARGS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --smoke)
            CONFIG_FILE="${PROJECT_ROOT}/configs/experiment/smoke.yaml"
            shift
            ;;
        --low-resource|--budget)
            BUDGET="$2"
            case "${BUDGET}" in
                10|25|50|100)
                    CONFIG_FILE="${PROJECT_ROOT}/configs/experiment/low_resource_${BUDGET}.yaml"
                    ;;
                *)
                    echo "[ERROR] Invalid low-resource budget: ${BUDGET}. Must be 10, 25, 50, or 100." >&2
                    exit 1
                    ;;
            esac
            shift 2
            ;;
        --resume)
            EXTRA_ARGS+=("--resume" "$2")
            shift 2
            ;;
        --epochs)
            EXTRA_ARGS+=("--epochs" "$2")
            shift 2
            ;;
        --batch-size)
            EXTRA_ARGS+=("--batch-size" "$2")
            shift 2
            ;;
        --seed)
            EXTRA_ARGS+=("--seed" "$2")
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS] [-- ADDITIONAL_CLI_ARGS]"
            echo ""
            echo "Options:"
            echo "  --config PATH            Path to experiment YAML (default: configs/experiment/audio_jepa_ctc.yaml)"
            echo "  --smoke                  Fast 1-epoch smoke test (configs/experiment/smoke.yaml)"
            echo "  --low-resource BUDGET    Use low-resource config (10, 25, 50, or 100)"
            echo "  --resume PATH            Resume training from checkpoint"
            echo "  --epochs N               Override number of epochs"
            echo "  --batch-size N           Override batch size"
            echo "  --seed N                 Override random seed"
            echo "  --help, -h               Show this help message"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

echo "======================================================================"
echo " [Step 4] Supervised Audio-JEPA + CTC Thai ASR Training"
echo " Config: ${CONFIG_FILE}"
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    echo " Extra:  ${EXTRA_ARGS[*]}"
fi
echo "======================================================================"

# Verify train manifest exists
TRAIN_MANIFEST="${PROJECT_ROOT}/data_cache/porjai/manifests/train.jsonl"
if [ ! -f "${TRAIN_MANIFEST}" ] && [ "${CONFIG_FILE}" != *"smoke"* ]; then
    echo "[ERROR] Manifest not found at: ${TRAIN_MANIFEST}" >&2
    echo "[ERROR] Please run 'bash bash/02_prepare_data.sh' first!" >&2
    exit 1
fi

uv run python -m thai_audio_jepa_asr.cli.train_asr \
    --config "${CONFIG_FILE}" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

echo "======================================================================"
echo "[SUCCESS] Step 4 finished. ASR model training completed."
echo "======================================================================"
