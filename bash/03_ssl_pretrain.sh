#!/usr/bin/env bash
# ==============================================================================
# Script: 03_ssl_pretrain.sh
# Purpose: Continued Self-Supervised Adaptation of Audio-JEPA on Thai speech
#          using Contiguous Temporal Masking and Normalized MSE latent prediction.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG_FILE="${PROJECT_ROOT}/configs/experiment/audio_jepa_thai_ssl.yaml"
EXTRA_ARGS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --smoke)
            CONFIG_FILE="${PROJECT_ROOT}/configs/experiment/smoke_ssl.yaml"
            shift
            ;;
        --resume)
            EXTRA_ARGS+=("--resume" "$2")
            shift 2
            ;;
        --epochs)
            EXTRA_ARGS+=("--epochs" "$2")
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS] [-- ADDITIONAL_CLI_ARGS]"
            echo ""
            echo "Options:"
            echo "  --config PATH   Path to SSL experiment YAML (default: configs/experiment/audio_jepa_thai_ssl.yaml)"
            echo "  --smoke         Fast 1-epoch smoke test (configs/experiment/smoke_ssl.yaml)"
            echo "  --resume PATH   Path to checkpoint to resume training from"
            echo "  --epochs N      Override number of training epochs"
            echo "  --help, -h      Show this help message"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

echo "======================================================================"
echo " [Step 3] Thai Audio-JEPA SSL Continued Pretraining"
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

uv run python -m thai_audio_jepa_asr.cli.adapt_ssl \
    --config "${CONFIG_FILE}" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

echo "======================================================================"
echo "[SUCCESS] Step 3 finished. Adapted encoder saved to output directory."
echo "======================================================================"
