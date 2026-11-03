#!/usr/bin/env bash
# ==============================================================================
# Script: run_all.sh
# Purpose: Master end-to-end pipeline runner for Thai Audio-JEPA ASR.
#          Executes Steps 1 to 5 sequentially:
#          1. Download dataset (skips if already present)
#          2. Prepare data & manifests
#          3. Thai SSL Continued Pretraining
#          4. Supervised CTC ASR Training
#          5. Evaluate best checkpoint (CER/WER)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

SMOKE=false
SKIP_DOWNLOAD=false
SKIP_PREP=false
SKIP_SSL=false
SKIP_TRAIN=false
SKIP_EVAL=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke)
            SMOKE=true
            shift
            ;;
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --skip-prep)
            SKIP_PREP=true
            shift
            ;;
        --skip-ssl)
            SKIP_SSL=true
            shift
            ;;
        --skip-train)
            SKIP_TRAIN=true
            shift
            ;;
        --skip-eval)
            SKIP_EVAL=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --smoke           Run pipeline in fast smoke-test mode (1 epoch, subset samples)"
            echo "  --skip-download   Skip Step 1 (download)"
            echo "  --skip-prep       Skip Step 2 (data preparation)"
            echo "  --skip-ssl        Skip Step 3 (Thai SSL pretraining)"
            echo "  --skip-train      Skip Step 4 (ASR fine-tuning)"
            echo "  --skip-eval       Skip Step 5 (Evaluation)"
            echo "  --help, -h        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

START_TIME=$(date +%s)
echo "======================================================================"
echo " 🚀 THAI AUDIO-JEPA ASR: END-TO-END PIPELINE"
echo " Mode: $( [ "${SMOKE}" = true ] && echo "SMOKE TEST (Fast Verification)" || echo "FULL RUN" )"
echo " Workspace: ${PROJECT_ROOT}"
echo "======================================================================"

# Step 1: Download data
if [ "${SKIP_DOWNLOAD}" = false ]; then
    echo ""
    echo "[PIPELINE] >>> Running Step 1: Download Data..."
    bash "${SCRIPT_DIR}/01_download_data.sh"
else
    echo "[PIPELINE] >>> Skipping Step 1 (Download Data)"
fi

# Step 2: Prepare data
if [ "${SKIP_PREP}" = false ]; then
    echo ""
    echo "[PIPELINE] >>> Running Step 2: Prepare Data..."
    if [ "${SMOKE}" = true ]; then
        bash "${SCRIPT_DIR}/02_prepare_data.sh" --smoke
    else
        bash "${SCRIPT_DIR}/02_prepare_data.sh" --max-samples 0
    fi
else
    echo "[PIPELINE] >>> Skipping Step 2 (Prepare Data)"
fi

# Step 3: Thai SSL Pretraining
if [ "${SKIP_SSL}" = false ]; then
    echo ""
    echo "[PIPELINE] >>> Running Step 3: Thai SSL Pretraining..."
    if [ "${SMOKE}" = true ]; then
        bash "${SCRIPT_DIR}/03_ssl_pretrain.sh" --smoke
    else
        bash "${SCRIPT_DIR}/03_ssl_pretrain.sh"
    fi
else
    echo "[PIPELINE] >>> Skipping Step 3 (Thai SSL Pretraining)"
fi

# Step 4: Supervised CTC ASR Fine-Tuning
if [ "${SKIP_TRAIN}" = false ]; then
    echo ""
    echo "[PIPELINE] >>> Running Step 4: Supervised ASR Training..."
    if [ "${SMOKE}" = true ]; then
        bash "${SCRIPT_DIR}/04_train_asr.sh" --smoke
    else
        bash "${SCRIPT_DIR}/04_train_asr.sh"
    fi
else
    echo "[PIPELINE] >>> Skipping Step 4 (Supervised ASR Training)"
fi

# Step 5: Evaluate Checkpoint
if [ "${SKIP_EVAL}" = false ]; then
    echo ""
    echo "[PIPELINE] >>> Running Step 5: Evaluation..."
    bash "${SCRIPT_DIR}/05_evaluate.sh" --split val
else
    echo "[PIPELINE] >>> Skipping Step 5 (Evaluation)"
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "======================================================================"
echo " 🎉 PIPELINE COMPLETED SUCCESSFULLY! (Elapsed: ${ELAPSED}s)"
echo "======================================================================"
