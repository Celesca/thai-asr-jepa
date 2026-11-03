#!/bin/bash
#SBATCH --job-name=thai_jepa
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpul40
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
# ==============================================================================
# SLURM version of bash/run_all.sh (same steps 1-5, same flags). Submit from the repo root:
#   mkdir -p logs
#   sbatch bash/run_all.sbatch --smoke                         # quick check first
#   sbatch bash/run_all.sbatch                                 # full run
#   sbatch bash/run_all.sbatch --skip-download --skip-prep     # data already prepared
# 12h limit: on TIMEOUT resubmit with --skip-* for finished steps and --resume-ssl / --resume-ctc, e.g.
#   sbatch bash/run_all.sbatch --skip-download --skip-prep --skip-ssl \
#       --resume-ctc outputs/runs/exp_C_thai_ssl_ctc/checkpoints/checkpoint_latest.pt
# ==============================================================================
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"     # sbatch runs a copy of this file, so locate the repo by submit dir
[ -f bash/run_all.sh ] || { echo "Submit from the repo root: sbatch bash/run_all.sbatch" >&2; exit 1; }
SCRIPT_DIR="$(pwd)/bash"

# Experiment C: Thai SSL -> CTC. (run_all.sh's default CTC config ignores the SSL output.)
SSL_CFG=configs/experiment/exp_C_ssl_pretrain.yaml
CTC_CFG=configs/experiment/exp_C_thai_ssl_ctc.yaml
CTC_RUN=outputs/runs/exp_C_thai_ssl_ctc

SMOKE=false
SKIP_DOWNLOAD=false
SKIP_PREP=false
SKIP_SSL=false
SKIP_TRAIN=false
SKIP_EVAL=false
RESUME_SSL=()
RESUME_CTC=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke)         SMOKE=true; shift ;;
        --skip-download) SKIP_DOWNLOAD=true; shift ;;
        --skip-prep)     SKIP_PREP=true; shift ;;
        --skip-ssl)      SKIP_SSL=true; shift ;;
        --skip-train)    SKIP_TRAIN=true; shift ;;
        --skip-eval)     SKIP_EVAL=true; shift ;;
        --resume-ssl)    RESUME_SSL=(--resume "$2"); shift 2 ;;
        --resume-ctc)    RESUME_CTC=(--resume "$2"); shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

echo "=== Host ==="
echo "Job executed on: $(hostname)"
echo "Job started on: $(date)"
echo "Mode: $( [ "${SMOKE}" = true ] && echo "SMOKE TEST" || echo "FULL RUN" )"

export PATH="$HOME/.local/bin:$PATH"            # uv
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"      # Audio-JEPA checkpoint cache

echo "=== Python ==="
echo "uv location: $(which uv)"
echo "Python version: $(uv run python --version)"

echo "=== GPU ==="
nvidia-smi || echo "WARNING: nvidia-smi failed on $(hostname)"
uv run python -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available on this node'; print('CUDA OK:', torch.cuda.get_device_name(0))"

echo "=== Execution ==="
START_TIME=$(date +%s)

# Step 1: Download data (skips if already present)
if [ "${SKIP_DOWNLOAD}" = false ]; then
    echo "[PIPELINE] >>> Step 1: Download Data"
    bash "${SCRIPT_DIR}/01_download_data.sh"
else
    echo "[PIPELINE] >>> Skipping Step 1 (Download Data)"
fi

# Step 2: Prepare data
if [ "${SKIP_PREP}" = false ]; then
    echo "[PIPELINE] >>> Step 2: Prepare Data"
    if [ "${SMOKE}" = true ]; then
        bash "${SCRIPT_DIR}/02_prepare_data.sh" --smoke
    else
        # --force: otherwise a previous 500-sample smoke prep is silently reused as the "full" data
        bash "${SCRIPT_DIR}/02_prepare_data.sh" --max-samples 0 --force
    fi
else
    echo "[PIPELINE] >>> Skipping Step 2 (Prepare Data)"
fi

if [ "${SMOKE}" = true ]; then
    # smoke.yaml / smoke_ssl.yaml read these, but no script creates them
    M=data_cache/porjai/manifests
    head -200 "$M/train.jsonl" > "$M/smoke_train.jsonl"
    head -50  "$M/val.jsonl"   > "$M/smoke_val.jsonl"
fi

# Step 3: Thai SSL Pretraining
if [ "${SKIP_SSL}" = false ]; then
    echo "[PIPELINE] >>> Step 3: Thai SSL Pretraining"
    if [ "${SMOKE}" = true ]; then
        bash "${SCRIPT_DIR}/03_ssl_pretrain.sh" --smoke
    else
        bash "${SCRIPT_DIR}/03_ssl_pretrain.sh" --config "${SSL_CFG}" ${RESUME_SSL[@]+"${RESUME_SSL[@]}"}
    fi
else
    echo "[PIPELINE] >>> Skipping Step 3 (Thai SSL Pretraining)"
fi

# Step 4: Supervised CTC ASR Fine-Tuning
if [ "${SKIP_TRAIN}" = false ]; then
    echo "[PIPELINE] >>> Step 4: Supervised ASR Training"
    if [ "${SMOKE}" = true ]; then
        bash "${SCRIPT_DIR}/04_train_asr.sh" --smoke
    else
        # train_asr silently falls back to the original weights if the SSL encoder is missing
        SSL_CKPT=outputs/runs/exp_C_ssl_pretrain/checkpoints/adapted_encoder.pt
        [ -f "${SSL_CKPT}" ] || { echo "[ERROR] Missing ${SSL_CKPT}. Run Step 3 first." >&2; exit 1; }
        bash "${SCRIPT_DIR}/04_train_asr.sh" --config "${CTC_CFG}" ${RESUME_CTC[@]+"${RESUME_CTC[@]}"}
    fi
else
    echo "[PIPELINE] >>> Skipping Step 4 (Supervised ASR Training)"
fi

# Step 5: Evaluate best checkpoint (evaluate.yaml matches exp C / smoke: patch 1x16, native_pooling)
if [ "${SKIP_EVAL}" = false ]; then
    echo "[PIPELINE] >>> Step 5: Evaluation"
    RUN=$( [ "${SMOKE}" = true ] && echo outputs/runs/smoke_ctc || echo "${CTC_RUN}" )
    bash "${SCRIPT_DIR}/05_evaluate.sh" --checkpoint "${RUN}/checkpoints/best_checkpoint.pt" --output-dir "${RUN}/eval" --split val
else
    echo "[PIPELINE] >>> Skipping Step 5 (Evaluation)"
fi

echo "Job finished on: $(date) (Elapsed: $(( $(date +%s) - START_TIME ))s)"
