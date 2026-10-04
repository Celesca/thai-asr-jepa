# Thai Audio-JEPA ASR Research Project

[![Python 3.12](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

A reproducible research codebase investigating downstream Thai Automatic Speech Recognition (ASR) transfer and Thai-specific self-supervised continued pretraining of **Audio-JEPA** ([ICME 2025](https://arxiv.org/abs/2507.02915), `ltuncay/Audio-JEPA`) using the **Porjai Central Thai** speech corpus (`CMKL/Porjai-Thai-voice-dataset-central`).

---

## 📌 Project Objective

**Research Question:**
> *“Can Thai-specific self-supervised adaptation of Audio-JEPA improve Thai ASR under limited labeled-data conditions?”*

This project implements:
1. **Audio-JEPA ViT Speech Encoder Adapter** with native PyTorch 2.x SDPA attention and checkpoint loading from Hugging Face Hub (`ltuncay/Audio-JEPA`).
2. **Temporal Resolution Adaptation** resolving the native 1.6 fps ($16 \times 8$ patch) bottleneck for CTC alignment on Thai characters ($T_{\text{out}} \ge L_{\text{target}}$).
3. **Deterministic Leakage-Free Dataset Preparation** of the Porjai Central Thai speech corpus with Unicode NFKC normalization and character-level CTC tokenization.
4. **Thai-Specific Continued Self-Supervised Adaptation** using Audio-JEPA's masked latent prediction objective on unlabeled Thai audio.
5. **Low-Resource Comparative Benchmarking** across labeled budgets (10%, 25%, 50%, 100%) and baselines.

---

## ⚙️ Installation (via `uv`)

This project strictly uses [`uv`](https://github.com/astral-sh/uv) for reproducible dependency and environment management.

```bash
# 1. Install uv if not already present
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 2. Sync dependencies and create locked virtual environment
uv sync
```

### Hardware Requirements
- **GPU:** NVIDIA GPU with $\ge 8$ GB VRAM recommended (RTX 3080/4080/5080, A100, L4).
- **CPU:** Fully supported (all models automatically run on CPU if CUDA is unavailable).
- **Disk:** ~1 GB for smoke test / pilot subset; ~15 GB for full dataset and model weights.

---

## 🚀 Exact Execution Commands

All commands run through `uv run`:

### 1. Inspect Raw Dataset & Audio-JEPA Checkpoint
```bash
# Inspect Porjai dataset schema and sample audio
uv run python -m thai_audio_jepa_asr.cli.inspect_data

# Inspect Audio-JEPA ViT encoder specifications and checkpoint keys
uv run python -m thai_audio_jepa_asr.cli.inspect_audio_jepa
```

### 2. Prepare Porjai Thai Dataset
Streams and preprocesses audio, generating deterministic 80/10/10 train/validation/test manifests and vocabulary:
```bash
# Process verification subset (500 samples, fast)
uv run python -m thai_audio_jepa_asr.cli.prepare_data --config configs/data/porjai.yaml

# For full 700-hour dataset preparation:
uv run python -m thai_audio_jepa_asr.cli.prepare_data --config configs/data/porjai.yaml --max-samples 0
```

### 3. Run Tiny Smoke Test (CPU or GPU)
Runs an immediate end-to-end check of normalizer, tokenizer, mel-extractor, ViT forward/backward pass, CTC loss, and prediction:
```bash
uv run python -m thai_audio_jepa_asr.cli.smoke_test --config configs/experiment/smoke.yaml
```

### 4. Train Supervised CTC ASR
Trains the Audio-JEPA encoder + CTC head on Thai speech:
```bash
uv run python -m thai_audio_jepa_asr.cli.train_asr --config configs/experiment/audio_jepa_ctc.yaml
```

### 5. Continued Thai Self-Supervised Adaptation (SSL)
Performs masked latent prediction on unlabeled Thai audio to adapt Audio-JEPA before ASR fine-tuning:
```bash
uv run python -m thai_audio_jepa_asr.cli.adapt_ssl --config configs/experiment/audio_jepa_thai_ssl.yaml
```
*The adapted checkpoint is automatically saved to `outputs/runs/audio_jepa_thai_ssl/checkpoints/adapted_encoder.pt` and can be loaded via `configs/model/audio_jepa_adapted.yaml`.*

### 6. Evaluate Model Checkpoint
Evaluates a checkpoint on validation or test split, computing CER and Thai WER:
```bash
uv run python -m thai_audio_jepa_asr.cli.evaluate --config configs/experiment/evaluate.yaml --checkpoint outputs/runs/smoke_test/checkpoints/best_checkpoint.pt --split test
```

### 7. Predict from a Single Audio File
Transcribes a WAV or MP3 file:
```bash
uv run python -m thai_audio_jepa_asr.cli.predict --checkpoint outputs/runs/smoke_test/checkpoints/best_checkpoint.pt --audio data_cache/porjai/audio/sample_000000.wav
```

### 8. Compare Completed Experiments
Collects summaries from `outputs/runs` and generates Markdown/CSV comparison tables:
```bash
uv run python -m thai_audio_jepa_asr.cli.compare --runs-dir outputs/runs
```

---

## 🧪 Running Unit Tests

Run the complete test suite with `pytest`:
```bash
uv run pytest tests/ -v
```

---

## 📁 Repository Structure

```
thai-audio-jepa-asr/
├── README.md
├── pyproject.toml
├── uv.lock
├── configs/
│   ├── data/porjai.yaml
│   ├── model/audio_jepa.yaml, audio_jepa_adapted.yaml, wav2vec2.yaml
│   └── experiment/smoke.yaml, audio_jepa_ctc.yaml, audio_jepa_thai_ssl.yaml, low_resource_*.yaml
├── src/thai_audio_jepa_asr/
│   ├── cli/             # 8 command-line interfaces
│   ├── data/            # Dataset, Porjai loader, collators
│   ├── models/          # ViT encoder, temporal adapter, CTC head, SSL module
│   ├── preprocessing/   # Normalizer, tokenizer, Kaldi fbank mel-spec
│   ├── training/        # ASR trainer, SSL trainer, optimizer, scheduler
│   ├── evaluation/      # CER, Thai WER, evaluator runner
│   └── utils/           # Seeds, logging, config loader, checkpointing, git info
├── tests/               # 6 comprehensive unit tests
├── outputs/             # Local experiment outputs & metrics
└── docs/
    ├── research_plan.md
    ├── experiment_protocol.md
    └── dataset_card_notes.md
```

---

## 🔍 The Audio-JEPA Temporal Resolution Bottleneck
- Pretrained Audio-JEPA uses $(16, 16)$ patches on $256 \times 128$ mel-spectrograms.
- This produces 16 temporal positions across 10 seconds of speech (**1.6 frames/second**).
- Because CTC mathematically requires $T_{\text{frames}} \ge L_{\text{characters}}$, any Thai sentence with $>16$ characters cannot be aligned directly.
- **Our Solution:** Configurable `temporal_adapter_mode: upsample_conv` ($4\times \rightarrow 64$ frames / 6.4 fps) which permits 100% CTC alignment, while keeping the unmodified baseline for scientific comparison.

---

## 📄 License
This project code is licensed under the Apache 2.0 License. The Porjai dataset is licensed under CC BY-SA 4.0. Pretrained Audio-JEPA weights are released under their respective author terms.
