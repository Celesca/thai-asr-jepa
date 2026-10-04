# Thai Audio-JEPA ASR Research Project

[![Python 3.12](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Branch](https://img.shields.io/badge/Branch-feat%2Faudio--jepa--thai--asr--temporal-green.svg)](https://github.com)

A reproducible research codebase adapting **Audio-JEPA** ([ICME 2025](https://arxiv.org/abs/2507.02915), `ltuncay/Audio-JEPA`) for Thai Automatic Speech Recognition (ASR) with Connectionist Temporal Classification (CTC) on the **Porjai Central Thai** speech corpus (`CMKL/Porjai-Thai-voice-dataset-central`).

Detailed architectural rationale and layer-by-layer transfer details are documented in [`docs/architecture_change.md`](file:///home/inteoty/Documents/ThaiAudioJEPA/docs/architecture_change.md).

---

## 📌 Architectural Modification Overview

### Why Finer Temporal Patching?
The original Audio-JEPA uses coarse `(16, 16)` square patches on a $256 \times 128$ mel-spectrogram. Across a 10-second audio clip, this creates only **16 temporal positions** (**1.6 frames/sec**, 625 ms/token).

In CTC speech recognition, the alignment condition strictly requires:
$$T_{\text{output}} \ge \text{target\_length} + \text{adjacent\_repeated\_token\_count}$$

Under the coarse 16-frame grid, more than 95% of Thai utterances cannot physically align. Rather than interpolating tokens *after* the encoder, this project modifies the **input patch embedding stage** before Transformer encoding:
- **Asymmetric Patch Size:** `patch_size: {time: 1, frequency: 16}`
- **Temporal Resolution:** **25.6 frames/second** (~39 ms/token, 256 temporal tokens per 10s clip).
- **Feasibility:** **100.0%** of Porjai Thai speech utterances are alignable.
- **Pretrained Transfer:** **99.99%** of model weights transfer directly from `ltuncay/Audio-JEPA` (86,629,632 / 86,641,920 parameters), reinitializing only the patch projection layer and interpolating 2D positional embeddings bicubically.

---

## 🔬 Experimental Configurations

| ID | Configuration | Patch Size | Temporal Rate | Initialization | Feasibility | Purpose |
|---|---|---|---|---|---|---|
| **A** | Original Audio-JEPA + CTC | `(16, 16)` | 1.6 fps | `JEPA.ckpt` (100%) | <5.0% | Diagnostic baseline |
| **B** | Modified Audio-JEPA + CTC | `(1, 16)` | 25.6 fps | `JEPA.ckpt` (99.99%) | 100.0% | Core Transfer |
| **C** | Modified Audio-JEPA + Thai SSL + CTC | `(1, 16)` | 25.6 fps | `JEPA.ckpt` $\rightarrow$ Thai SSL | 100.0% | Domain adaptation |

---

## ⚙️ Installation & Setup (via `uv`)

All dependencies are locked and managed with `uv`:

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 2. Sync dependencies
uv sync

# 3. Check Git branch status
uv run python -c "from thai_audio_jepa_asr.utils.git_info import get_git_info; print(get_git_info())"
```

---

## 🚀 Execution Commands

### 1. Inspect Checkpoint Transfer & Temporal Rate
Inspects original (1.6 fps) vs modified (25.6 fps) token rates and generates `outputs/checkpoint_loading_report.json`:
```bash
uv run python -m thai_audio_jepa_asr.cli.inspect_audio_jepa
```

### 2. Run Thai SSL Adaptation
Performs continued self-supervised pretraining with contiguous temporal masking:
```bash
# Full Thai SSL continued pretraining
uv run python -m thai_audio_jepa_asr.cli.adapt_ssl --config configs/experiment/audio_jepa_thai_ssl.yaml

# Fast smoke test (1 epoch, verified)
uv run python -m thai_audio_jepa_asr.cli.adapt_ssl --config configs/experiment/smoke_ssl.yaml
```
Saves adapted encoder checkpoint to `outputs/runs/audio_jepa_thai_ssl/checkpoints/adapted_encoder.pt`.

### 3. Train Supervised CTC ASR
Trains the modified Audio-JEPA encoder + CTC head:
```bash
# Full dataset supervised training
uv run python -m thai_audio_jepa_asr.cli.train_asr --config configs/experiment/audio_jepa_ctc.yaml

# Fast CTC smoke test (1 epoch, verified)
uv run python -m thai_audio_jepa_asr.cli.train_asr --config configs/experiment/smoke.yaml
```

### 4. Evaluate Checkpoint & Generate Predictions
Computes Character Error Rate (CER), Thai Word Error Rate (WER via PyThaiNLP), and prints sample transcriptions:
```bash
uv run python -m thai_audio_jepa_asr.cli.evaluate \
  --checkpoint outputs/runs/smoke_ctc/checkpoints/best_checkpoint.pt \
  --split val
```

### 5. Compare Completed Runs
Generates comparative Markdown tables across configurations:
```bash
uv run python -m thai_audio_jepa_asr.cli.compare --runs-dir outputs/runs
```

---

## 🧪 Comprehensive Unit Test Suite

Run all 27 unit tests:
```bash
uv run pytest tests/ -v
```

Tests cover:
- Checkpoint parameter loading percentage (>99.98%)
- Asymmetric patching tensor dimensions `(B, 1, 256, 128) -> (B, 2048, 768) -> (B, 256, 768)`
- 2D positional embedding grid interpolation `(16, 8) -> (256, 8)`
- CTC feasibility validation ($T_{\text{output}} \ge L_{\text{target}} + \text{repeats}$)
- End-to-end forward, backward, and optimization step
- Thai text normalization, char tokenization, collators, and splits.

---

## 📁 Repository Structure

```text
thai-audio-jepa-asr/
├── README.md
├── pyproject.toml
├── uv.lock
├── configs/
│   ├── model/
│   │   ├── audio_jepa_original.yaml  # Config A: coarse (16, 16) baseline
│   │   ├── audio_jepa.yaml           # Config B: asymmetric (1, 16) native pooling
│   │   └── audio_jepa_adapted.yaml   # Config C: SSL-adapted encoder
│   └── experiment/
│       ├── smoke_ssl.yaml            # Fast SSL adaptation check
│       ├── smoke.yaml                # Fast CTC training check
│       ├── audio_jepa_thai_ssl.yaml  # Full Thai SSL pretraining
│       └── audio_jepa_ctc.yaml       # Full supervised ASR
├── docs/
│   ├── architecture_change.md        # Comprehensive design rationale & report
│   ├── research_plan.md
│   └── dataset_card_notes.md
├── src/thai_audio_jepa_asr/
│   ├── cli/                          # Command-line interfaces
│   ├── data/                         # Dataset, collator (contiguous temporal masking)
│   ├── models/                       # ViT encoder, temporal adapter, CTC head, SSL module
│   ├── preprocessing/                # Normalizer, char tokenizer, Kaldi fbank
│   ├── training/                     # ASR trainer, SSL trainer, optimizer
│   └── utils/                        # Seeds, config, logging, git info, checkpoint report
└── tests/                            # 27 automated unit tests
```

---

## 📄 License
This project code is licensed under the Apache 2.0 License. The Porjai dataset is licensed under CC BY-SA 4.0. Pretrained Audio-JEPA weights are released under their respective author terms.
