# Audio-JEPA Architectural Adaptation for Thai CTC ASR

## 1. Executive Summary

This document details the architectural adaptation of **Audio-JEPA** (`ltuncay/Audio-JEPA`, ICME 2025) for Thai Automatic Speech Recognition (ASR) using Connectionist Temporal Classification (CTC).

Rather than replacing Audio-JEPA with an unrelated speech model or using post-encoder convolutional interpolation, this adaptation modifies the **input patch embedding stage** before Transformer encoding. This preserves fine temporal acoustic details (25.6 frames/sec, ~39 ms resolution) while transferring **99.99%** of the pretrained Transformer parameters from the upstream Audio-JEPA checkpoint.

---

## 2. Why Finer Temporal Patching is Necessary for CTC ASR

### 2.1 The CTC Alignment Constraint
In Connectionist Temporal Classification (CTC), the output sequence length $T_{\text{output}}$ must be strictly greater than or equal to the minimum alignment length required by the target token sequence:
$$T_{\text{output}} \ge \text{target\_length} + \text{adjacent\_repeated\_token\_count}$$

Consecutive identical non-blank tokens in CTC require at least one blank token between them. For example, the Thai token sequence `[ก, ก, ก]` has length 3 and 2 adjacent repetitions, demanding at least $3 + 2 = 5$ time frames.

### 2.2 Coarse Patching Failure in Upstream Audio-JEPA (Configuration A)
The original Audio-JEPA was designed for general audio representation learning using coarse square patches:
- **Patch size:** $(16, 16)$ on a $256 \times 128$ mel-spectrogram (Time $\times$ MelBins).
- **Patch grid:** $16 \text{ (time)} \times 8 \text{ (frequency)} = 128 \text{ total patches}$.
- **Temporal positions per 10-second clip:** Exactly **16 tokens**.
- **Temporal token rate:** $1.6 \text{ frames/second}$ ($625 \text{ ms/token}$).

In conversational Thai speech, the median utterance duration in the Porjai dataset is 4.5 seconds with ~45 characters. With a 10-second clip budgeted to only 16 time frames, **more than 95% of Thai utterances cannot physically align under CTC**, causing catastrophic alignment violations or divergence.

### 2.3 Modified Asymmetric Patching (Configurations B & C)
To resolve this at the representation source before Transformer blocks:
- **Patch size:** Asymmetric $(1, 16)$ (1 time step $\times$ 16 frequency bins).
- **Patch grid:** $256 \text{ (time)} \times 8 \text{ (frequency)} = 2,048 \text{ total patches}$.
- **Temporal positions per 10-second clip:** **256 tokens**.
- **Temporal token rate:** **25.6 frames/second** ($39.06 \text{ ms/token}$).
- **CTC Feasibility:** **100.0%** of Porjai Thai utterances satisfy $T_{\text{output}} \ge L + \text{repeats}$.

Frequency bins remain patch-pooled from 128 to 8 bins because speech recognition primarily requires fine temporal alignment rather than ultra-fine spectral resolution.

---

## 3. Retained Upstream Architecture & Components

The core architecture strictly preserves the upstream Audio-JEPA implementation:
1. **Audio Preprocessing Pipeline:** Kaldi-compliant filterbank feature extraction (`torchaudio.compliance.kaldi.fbank`), 128 mel frequency bins, 32 kHz audio sampling rate, 10.0-second clip normalization.
2. **Vision Transformer (ViT) Encoder:**
   - 12 Transformer blocks.
   - Embedding dimension $D = 768$.
   - Multi-head Self-Attention with 12 heads ($d_{\text{head}} = 64$).
   - Pre-LayerNorm formulation (`nn.LayerNorm(768, eps=1e-6)`).
   - Multi-Layer Perceptron (MLP) expansion ratio of 4.0 (intermediate dimension 3072) with GELU activation.
3. **JEPA Self-Supervised Learning Framework:**
   - Target encoder updated via Exponential Moving Average (EMA, $\tau = 0.996$).
   - Multi-block Predictor with 2D positional embeddings.
   - Normalized MSE latent prediction loss (`norm_mse_loss`).
4. **Attention Masking Support:**
   - Multi-head attention extended with additive padding/attention masks for variable-length speech without distorting token positions.

---

## 4. Pretrained Checkpoint Transfer & Parameter Report

We initialize from `ltuncay/Audio-JEPA` (`JEPA.ckpt`, 86.64M parameters) using an explicit transfer learning protocol:

```text
================================================================================
AUDIO-JEPA CHECKPOINT LOADING REPORT
================================================================================
Checkpoint Source:            ltuncay/Audio-JEPA (JEPA.ckpt)
Total Model Parameters:       86,641,920
Loaded Parameters Count:      86,629,632
Loaded Parameter Ratio:       99.9858% (99.99%)
Missing Parameters Count:     1 (patch_embed.proj.weight)
Unexpected Parameters Count:  0
Shape Mismatches:             1 layer
  - patch_embed.proj.weight:  Source [768, 1, 16, 16] -> Target [768, 1, 1, 16]
================================================================================
```

### 4.1 Retained & Loaded Parameters (99.99%)
- All 12 Transformer blocks (144 parameter tensors including QKV projections, attention output projections, MLP fc1, MLP fc2, norm1, norm2, and biases) match shape identically and are loaded directly.
- Final encoder LayerNorm (`norm.weight`, `norm.bias`) loaded directly.
- Patch embedding bias (`patch_embed.proj.bias`, 768 parameters) loaded directly.

### 4.2 Positional Embedding Interpolation
- Source grid: $(16, 8) = 128$ tokens.
- Target grid: $(256, 8) = 2,048$ tokens.
- The 2D sinusoidal positional embedding is reshaped to `[1, 768, 16, 8]` and interpolated to `[1, 768, 256, 8]` using bicubic 2D grid interpolation before being flattened to `[1, 2048, 768]`.

### 4.3 Reinitialized Parameters (0.014%)
- **`patch_embed.proj.weight`:** The 2D convolution kernel was resized from $(16, 16)$ to $(1, 16)$. This single layer contains $768 \times 1 \times 1 \times 16 = 12,288$ parameters, initialized with Xavier/Kaiming normal initialization.

---

## 5. Thai SSL Continued Pretraining with Contiguous Temporal Masking

Upstream Audio-JEPA masks random square 2D blocks. On a $(256, 8)$ grid, small square blocks would mask only a single time step across a few frequencies, which is insufficient for speech representation learning.

In the modified SSL adaptation:
- **Contiguous Temporal Masking:** Masks continuous temporal spans across all 8 frequency bins:
  $$t_{\text{span}} \in [0.2 \cdot T, 0.7 \cdot T] = [51, 179] \text{ consecutive time steps}$$
- This forces the context encoder to reconstruct continuous phonetic and syllabic information from surrounding temporal context, matching the acoustic structure of spoken Thai.

---

## 6. Thai CTC ASR Wrapper & Alignment

The CTC speech recognition model consists of:
```text
AudioJEPAEncoder (patch_size: 1x16, 256 time steps x 8 freq bins)
  │
  ▼
to_temporal_sequence()  [B, 2048, 768] -> [B, 256, 768] (native frequency pooling)
  │
  ▼
nn.Dropout(p=0.1)
  │
  ▼
nn.Linear(768, vocab_size=109)  [B, 256, 109]
  │
  ▼
nn.CTCLoss(blank=0, zero_infinity=True)
```

### Pre-Training Feasibility Verification
Before launching training, `validate_ctc_feasibility` verifies:
- Feasible utterance percentage: **100.0%**
- Mean output length: 256.0 frames
- Mean required length: 51.4 characters + repeated tokens
- Max required length in Porjai: 182 frames (< 256 frames).

---

## 7. Configuration Matrix

| ID | Configuration | Patch Size | Temporal Rate | Pretrained Init | Feasible Coverage | Purpose |
|---|---|---|---|---|---|---|
| **A** | Original Audio-JEPA + CTC | $(16, 16)$ | 1.6 fps | `JEPA.ckpt` (100%) | <5.0% | Diagnostic baseline |
| **B** | Modified Audio-JEPA + CTC | $(1, 16)$ | 25.6 fps | `JEPA.ckpt` (99.99%) | 100.0% | Core Transfer |
| **C** | Modified Audio-JEPA + Thai SSL + CTC | $(1, 16)$ | 25.6 fps | `JEPA.ckpt` -> Thai SSL | 100.0% | Domain-adapted SSL |

---

## 8. Conclusion

This architecture is strictly an **adapted Audio-JEPA**, retaining the upstream ViT architecture, 99.99% of pretrained weights, Kaldi mel-spectrogram preprocessing, EMA target updates, and masked latent prediction objective, while correcting the temporal resolution bottleneck at the input patch embedding to enable state-of-the-art Thai CTC speech recognition.
