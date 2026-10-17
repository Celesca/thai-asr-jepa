# Audio-JEPA Architectural Adaptation for Thai CTC ASR

## 1. Executive Summary

This document details the architectural adaptation of **Audio-JEPA** (`ltuncay/Audio-JEPA`, ICME 2025) for Thai Automatic Speech Recognition (ASR) using Connectionist Temporal Classification (CTC).

Rather than replacing Audio-JEPA with an unrelated speech model or using post-encoder convolutional interpolation, this adaptation addresses the temporal resolution bottleneck at the **input frontend and patch embedding stage** before Transformer encoding. This preserves fine temporal acoustic details (25–50 frames/sec, 20–40 ms resolution) while transferring **99.99%** of the pretrained Transformer parameters from the upstream Audio-JEPA checkpoint.

---

## 2. Why Finer Temporal Patching is Necessary for CTC ASR

### 2.1 The CTC Alignment Constraint
In Connectionist Temporal Classification (CTC), the output sequence length $T_{\text{output}}$ must be strictly greater than or equal to the minimum alignment length required by the target token sequence:
$$T_{\text{output}} \ge U + R$$
where $U$ is the target token sequence length and $R$ is the number of consecutive identical non-blank tokens.

Consecutive identical non-blank tokens in CTC require at least one blank token between them. For example, the Thai token sequence `[ก, ก, ก]` has length 3 and 2 adjacent repetitions, demanding at least $3 + 2 = 5$ time frames.

### 2.2 Coarse Patching Failure in Upstream Audio-JEPA (Configuration A)
The original Audio-JEPA was designed for general audio representation learning using coarse square patches:
- **Patch size:** $(16, 16)$ on a $256 \times 128$ mel-spectrogram (Time $\times$ MelBins).
- **Patch grid:** $16 \text{ (time)} \times 8 \text{ (frequency)} = 128 \text{ total patches}$.
- **Temporal positions per 10-second clip:** Exactly **16 tokens**.
- **Temporal token rate:** $1.6 \text{ frames/second}$ ($625 \text{ ms/token}$).

In conversational Thai speech, the median utterance duration in the Porjai dataset is 4.5 seconds with ~45 characters. With a 10-second clip budgeted to only 16 time frames, **more than 95% of Thai utterances cannot physically align under CTC**, causing catastrophic alignment violations or divergence.

### 2.3 Frontend & Patching Architectures

We provide three configurable frontend implementations (`original`, `speech_40ms`, `speech_20ms`):

| Frontend Config | Sample Rate | Mel Bins | Stride / Hop | ViT Time Tokens (10s) | Resolution / Rate | Patch Size |
|---|---|---|---|---|---|---|
| `original` | 32,000 Hz | 128 | 39.06 ms | 256 tokens | 39.06 ms (25.6 fps) | $(1, 16)$ |
| `speech_40ms` | 16,000 Hz | 80 | 10.00 ms (raw) | 250 tokens | 40.00 ms (25.0 fps) | Conv2d $4\times$ stride |
| `speech_20ms` | 16,000 Hz | 80 | 10.00 ms (raw) | 500 tokens | 20.00 ms (50.0 fps) | Conv2d $2\times$ stride |

#### 2.3.1 Exact Valid Frame and Token Calculation
Padded tensor lengths are never passed to CTC loss as valid input lengths. Instead, the exact number of valid acoustic frames and ViT time tokens are computed directly from the unpadded waveform length $N_{\text{samples}}$:
- **Kaldi Fbank frames:**
  $$T_{\text{fbank}} = \left\lfloor \frac{N_{\text{samples}} - N_{\text{win}}}{N_{\text{hop}}} \right\rfloor + 1$$
- **ViT Time Tokens:**
  $$T_{\text{valid}} = \left\lfloor \frac{T_{\text{fbank}} - K_t}{S_t} \right\rfloor + 1$$
For the `original` frontend, $K_t=1, S_t=1$, capped at the grid limit ($256$). For `speech_40ms`, $K_t=4, S_t=4$. For `speech_20ms`, $K_t=3, S_t=2$.

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
   - Target encoder updated via Exponential Moving Average (EMA, $\tau \in [0.996, 1.0]$ scheduled cosine).
   - Multi-block Predictor with 2D positional embeddings.
   - Normalized MSE latent prediction loss (`norm_mse_loss`).
4. **Attention Masking (SDPA):**
   - Padding tokens are excluded from attention using an additive float mask ($0.0$ for valid positions, $-10000.0$ for padded positions).
5. **Dynamic Positional Embeddings:**
   - Position embeddings are sliced dynamically along the temporal axis (`self.pos_embed[:, :x.shape[1]]`) to support variable sequence lengths without spatial index mismatch.

---

## 4. Pretrained Checkpoint Transfer & Parameter Decoupling

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
Note: Missing weights are expected when modifying patch embedding kernel from (16,16) to (1,16).
================================================================================
```

### 4.1 Parameter Group Decoupling in Optimization
Because `patch_embed.proj.weight` is reinitialized due to the kernel reshape, it must **never** be frozen when the Transformer backbone is frozen:
- **`freeze_patch_embed: false`**: Default setting. Even when `freeze_encoder: true`, `patch_embed` remains trainable.
- **Decoupled Optimizer Groups:**
  1. `backbone`: 12 ViT blocks + norm (LR: $\eta_{\text{backbone}}$ or frozen)
  2. `patch_embed`: Conv2d projection + bias (LR: $\eta_{\text{patch\_embed}} \ge \eta_{\text{backbone}}$)
  3. `head`: Temporal adapter + CTC linear classifier (LR: $\eta_{\text{head}}$)

---

## 5. Thai SSL Continued Pretraining with Short-Span Masking

Upstream Audio-JEPA masks random square 2D blocks. For speech representation learning:
- **Short-Span Masking:** Masks continuous temporal spans between 200 ms and 400 ms (5 to 10 time tokens) across all frequency bins.
- **Valid Position Sampling:** Spans are sampled strictly within each utterance's valid time tokens, ensuring context and target patches never sample zero-padding.
- **Target Coverage:** Targets cover 40%–60% of the active utterance length.
- **Latent Diagnostics:** During pretraining, collapse diagnostics (`feature_variance`, `inter_utterance_similarity`) are logged at every evaluation step to guarantee latent space diversity.

---

## 6. Thai CTC ASR Pipeline

```text
Input Mel-Spectrogram  [B, 1, T_fbank, F_bins]
  │
  ▼
PatchEmbed  [B, total_patches, 768] (with dynamic pos_embed slice)
  │
  ▼
AudioJEPA ViT Encoder (12 blocks, additive SDPA padding mask)
  │
  ▼
TemporalAdapter.to_temporal_sequence()  [B, total_patches, 768] -> [B, T_time, 768]
  │
  ▼
nn.Dropout(p=0.1)
  │
  ▼
CTC Linear Projection  [B, T_time, vocab_size=109]
  │
  ▼
nn.CTCLoss(blank=0, zero_infinity=False) with lengths = valid_time_tokens
```

### Feasibility Guarantee
`validate_ctc_feasibility` inspects all dataset batches and reports the minimum, median, mean, and maximum ratio $T_{\text{valid}} / (U + R)$. In Porjai Thai speech, all utterances strictly satisfy $T_{\text{valid}} \ge U + R$.

---

## 7. Configuration Matrix

| ID | Configuration | Patch Size / Stride | Temporal Rate | Pretrained Init | CTC Feasibility | Purpose |
|---|---|---|---|---|---|---|
| **A** | Original Audio-JEPA + CTC | $(16, 16)$ | 1.6 fps | `JEPA.ckpt` (100%) | <5.0% | Diagnostic baseline |
| **B** | Modified Audio-JEPA + CTC | $(1, 16)$ | 25.6 fps | `JEPA.ckpt` (99.99%) | 100.0% | Core Transfer |
| **C** | Modified Audio-JEPA + Thai SSL + CTC | $(1, 16)$ | 25.6 fps | `JEPA.ckpt` -> Thai SSL | 100.0% | In-domain SSL pretraining |
| **D** | Random Init + Thai SSL + CTC | $(1, 16)$ | 25.6 fps | Scratch -> Thai SSL | 100.0% | Isolates Audio-JEPA upstream value |
| **Ref** | Wav2Vec2 Thai Baseline | Conv 20ms | 50.0 fps | `airesearch/wav2vec2-large-xlsr-53-th` | 100.0% | Speech-native reference |

---

## 8. Conclusion

This architecture strictly preserves the upstream Audio-JEPA foundation while resolving every speech-specific bottleneck: temporal frame rate, variable sequence attention masking, unpadded CTC loss lengths, patch embedding parameter decoupling, and short-span phonemic SSL masking.
