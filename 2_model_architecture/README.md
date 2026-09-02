# Model Architecture

This note has two purposes:

1. summarize the published **Audio-JEPA** architecture faithfully; and
2. define a researchable architecture for converting the JEPA idea into a **Thai automatic speech recognition (ASR)** system.

The distinction is important: Audio-JEPA is a self-supervised **general-audio encoder**, not a complete speech-to-text model. The original paper evaluates frozen representations on classification and retrieval tasks, but omits the LibriSpeech-ASR task from its reported results. Therefore, the Thai ASR system below is a proposed extension, not an architecture or ASR result claimed by the Audio-JEPA authors.

## 1. Audio-JEPA summary

### 1.1 Core idea

Audio-JEPA transfers I-JEPA's masked latent-prediction objective from images to audio. It predicts the representation of hidden spectrogram patches instead of reconstructing waveform samples, spectrogram values, or discrete acoustic labels. It also does not require contrastive negative examples.

The model contains three modules during self-supervised pre-training:

- a **context encoder** that receives only visible spectrogram patches;
- a **target encoder** that receives the complete spectrogram and supplies stable target representations; and
- a smaller **predictor** that uses the visible context and positional mask tokens to predict the target encoder's representation at masked positions.

Only the context encoder and predictor receive gradients. The target encoder is updated as an exponential moving average (EMA) of the context encoder:

$$
\bar{\theta} \leftarrow \tau\bar{\theta} + (1-\tau)\theta.
$$

For a masked patch set $\mathcal{M}$, training minimizes the mean squared distance in representation space:

$$
\mathcal{L}_{\mathrm{JEPA}}
= \frac{1}{|\mathcal{M}|}
\sum_{j\in\mathcal{M}}
\left\|
g_{\phi}\!\left(f_{\theta}(x_{\setminus\mathcal{M}}), j\right)
- \operatorname{sg}\!\left[f_{\bar{\theta}}(x)_j\right]
\right\|_2^2,
$$

where `sg` denotes stop-gradient. At downstream inference, the predictor is discarded and an encoder is used to extract audio representations.

### 1.2 Published signal path

```text
10 s waveform at 32 kHz
        |
        v
128-band log-Mel spectrogram (256 time frames)
        |
        v
non-overlapping 16 x 16 time-frequency patches
        |
        +------------------------------+
        |                              |
 40-60% random mask              complete input
        |                              |
        v                              v
 context ViT-B/16                target ViT-B/16
        |                        (EMA, stop-gradient)
        v                              |
 6-layer predictor  -------------------+
        |
        v
L2 loss at masked patch positions
```

With 256 time frames and 128 Mel bands, the input becomes a grid of $16\times8=128$ patch tokens. The paper uses random independent patch masking throughout training; the number masked is sampled uniformly between 40% and 60%. Its preliminary experiments found I-JEPA-style block masking less effective for this general-audio setup.

### 1.3 Published architecture and training configuration

| Component | Configuration |
|---|---|
| Context encoder | ViT-Base; 12 layers; 768 hidden units; 12 heads; MLP ratio 4 |
| Target encoder | Architectural copy of context encoder; updated by EMA |
| Predictor | 6 Transformer layers; 384 hidden units; 12 heads; MLP ratio 4; projected back to 768 |
| Patch size | $16\times16$ time-frequency bins |
| Trainable parameters | 96.7M (context encoder + predictor) |
| Inference encoder | 85.4M parameters |
| Pre-training input | 10 s, 32 kHz AudioSet clips; 128 Mel bands; 256 time frames |
| Pre-training data | 1,921,982 clips, approximately 5,338 hours |
| Masking | random patches, 40%-60% per batch |
| Optimization | AdamW, batch size 256, 100,000 steps, warm-up plus cosine decay |

The paper reports 14 hours of pre-training on four NVIDIA V100 GPUs. Its main result is that this relatively direct audio adaptation learns competitive general-audio embeddings with substantially less pre-training than the wav2vec 2.0 and data2vec baselines used by the authors. However, performance is weak on several speech-centric X-ARES tasks, particularly keyword and command discrimination. This is evidence that speech-specific adaptation is necessary.

## 2. Why the published model is not directly suitable for ASR

ASR requires a dense, ordered sequence of acoustic representations so that phonetic events can be aligned with output symbols. The released Audio-JEPA frontend compresses 10 seconds into only 16 time-patch positions:

$$
\frac{10\ \mathrm{s}}{256\ \mathrm{frames}}\times16
\approx 0.625\ \mathrm{s/time\ patch}.
$$

A temporal resolution of approximately 625 ms is useful for semantic audio events, but it is too coarse for ordinary phone-, character-, or subword-level recognition. The two-dimensional token grid also contains eight separate frequency positions at each time position, whereas a CTC recognizer expects one ordered feature vector per time step.

Consequently, simply adding a linear CTC layer to the published checkpoint is not a strong main architecture. It is still useful as a baseline, but the Thai model needs:

- approximately 10-20 ms output steps rather than 625 ms steps;
- explicit conversion from time-frequency tokens to a time-only sequence;
- masking along time spans that correspond to speech units;
- an ASR objective and text vocabulary; and
- evaluation with Thai-aware text normalization and segmentation.

## 3. Relevant JEPA-to-ASR research

| Work | Representation | Temporal resolution | Downstream evidence | Lesson for this project |
|---|---|---:|---|---|
| **A-JEPA** (Fei et al., 2023/2024) | Mel spectrogram, ViT | classification-oriented | speech/audio classification | Shows that time-frequency-aware JEPA masking can work, but does not establish transcription performance. |
| **Audio-JEPA** (Tuncay et al., 2025) | Mel spectrogram, ViT | about 625 ms per time patch in the released setup | X-ARES classification/retrieval | Good reproducible starting point for general audio, but its frontend is too coarse for ASR. |
| **Audio JEPA design study** (Riou et al., 2024) | Mel spectrogram, ViT | setup-dependent | linear audio/speech probes | Random unstructured masking outperformed image-style multiblock and time-only masking in their general-audio experiments. Audio masking should be tested, not copied blindly from vision. |
| **WavJEPA** (Yuksel et al., 2025) | raw waveform, convolutional frontend + ViT | 10 ms | HEAR/ARCH general-audio tasks | Demonstrates a JEPA-compatible frontend with ASR-appropriate temporal density, although the paper does not report speech transcription. |
| **S-JEPA** (Ioannides et al., 2026 preprint) | raw waveform, HuBERT-style CNN + Transformer | 20 ms | SUPERB ASR with greedy CTC | Direct evidence that a speech-specific JEPA encoder can support ASR. It reports 12.10% WER on LibriSpeech test-clean with a frozen 51.8M encoder, while larger approximately 95M speech SSL baselines remain stronger. |

Two conclusions follow from this literature. First, **JEPA is best treated as the self-supervised pre-training method**, while CTC or a sequence-to-sequence decoder supplies the supervised transcription objective. Second, an ASR encoder must preserve fine temporal information even if a coarser semantic representation works well for clip-level audio tasks.

## 4. Proposed TAJA architecture

The recommended first model is **TAJA-CTC**: a high-resolution, spectrogram-based Thai Speech JEPA encoder followed by a character-level CTC head. This keeps the central contribution recognizably derived from Audio-JEPA while correcting its temporal bottleneck.

### 4.1 Stage A: Thai speech JEPA pre-training

```text
16 kHz Thai speech waveform
        |
        v
80-band log-Mel features
(25 ms window, 10 ms hop)
        |
        v
2-D convolutional patch/subsampling stem
(target output stride: 20 ms)
        |
        +-------------------------------+
        |                               |
 masked temporal spans             complete sequence
        |                               |
        v                               v
 context Transformer/Conformer     EMA target encoder
        |                           (stop-gradient)
        v                               |
 lightweight predictor ----------------+
        |
        v
normalized latent regression loss on masked frames
```

Recommended starting configuration:

| Part | Initial design |
|---|---|
| Acoustic input | 16 kHz mono; 80-bin log-Mel; 25 ms window; 10 ms hop |
| Subsampling | 2-D convolutional stem with total temporal stride 2, producing one vector every 20 ms |
| Context/target encoder | 12 layers, hidden size 768, 12 heads, feed-forward ratio 4 |
| Encoder block | Transformer for closest Audio-JEPA reproduction; Conformer as a planned ablation |
| Positional information | relative or rotary time positions; padding mask for variable-length utterances |
| Predictor | 4-6 Transformer layers, hidden size 384, projected to encoder dimension |
| Masking | contiguous time spans; 40%-60% masked frames; mix short and long spans |
| Target | normalized output of the EMA target encoder; optionally average its top $K$ layers |
| Loss | normalized MSE as the faithful baseline; smooth-L1 as an ablation |

The predictor and target branch are training-only. The online/context encoder becomes the acoustic encoder for ASR. If the released Audio-JEPA checkpoint is reused, only its Transformer block weights should be transferred; the new convolutional frontend and temporal positional representation must be initialized separately because the patch geometry has changed.

### 4.2 Stage B: supervised Thai CTC fine-tuning

```text
Thai waveform
    -> pretrained TAJA encoder (20 ms sequence)
    -> dropout
    -> linear projection to Thai vocabulary + CTC blank
    -> CTC loss
    -> greedy decoding or beam search + language model
    -> normalized Thai transcript
```

For an encoder sequence $H=(h_1,\ldots,h_T)$ and Thai label sequence $Y$, the supervised objective is:

$$
\mathcal{L}_{\mathrm{ASR}}=-\log
\sum_{\pi\in\mathcal{B}^{-1}(Y)}
\prod_{t=1}^{T}p(\pi_t\mid h_t),
$$

where $\mathcal{B}$ removes repeated labels and the CTC blank symbol. A character/grapheme vocabulary is the recommended first experiment because it avoids requiring word boundaries during model training and provides a clean comparison with Thai wav2vec 2.0 systems.

Thai text processing must be fixed before training:

- use one Unicode normalization policy for train, validation, test, and decoding;
- preserve Thai consonants, vowels, tone marks, digits, and any explicitly supported code-switch symbols;
- define consistent rules for punctuation, whitespace, numerals, abbreviations, and `ๆ`;
- compute **CER** directly on normalized character sequences; and
- compute **WER** only after applying the same fixed Thai word segmenter to references and hypotheses.

CER should be the primary architecture-development metric because Thai does not consistently delimit words with spaces. WER remains important for comparison and application quality. A beam-search decoder with a Thai language model should be reported separately from greedy CTC so that acoustic-encoder improvements are not confused with language-model gains. Earlier Thai wav2vec 2.0 work found that trigram language-model decoding materially improved WER, which makes this separation especially important.

### 4.3 Fine-tuning schedule

A stable low-resource schedule is:

1. train only the randomly initialized CTC projection while the encoder is frozen;
2. unfreeze the upper encoder layers;
3. unfreeze the complete encoder with a smaller learning rate than the CTC head; and
4. optionally retain a small JEPA auxiliary loss on unlabeled or labeled audio to reduce catastrophic forgetting.

The fourth step is experimental. The primary paper result should remain easy to interpret: JEPA pre-training followed by ordinary CTC fine-tuning.

## 5. Baselines and ablations required for a defensible paper

### 5.1 Baselines

At minimum, compare:

1. the same TAJA encoder trained from scratch with CTC;
2. Audio-JEPA Transformer weights transferred to the high-resolution frontend, then CTC fine-tuned;
3. Thai-speech JEPA pre-training followed by CTC fine-tuning (the proposed model);
4. a publicly available wav2vec 2.0/XLS-R Thai CTC model; and
5. if compute permits, a Whisper or MMS-family fine-tuning baseline.

This design isolates whether improvement comes from the architecture, generic AudioSet pre-training, Thai speech JEPA pre-training, or simply a stronger pretrained baseline.

### 5.2 High-value ablations

| Question | Values to test |
|---|---|
| Temporal resolution | 10, 20, and 40 ms |
| Mask geometry | random frames; contiguous time spans; mixed spans; time-frequency masking |
| Mask ratio | 40%, 50%, 60% |
| Target representation | final EMA layer vs. average of top 4/8 layers |
| Encoder block | Transformer vs. Conformer |
| Encoder use during ASR | frozen, partial fine-tuning, full fine-tuning |
| Output units | Thai graphemes vs. learned subwords |
| Decoding | greedy CTC vs. beam search; without vs. with the same language model |
| Pre-training data | generic AudioSet, Thai unlabeled speech, and sequential AudioSet-to-Thai adaptation |

The most important ablation is temporal resolution. Without it, a reviewer can reasonably argue that any ASR gain is due to replacing the original Audio-JEPA frontend rather than the JEPA objective.

## 6. Recommended experimental claim

A precise paper claim would be:

> We adapt Audio-JEPA's EMA teacher-student latent-prediction objective to high-resolution Thai speech and evaluate the learned encoder using a CTC recognizer.

Avoid claiming that Audio-JEPA itself is an ASR architecture or that a published Audio-JEPA ASR result has been reproduced. A strong study would show whether the JEPA objective improves CER/WER over an identical randomly initialized encoder and how it compares with established multilingual speech SSL encoders.

## 7. Research risks

- **Representation mismatch:** AudioSet features may emphasize acoustic events rather than Thai phonetic distinctions.
- **Checkpoint mismatch:** changing patch geometry prevents direct reuse of the original patch projection and fixed 2-D positional grid.
- **Loss of local detail:** latent semantic prediction can discard information needed to distinguish short vowels, consonants, and tones.
- **Target collapse or instability:** EMA schedule, normalization, mask ratio, and predictor capacity require monitoring.
- **Evaluation leakage:** speakers and duplicate utterances must not cross dataset splits.
- **Thai WER ambiguity:** conclusions can change with the word segmenter; report CER and name/version the WER tokenizer.
- **Confounded decoding gains:** report greedy and language-model-assisted results separately.

## 8. Primary sources

- Tuncay et al., [Audio-JEPA: Joint-Embedding Predictive Architecture for Audio Representation Learning](https://arxiv.org/abs/2507.02915), ICME 2025; [official implementation](https://github.com/LudovicTuncay/Audio-JEPA).
- Fei et al., [A-JEPA: Joint-Embedding Predictive Architecture Can Listen](https://arxiv.org/abs/2311.15830), 2023/2024.
- Riou et al., [Investigating Design Choices in Joint-Embedding Predictive Architectures for General Audio Representation Learning](https://arxiv.org/abs/2405.08679), 2024.
- Yuksel et al., [WavJEPA: Semantic Learning Unlocks Robust Audio Foundation Models for Raw Waveforms](https://arxiv.org/abs/2509.23238), 2025.
- Ioannides et al., [S-JEPA: Soft Clustering Anchors for Self-Supervised Speech Representation Learning](https://arxiv.org/abs/2606.19398), 2026 preprint.
- Zhang et al., [X-ARES: A Comprehensive Framework for Assessing Audio Encoder Performance](https://www.isca-archive.org/interspeech_2025/zhang25d.html), Interspeech 2025; [official implementation](https://github.com/jimbozhang/xares).
- Phatthiyaphaibun et al., [Thai Wav2Vec2.0 with CommonVoice V8](https://arxiv.org/abs/2208.04799), 2022.
- Babu et al., [XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale](https://arxiv.org/abs/2111.09296), 2021.
