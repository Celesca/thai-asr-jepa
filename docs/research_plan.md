# Research Plan: Thai Speech Recognition via Audio-JEPA Self-Supervised Adaptation

## 1. Research Question & Motivation
**Core Question:**
> *“Does speech-oriented adaptation and Thai self-supervised continued pretraining improve Audio-JEPA’s performance and labeled-data efficiency for Thai ASR?”*

Audio-JEPA (Joint-Embedding Predictive Architecture for Audio, ICME 2025) learns non-generative semantic audio representations by predicting target patch embeddings in latent space from context patches. While pretrained across general environmental and mixed audio (AudioSet ~5.3k hours), its transferability to tonal languages like Thai remains uncharacterized.

Thai speech presents distinctive acoustic challenges:
1. **Phonemic Tone Contours:** Five lexical tones (mid, low, falling, high, rising) distinguishing semantic word meaning.
2. **Vowel Length Distinctions:** Short vs long vowel pairs altering lexical identity.
3. **Continuous Script:** Lack of inter-word whitespace, necessitating character-level acoustic modeling and PyThaiNLP dictionary tokenization for word error rate evaluation.

---

## 2. Experimental Hypotheses
- **$H_1$ (Temporal Alignment & Padding Feasibility):** Upstream Audio-JEPA's coarse $16\times 16$ patches (1.6 fps) violate CTC length constraints on >95% of Thai utterances. An adapted speech patch embedding ($25.6$ fps) with exact valid-length calculation achieves 100% CTC alignment without training against zero-padded silence.
- **$H_2$ (Upstream Feature Transfer):** Direct transfer of Audio-JEPA weights (`JEPA.ckpt`, 99.99% parameter transfer) yields superior convergence and lower Character Error Rate (CER) compared to training an identical ViT architecture from random scratch initialization.
- **$H_3$ (Thai In-Domain Continued SSL):** Short-span temporal SSL pretraining on unlabeled Thai Porjai speech further refines representations for Thai acoustic-phonetic structures, outperforming raw upstream transfer.
- **$H_4$ (Low-Resource Sample Efficiency):** Thai SSL pretraining provides the largest relative CER gains under restricted labeled data regimes (1 hour, 10 hours, 50 hours).

---

## 3. Experimental Matrix

### 3.1 Model Configurations
1. **Experiment A (Scratch CTC Baseline):**
   - ViT architecture initialized from scratch (random weights).
   - Fine-tuned with CTC loss.
   - Config: `configs/experiment/exp_A_scratch_ctc.yaml`.
2. **Experiment B (Pretrained Audio-JEPA CTC):**
   - Transformer initialized from `JEPA.ckpt` (99.99% loaded).
   - Patch embedding trained, backbone fine-tuned.
   - Config: `configs/experiment/exp_B_pretrained_ctc.yaml`.
3. **Experiment C (Pretrained Audio-JEPA + Thai SSL -> CTC):**
   - Transformer initialized from `JEPA.ckpt`.
   - Continued pretraining with short-span JEPA latent prediction on unlabeled Thai speech (`configs/experiment/exp_C_ssl_pretrain.yaml`).
   - Downstream CTC fine-tuning (`configs/experiment/exp_C_thai_ssl_ctc.yaml`).
4. **Experiment D (Scratch + Thai SSL -> CTC):**
   - Randomly initialized ViT pre-trained with Thai SSL, then CTC fine-tuned.
   - Disentangles whether gains stem from the JEPA SSL objective versus the upstream AudioSet weights.
   - Config: `configs/experiment/exp_D_scratch_ssl_ctc.yaml`.
5. **Reference Baseline (Wav2Vec2 Thai):**
   - Pretrained speech-native acoustic model (`airesearch/wav2vec2-large-xlsr-53-th`).
   - Config: `configs/experiment/exp_ref_wav2vec2.yaml`.

### 3.2 Labeled Supervision Budgets
Evaluated across nested subsets:
- **1 Hour:** ~700–800 utterances (`configs/experiment/budget_1h.yaml`)
- **10 Hours:** ~7,000–8,000 utterances (`configs/experiment/budget_10h.yaml`)
- **50 Hours:** ~35,000–40,000 utterances
- **Full Dataset:** ~700 hours

---

## 4. Controlled Variables & Experimental Rigor

- **Dataset & Partitions:** `CMKL/Porjai-Thai-voice-dataset-central`.
  - *Speaker Metadata Note:* Porjai lacks explicit speaker IDs; train/val/test splits are constructed via deterministic SHA-256 hash partitioning of utterance IDs (80% train, 10% validation, 10% test), preventing data leakage.
  - *Length Filtering:* Utterances $>10.0$s are strictly excluded rather than truncated, preserving 100% audio-transcript consistency. Excluded sample counts and hours are reported.
- **Normalization & Vocabulary:** Unicode NFKC normalization, standardized 109-character Thai vocabulary (`pythainlp` tokenization for WER).
- **Optimization Decoupling:** Backbone, patch embedding, and CTC head use decoupled parameter groups. In frozen backbone runs, `patch_embed` remains trainable.
- **Model Selection Integrity:** The best checkpoint is chosen strictly using validation Character Error Rate (CER). Test set evaluation occurs exactly once after training completion.

---

## 5. Result Reporting Matrix

| Model Variant | Upstream Init | Thai SSL Pretraining | Labeled Budget | Freeze Policy | Val CER (%) | Test CER (%) | Test WER (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A: Scratch CTC** | None | None | 100% | Full FT | TBD | TBD | TBD |
| **B: Pretrained CTC** | `JEPA.ckpt` | None | 1h | Full FT | TBD | TBD | TBD |
| **B: Pretrained CTC** | `JEPA.ckpt` | None | 10h | Full FT | TBD | TBD | TBD |
| **B: Pretrained CTC** | `JEPA.ckpt` | None | 100% | Frozen BB | TBD | TBD | TBD |
| **B: Pretrained CTC** | `JEPA.ckpt` | None | 100% | Full FT | TBD | TBD | TBD |
| **C: Thai SSL -> CTC** | `JEPA.ckpt` | Porjai (200-400ms) | 1h | Full FT | TBD | TBD | TBD |
| **C: Thai SSL -> CTC** | `JEPA.ckpt` | Porjai (200-400ms) | 10h | Full FT | TBD | TBD | TBD |
| **C: Thai SSL -> CTC** | `JEPA.ckpt` | Porjai (200-400ms) | 100% | Full FT | TBD | TBD | TBD |
| **D: Scratch SSL -> CTC** | None | Porjai (200-400ms) | 100% | Full FT | TBD | TBD | TBD |
| **Ref: Wav2Vec2 Thai** | XLS-R | None | 100% | Full FT | TBD | TBD | TBD |
