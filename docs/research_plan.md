# Research Plan: Thai Speech Recognition via Audio-JEPA Self-Supervised Adaptation

## 1. Research Question & Motivation
**Core Question:**
> *“Can Thai-specific self-supervised adaptation of Audio-JEPA improve Thai ASR under limited labeled-data conditions?”*

Audio-JEPA (Joint-Embedding Predictive Architecture for Audio, ICME 2025) learns non-generative semantic audio representations by predicting target patch embeddings in latent space from context patches. While pretrained across general environmental and mixed audio (AudioSet ~5.3k hours), its transferability to non-tonal vs tonal languages like Thai remains uncharacterized.

Thai speech presents distinctive acoustic challenges:
1. **Phonemic Tone Contours:** Five lexical tones (mid, low, falling, high, rising) distinguishing semantic word meaning.
2. **Vowel Length Distinctions:** Short vs long vowel pairs altering lexical identity.
3. **Continuous Script:** Lack of inter-word whitespace, necessitating character-level acoustic modeling and dictionary-based evaluation.

## 2. Hypotheses
- **$H_1$ (Temporal Feasibility):** The native Audio-JEPA patch grid ($16 \times 8$ patches on 10s audio) provides only 1.6 temporal positions/sec, which is insufficient for CTC alignment of Thai utterances without temporal adaptation.
- **$H_2$ (Thai In-Domain Adaptation):** Continuing the masked latent prediction objective on unlabeled Thai speech will adapt the ViT representations to Thai formant and tonal characteristics, yielding lower downstream Character Error Rates (CER).
- **$H_3$ (Low-Resource Sample Efficiency):** Thai SSL adaptation provides the greatest relative WER/CER improvements when labeled supervision is scarce (e.g. 10% and 25% data budgets).

## 3. Experimental Stages

### Stage A: Supervised ASR Baselines
- **A1: Audio-JEPA Frozen Encoder + CTC Head**
  - Evaluates raw feature quality without speech task fine-tuning.
- **A2: Audio-JEPA Full Fine-Tuning + CTC Head**
  - Evaluates end-to-end representation adaptation.

### Stage B: Thai SSL Adaptation
- **B1: Out-of-the-Box Pretrained Audio-JEPA + CTC**
- **B2: Thai-Adapted Audio-JEPA + CTC**
  - Stage B2 takes the original Audio-JEPA checkpoint and trains under the Audio-JEPA masked latent prediction objective on unlabeled Thai audio from the Porjai training partition (completely uncoupled from validation/test transcripts), followed by CTC fine-tuning.

### Stage C: Low-Resource Labeled Data Study
Evaluates sample efficiency across labeled data budgets:
- **10% budget** (~50–70 hours)
- **25% budget** (~175 hours)
- **50% budget** (~350 hours)
- **100% budget** (~700 hours)

### Stage D: Comparative Architecture Analysis
Evaluates Audio-JEPA against speech-specific self-supervised baselines (Wav2Vec2 / HuBERT) through a shared unified speech encoder interface.

---

## 4. Controlled Variables & Experimental Rigor
- **Identical Audio Representation:** 32 kHz mono audio, Kaldi fbank mel-spectrograms ($128$ mel bins, $20$ Hz low frequency).
- **Identical Tokenization & Normalization:** Unicode NFKC normalization, standardized Thai character vocabulary.
- **Leakage Prevention:** Unlabeled SSL adaptation uses only the training partition; validation and test partitions remain unseen until checkpoint evaluation.
- **Model Selection Integrity:** The best checkpoint is chosen strictly using validation CER; the test set is evaluated exactly once after training concludes.
- **Reproducibility:** Seed setting across Python random, NumPy, PyTorch CPU & CUDA; Git commit hash and working tree status recorded for every run.

---

## 5. Expected Research Summary Table Format

| Model Variant | SSL Adaptation | Labeled Budget | Freeze Policy | Val CER (%) | Test CER (%) | Test WER (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Audio-JEPA Unmodified | None | 100% | Frozen | N/A (Failed) | N/A (Failed) | N/A (Failed) |
| Audio-JEPA + TempAdapter | None | 10% | Frozen | - | - | - |
| Audio-JEPA + TempAdapter | None | 25% | Frozen | - | - | - |
| Audio-JEPA + TempAdapter | None | 100% | Frozen | - | - | - |
| Audio-JEPA + TempAdapter | None | 100% | Full FT | - | - | - |
| Audio-JEPA + TempAdapter | **Thai Porjai SSL** | 10% | Full FT | - | - | - |
| Audio-JEPA + TempAdapter | **Thai Porjai SSL** | 25% | Full FT | - | - | - |
| Audio-JEPA + TempAdapter | **Thai Porjai SSL** | 50% | Full FT | - | - | - |
| Audio-JEPA + TempAdapter | **Thai Porjai SSL** | 100% | Full FT | - | - | - |
| Wav2Vec2 Baseline | None | 100% | Full FT | - | - | - |
