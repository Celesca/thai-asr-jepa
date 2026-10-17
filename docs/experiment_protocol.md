# Experiment Protocol: Thai Audio-JEPA ASR

## 1. Train / Validation / Test Protocol
1. **Partitioning:**
   - 80% Train, 10% Validation, 10% Test.
   - Deterministic assignment via MD5 hash of utterance ID modulo 100.
2. **Leakage Protection:**
   - SSL adaptation is strictly performed on audio from the `train` partition.
   - Validation and test transcripts are never seen during SSL adaptation.
   - Checkpoints are saved based on minimum Validation Character Error Rate (CER).
   - Test evaluation is run strictly once on the selected best checkpoint.

## 2. Evaluation Metrics Protocol
- **Primary Metric: Character Error Rate (CER)**
  $$\text{CER} = \frac{S_c + D_c + I_c}{N_c}$$
  where $S_c, D_c, I_c$ are character-level substitutions, deletions, and insertions, and $N_c$ is total reference characters.
- **Secondary Metric: Thai Word Error Rate (WER)**
  - Thai text has no inter-word whitespace.
  - Standard whitespace tokenization yields sentence-level error rates.
  - Protocol: Segment both reference and hypothesis strings using **PyThaiNLP `newmm` dictionary engine** before edit-distance calculation.

## 3. Seed Policy & Error Reporting
- All experiments pin seeds (`--seed 42` default).
- For final published research tables, experiments must be executed across 3 seeds (`42`, `43`, `44`).
- Results are reported as $\text{mean} \pm \text{std}$. Single-seed runs are explicitly labeled as **Preliminary**.

## 4. CTC Length Feasibility Check
Before executing any training run, the model validates that:
$$T_{\text{model}}(x) \ge L_{\text{target}}(y)$$
for all $(x, y)$ in the batch.
- In **unmodified baseline mode** (`temporal_mode: unmodified`), the model logs all sequence violations and skips invalid instances to measure exact baseline capability.
- In **temporal adaptation mode** (`temporal_mode: upsample_conv`), a 1D convolutional interpolation layer ($4\times$) expands the temporal resolution from $16 \rightarrow 64$ frames (6.4 fps), ensuring $100\%$ CTC feasibility.
