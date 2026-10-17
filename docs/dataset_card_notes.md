# Dataset Card Notes: Porjai Thai Voice Dataset (Central)

## 1. Upstream Dataset Details
- **Hugging Face Hub:** `CMKL/Porjai-Thai-voice-dataset-central`
- **Authors:** CMKL University (Wang Data Market)
- **Modality:** Audio and Transcripts
- **Language:** Thai (Central Dialect - `th`)
- **License:** CC BY-SA 4.0
- **Total Corpus Size:** ~700 hours (Central Thai), 335,674 examples.
- **Data File Format:** Parquet partitions containing embedded audio bytes and text strings.

## 2. Schema
| Column | Type | Description |
| :--- | :--- | :--- |
| `utterance` | string | Unique utterance identifier (e.g. `thai-central_000000`) |
| `sentence` | string | Orthographic Thai transcript (e.g. `ทีม จาก อิสราเอล ไม่ ควร ได้ เป็น เจ้าบ้าน...`) |
| `audio` | audio feature | Audio structure with embedded bytes / local path, sampling rate (typically 16–44.1 kHz) |

## 3. Known Limitations & Handling
1. **Official Split Absence:** The dataset ships on Hugging Face as a single monolithic `train` partition. We address this by computing a deterministic 80/10/10 split using MD5 hashing on `utterance` modulo 100.
2. **Speaker ID Absence:** The published dataset does not expose explicit speaker IDs in the metadata columns. Utterance hashing distributes utterances deterministically, but cross-speaker leakage cannot be mathematically eliminated without speaker clustering; this is documented as an experimental limitation.
3. **Audio Decoding:** Recent versions of Hugging Face `datasets` require system FFmpeg libraries or `torchcodec` for automatic decoding. To ensure 100% reproducibility across environments, our loader loads audio with `decode=False` and handles decompression via `PyAV` (`av`) or `soundfile` directly from memory buffers.
4. **Spacing Variations:** The raw transcripts occasionally contain space-separated word boundaries from crowdsourced transcribers. Our configurable normalizer standardizes whitespace and punctuation before character tokenization.
