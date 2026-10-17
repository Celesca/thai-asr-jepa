"""CLI command: Predict Thai transcript for a single WAV or MP3 audio file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import torch

from thai_audio_jepa_asr.models.audio_jepa import AudioJEPAEncoder
from thai_audio_jepa_asr.models.ctc_model import AudioJEPACTCAudioModel
from thai_audio_jepa_asr.preprocessing.audio import AudioJEPAMelSpec, load_audio
from thai_audio_jepa_asr.preprocessing.tokenizer import ThaiCharTokenizer
from thai_audio_jepa_asr.utils.checkpoint import load_checkpoint
from thai_audio_jepa_asr.utils.logging import setup_logger

logger = setup_logger("predict")


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict Thai transcription for a single audio file.")
    parser.add_argument("--checkpoint", required=True, help="Path to trained ASR checkpoint .pt file.")
    parser.add_argument("--audio", required=True, help="Path to input audio file (WAV, MP3, etc.).")
    parser.add_argument("--vocab", default="data_cache/porjai/vocab.json", help="Path to tokenizer vocab.json.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Inference device.")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.is_file():
        logger.error("Audio file does not exist: %s", audio_path)
        return 1

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        logger.error("Checkpoint file does not exist: %s", checkpoint_path)
        return 1

    vocab_path = Path(args.vocab)
    if vocab_path.is_file():
        tokenizer = ThaiCharTokenizer.load(vocab_path)
    else:
        tokenizer = ThaiCharTokenizer()

    device = torch.device(args.device)

    # Load audio and extract Mel-Spectrogram
    waveform, sr = load_audio(audio_path, target_sr=32000)
    mel_extractor = AudioJEPAMelSpec(sr=32000)
    spectrogram = mel_extractor(waveform).unsqueeze(0).to(device)  # [1, 1, 256, 128]

    # Initialize model
    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    config = ckpt.get("config", {})
    model_cfg = config.get("model", {})

    encoder = AudioJEPAEncoder()
    model = AudioJEPACTCAudioModel(
        encoder=encoder,
        vocab_size=tokenizer.vocab_size,
        temporal_adapter_mode=model_cfg.get("temporal_adapter_mode", "upsample_conv"),
        upsample_factor=model_cfg.get("upsample_factor", 4),
        pad_token_id=tokenizer.pad_token_id,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    with torch.no_grad():
        out = model(spectrogram)
        logits = out["logits"]  # [1, T, V]
        best_tokens = logits.argmax(dim=-1)[0]  # [T]
        predicted_text = tokenizer.decode_ctc(best_tokens)

    print("=" * 60)
    print("THAI ASR PREDICTION RESULT")
    print("=" * 60)
    print(f"Audio File:   {audio_path}")
    print(f"Duration:     {waveform.shape[1] / 32000.0:.2f} seconds")
    print(f"Checkpoint:   {checkpoint_path}")
    print(f"Prediction:   {predicted_text}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    code = main()
    import os
    os._exit(code)
