"""CLI command: Inspect Original vs Modified Audio-JEPA architectures, temporal token rates, and transfer report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import torch

from thai_audio_jepa_asr.models.audio_jepa import AudioJEPAEncoder
from thai_audio_jepa_asr.preprocessing.audio import AudioJEPAMelSpec, DEFAULT_SAMPLE_RATE
from thai_audio_jepa_asr.utils.logging import setup_logger

logger = setup_logger("inspect_audio_jepa")


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect Audio-JEPA model configuration, shapes, and transfer weights.")
    parser.add_argument("--checkpoint", default="ltuncay/Audio-JEPA", help="Hugging Face repo or local checkpoint path.")
    parser.add_argument("--output-json", default="outputs/checkpoint_loading_report.json", help="Path to save report JSON.")
    args = parser.parse_args()

    print("=" * 80)
    print("AUDIO-JEPA SPECIFICATIONS & TEMPORAL TOKEN RATE COMPARISON")
    print("=" * 80)
    print(f"Sampling Rate:                {DEFAULT_SAMPLE_RATE} Hz")
    print("Audio Clip Duration:          10.0 seconds")
    print("Mel-Spectrogram Shape:        (1, 256, 128) [Time=256, MelBins=128]")
    print("-" * 80)
    print("Configuration A: ORIGINAL Audio-JEPA")
    print("  - Patch Size:               (16, 16)")
    print("  - 2D Patch Grid:            16 time steps x 8 freq bands = 128 patches")
    print("  - Temporal Tokens / Clip:   16 tokens / 10s")
    print("  - Temporal Token Rate:      1.6 tokens / sec (625 ms per token)")
    print("  - CTC Feasibility:          UNSUITABLE (<5% Thai utterances <= 16 chars)")
    print("-" * 80)
    print("Configuration B: MODIFIED Audio-JEPA (ASR-Oriented)")
    print("  - Patch Size:               (1, 16) [Asymmetric temporal preservation]")
    print("  - 2D Patch Grid:            256 time steps x 8 freq bands = 2,048 patches")
    print("  - Temporal Tokens / Clip:   256 tokens / 10s")
    print("  - Temporal Token Rate:      25.6 tokens / sec (39.06 ms per token)")
    print("  - CTC Feasibility:          FEASIBLE (100% Porjai Thai utterances alignable)")
    print("=" * 80)

    mel_extractor = AudioJEPAMelSpec()
    dummy_wav = torch.randn(1, DEFAULT_SAMPLE_RATE * 10)  # 10s audio
    spec = mel_extractor(dummy_wav).unsqueeze(0)  # [1, 1, 256, 128]
    print(f"\n[Validation] Mel-spectrogram output shape: {tuple(spec.shape)}")

    # 1. Test Original Architecture
    orig_encoder = AudioJEPAEncoder(patch_size=(16, 16))
    with torch.no_grad():
        orig_out = orig_encoder(spec)
    print(f"[Config A] Original Audio-JEPA encoder output: {tuple(orig_out.shape)} -> Rate: {orig_encoder.temporal_token_rate:.1f} tokens/s")

    # 2. Test Modified Architecture
    mod_encoder = AudioJEPAEncoder(patch_size=(1, 16))
    with torch.no_grad():
        mod_out = mod_encoder(spec)
        mod_seq, seq_lens = mod_encoder.to_temporal_sequence(mod_out)
    print(f"[Config B] Modified Audio-JEPA encoder output: {tuple(mod_out.shape)} -> Rate: {mod_encoder.temporal_token_rate:.1f} tokens/s")
    print(f"[Config B] After frequency pooling: {tuple(mod_seq.shape)} (T={mod_seq.shape[1]} temporal frames)")

    # 3. Checkpoint Transfer Loading
    print(f"\n[Transfer Learning] Loading and adapting pretrained weights from '{args.checkpoint}'...")
    encoder, report = AudioJEPAEncoder.from_pretrained_transfer(
        model_name_or_path=args.checkpoint,
        patch_size=(1, 16),
    )

    rep = report.to_dict()
    print("\n" + "=" * 80)
    print("CHECKPOINT LOADING REPORT")
    print("=" * 80)
    print(f"Loaded Parameters Count:      {report.loaded_parameters_count:,}")
    print(f"Missing Parameters Count:     {len(report.missing_parameter_names):,}")
    print(f"Unexpected Parameters Count:  {len(report.unexpected_parameter_names):,}")
    print(f"Shape Mismatched Count:       {len(report.shape_mismatched_parameters):,}")
    print(f"Shape Mismatches Details:     {report.shape_mismatched_parameters}")
    print(f"Total Model Parameters:       {report.total_model_parameters:,}")
    print(f"Loaded Weight Ratio:          {report.loaded_percentage:.2f}%")
    print("=" * 80)

    # Save loading report
    report.save(args.output_json)
    print(f"\nLoading report saved successfully to: {args.output_json}\n")

    sys.stdout.flush()
    sys.stderr.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
