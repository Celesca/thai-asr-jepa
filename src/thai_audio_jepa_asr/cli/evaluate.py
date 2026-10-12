"""CLI command: Evaluate a trained ASR checkpoint on validation or test sets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from thai_audio_jepa_asr.data.dataset import ThaiSpeechASRDataset
from thai_audio_jepa_asr.data.collator import ASRDataCollator
from thai_audio_jepa_asr.evaluation.evaluator import ASREvaluator
from thai_audio_jepa_asr.models.audio_jepa import AudioJEPAEncoder
from thai_audio_jepa_asr.models.ctc_model import AudioJEPACTCAudioModel
from thai_audio_jepa_asr.preprocessing.audio import AudioJEPAMelSpec
from thai_audio_jepa_asr.preprocessing.tokenizer import ThaiCharTokenizer
from thai_audio_jepa_asr.utils.checkpoint import load_checkpoint
from thai_audio_jepa_asr.utils.config import resolve_config
from thai_audio_jepa_asr.utils.logging import setup_logger

logger = setup_logger("evaluate")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Thai ASR model checkpoint.")
    parser.add_argument("--config", default="configs/experiment/evaluate.yaml", help="Evaluation configuration YAML.")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint .pt file.")
    parser.add_argument("--manifest", default=None, help="Direct path to manifest JSONL.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="Dataset split to evaluate.")
    parser.add_argument("--output-dir", default=None, help="Directory to save evaluation results.")
    args = parser.parse_args()

    config = resolve_config(args.config)
    eval_cfg = config.get("evaluation", config)
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})

    checkpoint_path = args.checkpoint or eval_cfg.get("checkpoint")
    if not checkpoint_path or not Path(checkpoint_path).is_file():
        logger.error("Valid checkpoint path is required! Provided: %s", checkpoint_path)
        return 1

    device = torch.device(eval_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir or eval_cfg.get("output_dir", "outputs/evaluation"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer
    vocab_path = data_cfg.get("vocab_path", "data_cache/porjai/vocab.json")
    tokenizer = ThaiCharTokenizer.load(vocab_path)

    # Model
    patch_size_cfg = model_cfg.get("patch_size", {"time": 1, "frequency": 16})
    if isinstance(patch_size_cfg, dict):
        patch_h = int(patch_size_cfg.get("time", 1))
        patch_w = int(patch_size_cfg.get("frequency", 16))
    else:
        patch_h = int(patch_size_cfg[0])
        patch_w = int(patch_size_cfg[1])

    encoder = AudioJEPAEncoder(patch_size=(patch_h, patch_w))
    model = AudioJEPACTCAudioModel(
        encoder=encoder,
        vocab_size=tokenizer.vocab_size,
        temporal_adapter_mode=model_cfg.get("temporal_adapter_mode", "native_pooling"),
        upsample_factor=model_cfg.get("upsample_factor", 1),
        pad_token_id=tokenizer.pad_token_id,
    )

    logger.info("Loading checkpoint from: %s", checkpoint_path)
    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    # Dataset & Dataloader
    if args.manifest:
        manifest_path = Path(args.manifest)
    else:
        manifest_dir = Path(data_cfg.get("manifest_dir", "data_cache/porjai/manifests"))
        manifest_path = manifest_dir / f"{args.split}.jsonl"

    if not manifest_path.is_file():
        logger.error("Manifest not found: %s", manifest_path)
        return 1

    frontend_name = data_cfg.get("frontend", "original")
    mel_transform = AudioJEPAMelSpec(
        frontend_type=frontend_name,
        sr=data_cfg.get("sample_rate", 32000),
        clip_length_s=data_cfg.get("clip_length_s", 10.0),
        target_time_bins=data_cfg.get("target_time_bins", None),
    )

    dataset = ThaiSpeechASRDataset(
        manifest_path=manifest_path,
        tokenizer=tokenizer,
        mel_spec_transform=mel_transform,
    )
    collator = ASRDataCollator(
        pad_token_id=tokenizer.pad_token_id,
        grid_w=128 // mel_transform.cfg.patch_freq,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=eval_cfg.get("batch_size", 4),
        shuffle=False,
        collate_fn=collator,
    )

    evaluator = ASREvaluator(model, tokenizer, device=device)
    logger.info("Running evaluation on %s (%d samples)...", manifest_path.name, len(dataset))
    results = evaluator.evaluate(dataloader, output_dir=output_dir, prefix=args.split)

    metrics = results["metrics"]
    predictions = results.get("predictions", [])

    print("=" * 70)
    print(f"EVALUATION RESULTS ({args.split.upper()} SPLIT)")
    print("=" * 70)
    print(f"CER:  {metrics['cer']*100:.2f}%")
    print(f"WER:  {metrics['wer']*100:.2f}%")
    print(f"Loss: {metrics.get('loss', 0.0):.4f}")
    print(f"Total Reference Characters: {metrics.get('total_ref_chars', 0)}")
    print(f"Total Reference Words:      {metrics.get('total_ref_words', 0)}")
    print(f"Results saved to:           {output_dir}")
    print("-" * 70)
    print("Sample Model Predictions:")
    for i, pred in enumerate(predictions[:3]):
        print(f"  [{i+1}] Ref:  {pred.get('reference', '')}")
        print(f"      Hyp:  {pred.get('hypothesis', '')}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    code = main()
    import os
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
