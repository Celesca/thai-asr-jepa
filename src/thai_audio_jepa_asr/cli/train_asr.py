"""CLI command: Train Supervised Thai Audio-JEPA + CTC ASR."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from thai_audio_jepa_asr.data.dataset import ThaiSpeechASRDataset
from thai_audio_jepa_asr.data.collator import ASRDataCollator
from thai_audio_jepa_asr.models.audio_jepa import AudioJEPAEncoder
from thai_audio_jepa_asr.models.ctc_model import AudioJEPACTCAudioModel
from thai_audio_jepa_asr.preprocessing.audio import AudioJEPAMelSpec
from thai_audio_jepa_asr.preprocessing.tokenizer import ThaiCharTokenizer
from thai_audio_jepa_asr.training.optimizer import build_asr_optimizer
from thai_audio_jepa_asr.training.lr_scheduler import get_warmup_cosine_scheduler
from thai_audio_jepa_asr.training.trainer import ASRTrainer
from thai_audio_jepa_asr.utils.config import resolve_config, save_config
from thai_audio_jepa_asr.utils.logging import setup_logger
from thai_audio_jepa_asr.utils.seed import seed_everything

logger = setup_logger("train_asr")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Supervised Thai ASR with Audio-JEPA and CTC.")
    parser.add_argument("--config", default="configs/experiment/audio_jepa_ctc.yaml", help="Path to experiment YAML.")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume training from.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    args = parser.parse_args()

    config = resolve_config(args.config)
    exp_cfg = config.get("experiment", {})
    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})

    seed = args.seed or exp_cfg.get("seed", 42)
    seed_everything(seed)

    output_dir = Path(exp_cfg.get("output_dir", "outputs/runs/audio_jepa_ctc"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "resolved_config.yaml")

    # Load tokenizer
    vocab_path = data_cfg.get("vocab_path", "data_cache/porjai/vocab.json")
    if Path(vocab_path).is_file():
        tokenizer = ThaiCharTokenizer.load(vocab_path)
    else:
        logger.info("Vocabulary not found at %s; generating canonical Thai tokenizer.", vocab_path)
        tokenizer = ThaiCharTokenizer()
        tokenizer.save(vocab_path)

    # Initialize Mel-Spectrogram transform
    mel_transform = AudioJEPAMelSpec(
        sr=data_cfg.get("sample_rate", 32000),
        clip_length_s=data_cfg.get("clip_length_s", 10.0),
        target_time_bins=data_cfg.get("target_time_bins", 256),
    )

    # Datasets
    manifest_dir = Path(data_cfg.get("manifest_dir", "data_cache/porjai/manifests"))
    train_manifest = Path(data_cfg.get("train_manifest", manifest_dir / "train.jsonl"))
    val_manifest = Path(data_cfg.get("val_manifest", manifest_dir / "val.jsonl"))
    test_manifest = Path(data_cfg.get("test_manifest", manifest_dir / "test.jsonl"))

    if not train_manifest.is_file():
        logger.error("Train manifest not found: %s. Run prepare_data first!", train_manifest)
        return 1

    budget_ratio = data_cfg.get("budget_ratio", 1.0)
    train_dataset = ThaiSpeechASRDataset(
        train_manifest,
        tokenizer=tokenizer,
        mel_spec_transform=mel_transform,
        budget_ratio=budget_ratio,
        subsample_seed=seed,
    )
    val_dataset = ThaiSpeechASRDataset(
        val_manifest,
        tokenizer=tokenizer,
        mel_spec_transform=mel_transform,
    ) if val_manifest.is_file() else None

    test_dataset = ThaiSpeechASRDataset(
        test_manifest,
        tokenizer=tokenizer,
        mel_spec_transform=mel_transform,
    ) if test_manifest.is_file() else None

    batch_size = args.batch_size or train_cfg.get("batch_size", 4)
    collator = ASRDataCollator(pad_token_id=tokenizer.pad_token_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=train_cfg.get("num_workers", 2),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    ) if val_dataset else None

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    ) if test_dataset else None

    # Build model with asymmetric or baseline patch size
    patch_size_cfg = model_cfg.get("patch_size", {"time": 1, "frequency": 16})
    if isinstance(patch_size_cfg, dict):
        patch_h = int(patch_size_cfg.get("time", 1))
        patch_w = int(patch_size_cfg.get("frequency", 16))
    else:
        patch_h = int(patch_size_cfg[0])
        patch_w = int(patch_size_cfg[1])

    ckpt_source = model_cfg.get("pretrained_checkpoint", "ltuncay/Audio-JEPA")
    adapted_weights = model_cfg.get("adapted_checkpoint", None)

    if adapted_weights and Path(adapted_weights).is_file():
        logger.info("Initializing encoder from Thai SSL adapted weights: %s", adapted_weights)
        encoder = AudioJEPAEncoder(patch_size=(patch_h, patch_w))
        encoder_sd = torch.load(adapted_weights, map_location="cpu", weights_only=False)
        encoder.load_state_dict(encoder_sd, strict=False)
        report = None
    else:
        logger.info("Initializing encoder via transfer learning from %s (patch_size=%s)", ckpt_source, (patch_h, patch_w))
        encoder, report = AudioJEPAEncoder.from_pretrained_transfer(
            model_name_or_path=ckpt_source,
            patch_size=(patch_h, patch_w),
        )
        report.save(output_dir / "checkpoint_loading_report.json")

    model = AudioJEPACTCAudioModel(
        encoder=encoder,
        vocab_size=tokenizer.vocab_size,
        temporal_adapter_mode=model_cfg.get("temporal_adapter_mode", "native_pooling"),
        upsample_factor=model_cfg.get("upsample_factor", 1),
        freeze_encoder=model_cfg.get("freeze_encoder", False),
        unfreeze_last_n_blocks=model_cfg.get("unfreeze_last_n_blocks", None),
        head_dropout=model_cfg.get("head_dropout", 0.1),
        pad_token_id=tokenizer.pad_token_id,
        loading_report=report.to_dict() if report else None,
    )

    params_info = model.count_parameters()
    logger.info("Model parameters: %s", params_info)

    # Optimizer & Scheduler
    epochs = args.epochs or train_cfg.get("epochs", 15)
    optimizer = build_asr_optimizer(
        model,
        encoder_lr=float(train_cfg.get("encoder_lr", 1e-4)),
        head_lr=float(train_cfg.get("head_lr", 5e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-2)),
    )
    total_steps = len(train_loader) * epochs
    warmup_steps = int(train_cfg.get("warmup_ratio", 0.1) * total_steps)
    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)

    # Trainer
    trainer = ASRTrainer(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        output_dir=output_dir,
        device=train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        grad_clip_norm=train_cfg.get("grad_clip_norm", 1.0),
        log_interval=train_cfg.get("log_interval", 10),
        eval_interval=train_cfg.get("eval_interval", 1),
        early_stopping_patience=train_cfg.get("early_stopping_patience", 10),
        config=config,
    )

    summary = trainer.train(num_epochs=epochs, resume_from=args.resume)
    logger.info("ASR training completed successfully. Best val CER: %s", summary.get("best_val_cer"))
    return 0


if __name__ == "__main__":
    code = main()
    import sys, os
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
