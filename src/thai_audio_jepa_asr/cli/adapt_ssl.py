"""CLI command: Continued Self-Supervised Adaptation of Audio-JEPA on Thai speech."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from thai_audio_jepa_asr.data.dataset import ThaiAudioSSLDataset
from thai_audio_jepa_asr.data.collator import SSLMaskCollator
from thai_audio_jepa_asr.models.audio_jepa import AudioJEPAEncoder
from thai_audio_jepa_asr.models.jepa_ssl import AudioJEPASSLModule
from thai_audio_jepa_asr.preprocessing.audio import AudioJEPAMelSpec
from thai_audio_jepa_asr.training.lr_scheduler import get_warmup_cosine_scheduler
from thai_audio_jepa_asr.training.ssl_trainer import SSLTrainer
from thai_audio_jepa_asr.utils.config import resolve_config, save_config
from thai_audio_jepa_asr.utils.logging import setup_logger
from thai_audio_jepa_asr.utils.seed import seed_everything

logger = setup_logger("adapt_ssl")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Thai-specific self-supervised adaptation of Audio-JEPA.")
    parser.add_argument("--config", default="configs/experiment/audio_jepa_thai_ssl.yaml", help="Path to SSL YAML config.")
    parser.add_argument("--resume", default=None, help="Resume checkpoint.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs.")
    args = parser.parse_args()

    config = resolve_config(args.config)
    exp_cfg = config.get("experiment", {})
    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})

    seed = exp_cfg.get("seed", 42)
    seed_everything(seed)

    output_dir = Path(exp_cfg.get("output_dir", "outputs/runs/audio_jepa_thai_ssl"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "resolved_config.yaml")

    # Datasets
    manifest_override = data_cfg.get("manifest_path")
    if manifest_override:
        ssl_manifest = Path(manifest_override)
    else:
        manifest_dir = Path(data_cfg.get("manifest_dir", "data_cache/porjai/manifests"))
        ssl_manifest = manifest_dir / "train.jsonl"

    if not ssl_manifest.is_file():
        logger.error("SSL train manifest not found: %s. Run prepare_data first!", ssl_manifest)
        return 1

    frontend_name = data_cfg.get("frontend", "original")
    mel_transform = AudioJEPAMelSpec(
        frontend_type=frontend_name,
        sr=data_cfg.get("sample_rate", 32000),
        clip_length_s=data_cfg.get("clip_length_s", 10.0),
        target_time_bins=data_cfg.get("target_time_bins", None),
    )
    logger.info(
        "Configured SSL frontend '%s': token_rate=%.2f fps, output_interval=%.2f ms, max_tokens=%d",
        mel_transform.cfg.name,
        mel_transform.cfg.temporal_token_rate,
        mel_transform.cfg.effective_output_interval_ms,
        mel_transform.cfg.compute_max_time_tokens(),
    )

    dataset = ThaiAudioSSLDataset(
        manifest_path=ssl_manifest,
        mel_spec_transform=mel_transform,
        max_duration_s=float(data_cfg.get("max_duration_s", 10.0)),
    )

    patch_size_cfg = model_cfg.get("patch_size", None)
    if patch_size_cfg is not None:
        if isinstance(patch_size_cfg, dict):
            patch_h = int(patch_size_cfg.get("time", mel_transform.cfg.patch_time))
            patch_w = int(patch_size_cfg.get("frequency", mel_transform.cfg.patch_freq))
        else:
            patch_h = int(patch_size_cfg[0])
            patch_w = int(patch_size_cfg[1])
    else:
        patch_h = mel_transform.cfg.patch_time
        patch_w = mel_transform.cfg.patch_freq

    grid_h = mel_transform.target_time_bins // patch_h
    grid_w = 128 // patch_w
    masking_mode = model_cfg.get("masking_mode", "short_span")

    collator = SSLMaskCollator(
        grid_size=(grid_h, grid_w),
        masking_mode=masking_mode,
        span_duration_ms=float(model_cfg.get("span_duration_ms", 300.0)),
        token_interval_ms=mel_transform.cfg.effective_output_interval_ms,
        mask_ratio=float(model_cfg.get("mask_ratio", 0.50)),
        enc_mask_scale=tuple(model_cfg.get("enc_mask_scale", (0.2, 0.7))),
        pred_mask_scale=tuple(model_cfg.get("pred_mask_scale", (0.1, 0.25))),
    )

    batch_size = train_cfg.get("batch_size", 4)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=train_cfg.get("num_workers", 2),
    )

    # Initialize encoder using transfer learning or scratch
    ckpt_source = model_cfg.get("pretrained_checkpoint", "ltuncay/Audio-JEPA")
    if ckpt_source in ("scratch", "none", None):
        logger.info("Initializing SSL encoder from scratch (random initialization) with patch_size=%s", (patch_h, patch_w))
        encoder = AudioJEPAEncoder(patch_size=(patch_h, patch_w), input_size=(mel_transform.target_time_bins, 128))
        report = None
    else:
        encoder, report = AudioJEPAEncoder.from_pretrained_transfer(
            model_name_or_path=ckpt_source,
            patch_size=(patch_h, patch_w),
            input_size=(mel_transform.target_time_bins, 128),
        )
        report.save(output_dir / "checkpoint_loading_report.json")
        logger.info("Pretrained checkpoint transferred: %.2f%% weights initialized.", report.loaded_percentage)

    ssl_module = AudioJEPASSLModule(
        encoder=encoder,
        predictor_depth=model_cfg.get("predictor_depth", 6),
        predictor_embed_dim=model_cfg.get("predictor_embed_dim", 384),
        ema_tau=float(model_cfg.get("ema_tau", 0.996)),
        ema_tau_end=float(model_cfg.get("ema_tau_end", 1.0)),
        use_ema_schedule=bool(model_cfg.get("use_ema_schedule", False)),
        norm_pix_loss=model_cfg.get("norm_pix_loss", True),
    )

    # Optimizer
    lr = float(train_cfg.get("lr", 1e-4))
    optimizer = torch.optim.AdamW(
        ssl_module.parameters(),
        lr=lr,
        weight_decay=float(train_cfg.get("weight_decay", 1e-2)),
    )

    epochs = args.epochs or train_cfg.get("epochs", 5)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(train_cfg.get("warmup_ratio", 0.1) * total_steps)
    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)

    trainer = SSLTrainer(
        ssl_module=ssl_module,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        train_loader=train_loader,
        output_dir=output_dir,
        device=train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        grad_clip_norm=train_cfg.get("grad_clip_norm", 1.0),
        log_interval=train_cfg.get("log_interval", 10),
        config=config,
    )

    summary = trainer.train(num_epochs=epochs, resume_from=args.resume)
    logger.info("Thai SSL Adaptation completed successfully. Final loss: %s", summary.get("final_loss"))
    return 0


if __name__ == "__main__":
    code = main()
    import os
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
