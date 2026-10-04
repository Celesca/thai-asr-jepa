"""Self-Supervised Adaptation Trainer for Audio-JEPA on Thai speech."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from thai_audio_jepa_asr.models.jepa_ssl import AudioJEPASSLModule
from thai_audio_jepa_asr.utils.checkpoint import save_checkpoint, load_checkpoint
from thai_audio_jepa_asr.utils.git_info import get_git_info

logger = logging.getLogger(__name__)


class SSLTrainer:
    """Trainer for continued Self-Supervised Adaptation of Audio-JEPA on unlabeled speech."""

    def __init__(
        self,
        ssl_module: AudioJEPASSLModule,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[Any] = None,
        train_loader: DataLoader = None,
        output_dir: str | Path = "outputs/runs/ssl_adaptation",
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
        grad_clip_norm: float = 1.0,
        log_interval: int = 10,
        save_interval_epochs: int = 1,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.ssl_module = ssl_module
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        self.grad_clip_norm = grad_clip_norm
        self.log_interval = log_interval
        self.save_interval_epochs = save_interval_epochs
        self.config = config or {}

        tb_dir = self.output_dir / "tensorboard"
        self.writer = SummaryWriter(log_dir=str(tb_dir))
        self.ssl_module.to(self.device)
        self.git_info = get_git_info()

    def train(self, num_epochs: int, resume_from: Optional[str | Path] = None) -> Dict[str, Any]:
        """Execute SSL adaptation loop."""
        start_epoch = 0
        global_step = 0

        if resume_from:
            logger.info("Resuming SSL adaptation from checkpoint: %s", resume_from)
            ckpt = load_checkpoint(resume_from, map_location=self.device)
            self.ssl_module.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if self.lr_scheduler and "scheduler_state_dict" in ckpt:
                self.lr_scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            global_step = ckpt.get("global_step", 0)

        start_time = time.time()
        logger.info(
            "Starting Thai SSL Adaptation on %s: %d epochs, %d batches per epoch.",
            self.device,
            num_epochs,
            len(self.train_loader),
        )

        history = []

        for epoch in range(start_epoch, num_epochs):
            self.ssl_module.train()
            epoch_loss = 0.0
            epoch_batches = 0
            t0 = time.time()

            for step, batch in enumerate(self.train_loader):
                spectrograms = batch["spectrograms"].to(self.device)
                context_masks = [m.to(self.device) for m in batch["context_masks"]]
                prediction_masks = [m.to(self.device) for m in batch["prediction_masks"]]

                self.optimizer.zero_grad()
                out = self.ssl_module(spectrograms, context_masks, prediction_masks)
                loss = out["loss"]

                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning("Step %d: NaN or Inf SSL loss encountered! Skipping batch.", global_step)
                    continue

                loss.backward()

                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.ssl_module.parameters(), self.grad_clip_norm)

                self.optimizer.step()
                # Update target encoder via EMA
                self.ssl_module.update_target_encoder()

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                loss_val = loss.item()
                epoch_loss += loss_val
                epoch_batches += 1
                global_step += 1

                if global_step % self.log_interval == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    self.writer.add_scalar("ssl/loss", loss_val, global_step)
                    self.writer.add_scalar("ssl/lr", lr, global_step)
                    logger.info("SSL Epoch [%d/%d] Step [%d] Loss: %.4f LR: %.2e", epoch + 1, num_epochs, global_step, loss_val, lr)

            avg_loss = epoch_loss / max(1, epoch_batches)
            epoch_duration = time.time() - t0
            self.writer.add_scalar("ssl/epoch_loss", avg_loss, epoch)

            history.append({
                "epoch": epoch + 1,
                "loss": round(avg_loss, 4),
                "duration_sec": round(epoch_duration, 2),
            })

            # Save checkpoint (both full SSL state and adapted encoder alone for ASR fine-tuning)
            if (epoch + 1) % self.save_interval_epochs == 0 or (epoch + 1) == num_epochs:
                ckpt_dir = self.output_dir / "checkpoints"
                ckpt_state = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": self.ssl_module.state_dict(),
                    "encoder_state_dict": self.ssl_module.encoder.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "config": self.config,
                    "git_info": self.git_info,
                }
                save_checkpoint(ckpt_state, ckpt_dir, filename=f"ssl_epoch_{epoch+1}.pt")
                # Also save canonical adapted encoder weights for downstream ASR fine-tuning
                adapted_encoder_path = ckpt_dir / "adapted_encoder.pt"
                torch.save(self.ssl_module.encoder.state_dict(), adapted_encoder_path)
                logger.info("Saved adapted encoder weights to: %s", adapted_encoder_path)

        total_time = time.time() - start_time
        summary = {
            "total_epochs": epoch + 1,
            "total_duration_sec": round(total_time, 2),
            "final_loss": round(avg_loss, 4),
            "history": history,
            "git_info": self.git_info,
        }
        with open(self.output_dir / "ssl_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        self.writer.close()
        return summary
