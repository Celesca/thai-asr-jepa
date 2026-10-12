"""Supervised ASR training pipeline for Thai Audio-JEPA + CTC."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from thai_audio_jepa_asr.evaluation.evaluator import ASREvaluator
from thai_audio_jepa_asr.preprocessing.tokenizer import ThaiCharTokenizer
from thai_audio_jepa_asr.utils.checkpoint import save_checkpoint, load_checkpoint
from thai_audio_jepa_asr.utils.git_info import get_git_info

logger = logging.getLogger(__name__)


class ASRTrainer:
    """Trainer for Supervised CTC Speech Recognition."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: ThaiCharTokenizer,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[Any] = None,
        train_loader: DataLoader = None,
        val_loader: DataLoader = None,
        test_loader: Optional[DataLoader] = None,
        output_dir: str | Path = "outputs/runs/default",
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
        grad_clip_norm: float = 1.0,
        log_interval: int = 10,
        eval_interval: int = 1,
        early_stopping_patience: int = 10,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        self.grad_clip_norm = grad_clip_norm
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.early_stopping_patience = early_stopping_patience
        self.config = config or {}

        self.evaluator = ASREvaluator(self.model, self.tokenizer, device=self.device)

        # TensorBoard writer
        tb_dir = self.output_dir / "tensorboard"
        self.writer = SummaryWriter(log_dir=str(tb_dir))

        self.model.to(self.device)
        self.git_info = get_git_info()

    def train(self, num_epochs: int, resume_from: Optional[str | Path] = None) -> Dict[str, Any]:
        """Execute full training loop with validation checkpoint selection."""
        start_epoch = 0
        global_step = 0
        best_val_cer = float("inf")
        patience_counter = 0

        # Resume if requested
        if resume_from:
            logger.info("Resuming training from checkpoint: %s", resume_from)
            ckpt = load_checkpoint(resume_from, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if self.lr_scheduler and "scheduler_state_dict" in ckpt:
                self.lr_scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            global_step = ckpt.get("global_step", 0)
            best_val_cer = ckpt.get("val_cer", float("inf"))

        training_history = []
        start_time = time.time()

        logger.info(
            "Starting ASR training on %s: %d epochs, %d batches per epoch.",
            self.device,
            num_epochs,
            len(self.train_loader),
        )

        for epoch in range(start_epoch, num_epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_batches = 0
            t0 = time.time()

            for step, batch in enumerate(self.train_loader):
                spectrograms = batch["spectrograms"].to(self.device)
                targets = batch["targets"].to(self.device)
                target_lengths = batch["target_lengths"].to(self.device)
                valid_time_tokens = batch.get("valid_time_tokens")
                if valid_time_tokens is not None:
                    valid_time_tokens = valid_time_tokens.to(self.device)
                padding_mask = batch.get("padding_mask")
                if padding_mask is not None:
                    padding_mask = padding_mask.to(self.device)

                self.optimizer.zero_grad()
                out = self.model(
                    spectrograms,
                    targets=targets,
                    target_lengths=target_lengths,
                    valid_time_tokens=valid_time_tokens,
                    padding_mask=padding_mask,
                )
                loss = out["loss"]

                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning("Step %d: NaN or Inf loss encountered! Skipping batch.", global_step)
                    continue

                loss.backward()

                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                loss_val = loss.item()
                epoch_loss += loss_val
                epoch_batches += 1
                global_step += 1

                if global_step % self.log_interval == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    self.writer.add_scalar("train/loss", loss_val, global_step)
                    self.writer.add_scalar("train/lr", lr, global_step)
                    logger.info(
                        "Epoch [%d/%d] Step [%d] Loss: %.4f LR: %.2e",
                        epoch + 1,
                        num_epochs,
                        global_step,
                        loss_val,
                        lr,
                    )

            avg_train_loss = epoch_loss / max(1, epoch_batches)
            epoch_duration = time.time() - t0
            self.writer.add_scalar("train/epoch_loss", avg_train_loss, epoch)

            # Validation evaluation
            val_results = None
            if self.val_loader is not None and ((epoch + 1) % self.eval_interval == 0 or (epoch + 1) == num_epochs):
                logger.info("Evaluating on validation partition...")
                val_eval = self.evaluator.evaluate(
                    self.val_loader,
                    output_dir=self.output_dir / "val_eval",
                    prefix=f"epoch_{epoch+1}",
                )
                val_metrics = val_eval["metrics"]
                val_cer = val_metrics["cer"]
                val_wer = val_metrics["wer"]
                val_loss = val_metrics.get("loss", 0.0)

                self.writer.add_scalar("val/cer", val_cer, epoch)
                self.writer.add_scalar("val/wer", val_wer, epoch)
                self.writer.add_scalar("val/loss", val_loss, epoch)

                is_best = val_cer < best_val_cer
                if is_best:
                    best_val_cer = val_cer
                    patience_counter = 0
                    logger.info("New best validation CER: %.2f%%! Saving best checkpoint.", best_val_cer * 100.0)
                else:
                    patience_counter += 1

                # Save checkpoint
                ckpt_state = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                    "val_cer": val_cer,
                    "val_wer": val_wer,
                    "best_val_cer": best_val_cer,
                    "config": self.config,
                    "git_info": self.git_info,
                }
                save_checkpoint(ckpt_state, self.output_dir / "checkpoints", is_best=is_best)

                val_results = val_metrics

            epoch_record = {
                "epoch": epoch + 1,
                "train_loss": round(avg_train_loss, 4),
                "duration_sec": round(epoch_duration, 2),
                "val_metrics": val_results,
            }
            training_history.append(epoch_record)

            # Early stopping check
            if patience_counter >= self.early_stopping_patience:
                logger.info("Early stopping triggered after %d epochs without improvement.", self.early_stopping_patience)
                break

        total_training_time = time.time() - start_time
        logger.info("Training complete in %.2f seconds.", total_training_time)

        # Final evaluation on test partition using best checkpoint (strictly avoiding model selection leakage)
        test_metrics = None
        best_ckpt_path = self.output_dir / "checkpoints" / "best_checkpoint.pt"
        if best_ckpt_path.is_file() and self.test_loader is not None:
            logger.info("Loading best checkpoint for unbiased final test evaluation: %s", best_ckpt_path)
            ckpt = load_checkpoint(best_ckpt_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            test_eval = self.evaluator.evaluate(
                self.test_loader,
                output_dir=self.output_dir / "test_eval",
                prefix="test",
            )
            test_metrics = test_eval["metrics"]
            logger.info("Final Test Results: CER: %.2f%% | WER: %.2f%%", test_metrics["cer"] * 100.0, test_metrics["wer"] * 100.0)

        # Save summary report
        summary = {
            "total_epochs": epoch + 1,
            "total_duration_sec": round(total_training_time, 2),
            "best_val_cer": best_val_cer,
            "test_metrics": test_metrics,
            "history": training_history,
            "git_info": self.git_info,
            "device": str(self.device),
        }
        with open(self.output_dir / "training_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        self.writer.close()
        return summary
