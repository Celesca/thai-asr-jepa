"""Evaluator runner for CTC speech recognition models."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
from torch.utils.data import DataLoader

from thai_audio_jepa_asr.evaluation.metrics import compute_asr_metrics
from thai_audio_jepa_asr.preprocessing.tokenizer import ThaiCharTokenizer

logger = logging.getLogger(__name__)


class ASREvaluator:
    """Evaluates an ASR model on a dataset, saving structured metrics and prediction logs."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: ThaiCharTokenizer,
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        output_dir: Optional[str | Path] = None,
        prefix: str = "eval",
    ) -> Dict[str, Any]:
        """Run complete evaluation over dataloader.

        Args:
            dataloader: DataLoader providing 'spectrograms', 'targets', 'texts', 'ids'.
            output_dir: Optional directory to save predictions.jsonl, metrics.json, and metrics.csv.
            prefix: Prefix for output files (e.g. 'val' or 'test').

        Returns:
            Dictionary containing aggregate metrics and sample predictions.
        """
        self.model.eval()
        self.model.to(self.device)

        all_ids: List[str] = []
        all_refs: List[str] = []
        all_preds: List[str] = []
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            spectrograms = batch["spectrograms"].to(self.device)
            targets = batch.get("targets")
            target_lengths = batch.get("target_lengths")
            if targets is not None:
                targets = targets.to(self.device)
            if target_lengths is not None:
                target_lengths = target_lengths.to(self.device)

            out = self.model(spectrograms, targets=targets, target_lengths=target_lengths)

            if "loss" in out and out["loss"] is not None:
                total_loss += out["loss"].item()
                num_batches += 1

            # Decode greedy predictions
            # logits: [B, T, V]
            best_tokens = out["logits"].argmax(dim=-1)  # [B, T]
            for i in range(len(best_tokens)):
                pred_text = self.tokenizer.decode_ctc(best_tokens[i])
                all_preds.append(pred_text)

            all_refs.extend(batch["texts"])
            all_ids.extend(batch["ids"])

        # Compute CER and WER
        metrics = compute_asr_metrics(all_refs, all_preds)
        if num_batches > 0:
            metrics["loss"] = round(total_loss / num_batches, 4)
        else:
            metrics["loss"] = 0.0

        # Build per-sample error details
        per_sample: List[Dict[str, Any]] = []
        for uid, ref, pred in zip(all_ids, all_refs, all_preds):
            sample_cer = compute_asr_metrics([ref], [pred])["cer"]
            per_sample.append({
                "id": uid,
                "reference": ref,
                "prediction": pred,
                "cer": sample_cer,
            })

        logger.info(
            "[%s] Loss: %.4f | CER: %.2f%% | WER: %.2f%%",
            prefix,
            metrics["loss"],
            metrics["cer"] * 100.0,
            metrics["wer"] * 100.0,
        )

        # Save files if output_dir provided
        if output_dir is not None:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)

            # JSON summary
            summary_path = out_path / f"{prefix}_metrics.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)

            # CSV summary
            csv_path = out_path / f"{prefix}_metrics.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
                writer.writeheader()
                writer.writerow(metrics)

            # Predictions JSONL
            pred_path = out_path / f"{prefix}_predictions.jsonl"
            with open(pred_path, "w", encoding="utf-8") as f:
                for item in per_sample:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            # Qualitative examples (sorted by CER)
            sorted_samples = sorted(per_sample, key=lambda x: x["cer"])
            qualitative = {
                "best_examples": sorted_samples[:5],
                "worst_examples": sorted_samples[-5:],
            }
            qual_path = out_path / f"{prefix}_qualitative_examples.json"
            with open(qual_path, "w", encoding="utf-8") as f:
                json.dump(qualitative, f, ensure_ascii=False, indent=2)

        return {
            "metrics": metrics,
            "sample_count": len(all_refs),
            "predictions": per_sample,
        }
