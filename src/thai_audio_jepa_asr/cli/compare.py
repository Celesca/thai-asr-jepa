"""CLI command: Compare completed experimental runs across configurations and data budgets."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
from tabulate import tabulate

from thai_audio_jepa_asr.utils.logging import setup_logger

logger = setup_logger("compare")


def gather_run_summaries(runs_dir: str | Path) -> List[Dict[str, Any]]:
    """Scan runs_dir and extract training summaries and evaluation metrics."""
    runs_dir = Path(runs_dir)
    summaries = []

    for summary_file in runs_dir.rglob("training_summary.json"):
        run_folder = summary_file.parent
        run_name = run_folder.name
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Check if resolved config is present
            cfg_file = run_folder / "resolved_config.yaml"
            cfg: Dict[str, Any] = {}
            if cfg_file.is_file():
                import yaml
                with open(cfg_file, "r", encoding="utf-8") as yf:
                    cfg = yaml.safe_load(yf) or {}

            exp_cfg = cfg.get("experiment", {})
            model_cfg = cfg.get("model", {})
            data_cfg = cfg.get("data", {})

            test_metrics = data.get("test_metrics") or {}
            val_cer = data.get("best_val_cer", None)

            record = {
                "run_name": run_name,
                "experiment_id": exp_cfg.get("name", run_name),
                "seed": exp_cfg.get("seed", 42),
                "freeze_encoder": model_cfg.get("freeze_encoder", False),
                "temporal_mode": model_cfg.get("temporal_adapter_mode", "upsample_conv"),
                "budget_ratio": data_cfg.get("budget_ratio", 1.0),
                "val_cer": val_cer if val_cer is not None else float("nan"),
                "test_cer": test_metrics.get("cer", float("nan")),
                "test_wer": test_metrics.get("wer", float("nan")),
                "total_duration_sec": data.get("total_duration_sec", 0.0),
            }
            summaries.append(record)
        except Exception as e:
            logger.warning("Error reading %s: %s", summary_file, e)

    return summaries


def format_table(summaries: List[Dict[str, Any]]) -> str:
    """Format summaries into an aligned comparative table."""
    headers = [
        "Experiment", "Run Name", "Seed", "Budget", "Temporal Mode", "Frozen?", "Val CER (%)", "Test CER (%)", "Test WER (%)"
    ]
    rows = []
    for s in summaries:
        val_cer_str = f"{s['val_cer']*100:.2f}%" if not np.isnan(s['val_cer']) else "N/A"
        test_cer_str = f"{s['test_cer']*100:.2f}%" if not np.isnan(s['test_cer']) else "N/A"
        test_wer_str = f"{s['test_wer']*100:.2f}%" if not np.isnan(s['test_wer']) else "N/A"
        budget_str = f"{int(s['budget_ratio']*100)}%"

        rows.append([
            s["experiment_id"],
            s["run_name"],
            s["seed"],
            budget_str,
            s["temporal_mode"],
            "Yes" if s["freeze_encoder"] else "No",
            val_cer_str,
            test_cer_str,
            test_wer_str,
        ])
    return tabulate(rows, headers=headers, tablefmt="github")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare completed Thai Audio-JEPA ASR experiments.")
    parser.add_argument("--runs-dir", default="outputs/runs", help="Directory containing experiment runs.")
    parser.add_argument("--output-csv", default=None, help="Optional path to save comparison CSV.")
    parser.add_argument("--output-json", default=None, help="Optional path to save comparison JSON.")
    args = parser.parse_args()

    runs_path = Path(args.runs_dir)
    if not runs_path.is_dir():
        logger.warning("Runs directory not found: %s. No experiments to compare yet.", runs_path)
        print("No completed runs found.")
        return 0

    summaries = gather_run_summaries(runs_path)
    if not summaries:
        print(f"No completed runs with 'training_summary.json' found in: {runs_path}")
        return 0

    print("=" * 80)
    print("THAI AUDIO-JEPA ASR: EXPERIMENTAL COMPARISON MATRIX")
    print("=" * 80)
    table_str = format_table(summaries)
    print(table_str)
    print("=" * 80)

    # Save to disk
    csv_file = Path(args.output_csv or (runs_path / "comparison_matrix.csv"))
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Comparison CSV saved to: {csv_file}")

    json_file = Path(args.output_json or (runs_path / "comparison_matrix.json"))
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    print(f"Comparison JSON saved to: {json_file}")
    return 0


if __name__ == "__main__":
    code = main()
    import os
    os._exit(code)
