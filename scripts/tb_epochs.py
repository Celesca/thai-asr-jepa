"""Per-epoch summary of a run's TensorBoard scalars (SSL or ASR).

    uv run python scripts/tb_epochs.py outputs/runs/exp_C_ssl_pretrain/tensorboard

Epochs are split by the wall time of each `*/epoch_loss` event, so resumed runs (one event file per job) work.
"""

import statistics as st
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def summarize(logdir: str) -> None:
    for f in sorted(Path(logdir).glob("events.out.tfevents.*")):
        ea = EventAccumulator(str(f), size_guidance={"scalars": 0})
        ea.Reload()
        tags = ea.Tags()["scalars"]
        ep_tag = next((t for t in tags if t.endswith("epoch_loss")), None)
        if ep_tag is None:
            continue
        per_epoch = [t for t in tags if t.startswith("val/")]  # logged once per epoch, step = epoch index
        per_step = [t for t in tags if t != ep_tag and t not in per_epoch]
        print(f"\n===== {f.name}")
        start = float("-inf")
        for e in ea.Scalars(ep_tag):
            pts = [s for s in ea.Scalars(per_step[0]) if start < s.wall_time <= e.wall_time]
            hrs = (pts[-1].wall_time - pts[0].wall_time) / 3600 if pts else 0.0
            print(f"\nEpoch {e.step + 1}  {ep_tag}={e.value:.4f}  steps {pts[0].step if pts else '-'}-{pts[-1].step if pts else '-'}  {hrs:.2f}h")
            for t in per_epoch:
                v = [s.value for s in ea.Scalars(t) if s.step == e.step]
                if v:
                    print(f"  {t:<34} {v[0]:.4f}")
            for t in per_step:
                v = [s.value for s in ea.Scalars(t) if start < s.wall_time <= e.wall_time]
                if v:
                    q = max(1, len(v) // 10)  # first/last 10% of the epoch
                    print(f"  {t:<34} start {st.mean(v[:q]):.4g} -> end {st.mean(v[-q:]):.4g} | "
                          f"mean {st.mean(v):.4g} min {min(v):.4g} max {max(v):.4g}")
            start = e.wall_time
        lost = [s for s in ea.Scalars(per_step[0]) if s.wall_time > start]
        if lost:
            print(f"\n(unfinished epoch: {len(lost)} logged points, steps {lost[0].step}-{lost[-1].step}, not checkpointed)")


if __name__ == "__main__":
    summarize(sys.argv[1] if len(sys.argv) > 1 else "outputs/runs/exp_C_ssl_pretrain/tensorboard")
