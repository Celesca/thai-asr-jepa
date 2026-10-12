#!/usr/bin/env python3
import sys
sys.path.insert(0, ".venv/lib/python3.12/site-packages")
from dulwich.repo import Repo
from dulwich import porcelain

repo = Repo(".")
branch_ref = b"refs/heads/feat/audio-jepa-thai-asr-temporal"
files = [
    "README.md",
    "docs/architecture_change.md",
    "configs/experiment/smoke_ssl.yaml",
    "configs/experiment/smoke.yaml",
    "configs/experiment/evaluate.yaml",
    "src/thai_audio_jepa_asr/cli/evaluate.py",
    "src/thai_audio_jepa_asr/cli/adapt_ssl.py",
    "src/thai_audio_jepa_asr/cli/train_asr.py",
    "src/thai_audio_jepa_asr/models/temporal_adapter.py",
    "tests/test_checkpoint_transfer.py",
]
porcelain.add(repo, files)
commit = porcelain.commit(
    repo,
    message="docs & test: add architecture change docs, smoke tests, and full test suite",
    author="Researcher <researcher@example.com>",
    committer="Researcher <researcher@example.com>",
)
repo.refs[branch_ref] = commit.encode() if isinstance(commit, str) else commit
print("Committed to feat branch:", commit.decode() if isinstance(commit, bytes) else commit)
print("Current HEAD:", repo.refs.read_ref(b"HEAD").decode())
