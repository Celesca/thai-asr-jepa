#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Add site-packages
sys.path.insert(0, ".venv/lib/python3.12/site-packages")
from dulwich.repo import Repo
from dulwich import porcelain

print("1. Opening or initializing Repo...", flush=True)
if Path(".git").exists():
    repo = Repo(".")
else:
    repo = Repo.init(".")

print("2. Staging initial base files...", flush=True)
base_files = [".gitignore", "pyproject.toml", "README.md"]
porcelain.add(repo, base_files)

print("3. Committing initial project state on master...", flush=True)
commit1 = porcelain.commit(
    repo,
    message="Initial commit: Audio-JEPA base project",
    author="Researcher <researcher@example.com>",
    committer="Researcher <researcher@example.com>",
)
base_sha = commit1.decode() if isinstance(commit1, bytes) else commit1
print("Base commit SHA:", base_sha, flush=True)

# Ensure refs/heads/master
repo.refs[b"refs/heads/master"] = commit1.encode() if isinstance(commit1, str) else commit1

print("4. Creating branch feat/audio-jepa-thai-asr-temporal...", flush=True)
branch_ref = b"refs/heads/feat/audio-jepa-thai-asr-temporal"
repo.refs[branch_ref] = commit1.encode() if isinstance(commit1, str) else commit1
repo.refs.set_symbolic_ref(b"HEAD", branch_ref)
print("HEAD points to:", repo.refs.read_ref(b"HEAD").decode(), flush=True)

print("5. Staging all experiment code and configs...", flush=True)
exp_files = []
for folder in ["configs", "src", "tests", "docs"]:
    p = Path(folder)
    if p.exists():
        for root, dirs, files in os.walk(p):
            dirs[:] = [d for d in dirs if d != "__pycache__" and not d.startswith(".")]
            for f in files:
                if not f.endswith((".pyc", ".pyo", ".log")):
                    exp_files.append(os.path.relpath(os.path.join(root, f), "."))

print(f"Staging {len(exp_files)} files...", flush=True)
porcelain.add(repo, exp_files)

print("6. Committing experiment changes on feature branch...", flush=True)
commit2 = porcelain.commit(
    repo,
    message="feat: adapt Audio-JEPA with asymmetric temporal patching and CTC ASR adapter",
    author="Researcher <researcher@example.com>",
    committer="Researcher <researcher@example.com>",
)
feat_sha = commit2.decode() if isinstance(commit2, bytes) else commit2
print("Feature commit SHA:", feat_sha, flush=True)
repo.refs[branch_ref] = commit2.encode() if isinstance(commit2, str) else commit2

print("\n7. Final Verification:", flush=True)
print("HEAD:", repo.refs.read_ref(b"HEAD").decode(), flush=True)
for ref, sha in repo.get_refs().items():
    print(f"  {ref.decode()} -> {sha.decode()}", flush=True)
print("\nDone! Git repository successfully initialized with base project and switched to feature branch!", flush=True)
