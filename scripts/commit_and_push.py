#!/usr/bin/env python3
"""Stage all files, commit to improve/thai-audio-jepa-asr, and push to origin."""

import os
import sys
from pathlib import Path

# Add .venv site-packages
venv_site = Path(".venv/lib/python3.12/site-packages")
if venv_site.exists():
    sys.path.insert(0, str(venv_site.resolve()))

from dulwich.repo import Repo
from dulwich import porcelain

repo_path = Path.cwd()
repo = Repo(str(repo_path))

branch_name = "improve/thai-audio-jepa-asr"
branch_ref = b"refs/heads/" + branch_name.encode()

# Verify HEAD points to this branch
current_head = repo.refs.read_ref(b"HEAD")
print(f"Current symbolic HEAD: {current_head.decode()}", flush=True)

# Collect all project files to stage
files_to_stage = []
for top_file in [".gitignore", "pyproject.toml", "README.md"]:
    if (repo_path / top_file).is_file():
        files_to_stage.append(top_file)

for folder in ["bash", "configs", "docs", "src", "tests", "scripts"]:
    p = repo_path / folder
    if p.is_dir():
        for root, dirs, files in os.walk(p):
            dirs[:] = [d for d in dirs if d != "__pycache__" and not d.startswith(".")]
            for f in files:
                if not f.endswith((".pyc", ".pyo", ".log", ".tmp")):
                    rel_f = os.path.relpath(os.path.join(root, f), repo_path)
                    files_to_stage.append(rel_f)

print(f"Staging {len(files_to_stage)} files on branch {branch_name}...", flush=True)
porcelain.add(repo, files_to_stage)

# Commit
commit_msg = (
    "feat: speech-oriented adaptation, CTC valid lengths, SDPA attention masking, "
    "short-span SSL masking, decoupled optimizer groups, and experiment configs"
)

try:
    commit_sha = porcelain.commit(
        repo,
        message=commit_msg,
        author="Celesca <celesca@example.com>",
        committer="Celesca <celesca@example.com>",
    )
    commit_sha_str = commit_sha.decode() if isinstance(commit_sha, bytes) else commit_sha
    print(f"Created commit on {branch_name}: {commit_sha_str}", flush=True)
    repo.refs[branch_ref] = commit_sha.encode() if isinstance(commit_sha, str) else commit_sha
except Exception as e:
    print(f"Commit note: {e}", flush=True)

print(f"Branch ref {branch_name} now at: {repo.refs[branch_ref].decode()}", flush=True)

# Attempt to push to remote
remote_url = "https://github.com/Celesca/thai-asr-jepa"
print(f"\nAttempting push to {remote_url} (branch {branch_name})...", flush=True)

try:
    porcelain.push(
        repo,
        remote_location=remote_url,
        refspecs=f"refs/heads/{branch_name}:refs/heads/{branch_name}".encode(),
    )
    print("Push succeeded successfully!", flush=True)
except Exception as e:
    print(f"Push result/status: {type(e).__name__}: {e}", flush=True)
