#!/usr/bin/env python3
"""Stage all files, commit to improve/thai-audio-jepa-asr, and push without blocking."""

import os
import sys
from pathlib import Path

# Disable interactive stdin prompts
os.environ["GIT_TERMINAL_PROMPT"] = "0"

import builtins
import getpass

def no_prompt(*args, **kwargs):
    raise RuntimeError("Authentication required: interactive prompt requested but not available in non-interactive environment")

builtins.input = no_prompt
getpass.getpass = no_prompt

# Add .venv site-packages
venv_site = Path(".venv/lib/python3.12/site-packages")
if venv_site.exists():
    sys.path.insert(0, str(venv_site.resolve()))

from dulwich.repo import Repo
from dulwich import porcelain

log_lines = []

def log(msg):
    print(msg, flush=True)
    log_lines.append(str(msg))

repo_path = Path.cwd()
repo = Repo(str(repo_path))

branch_name = "improve/thai-audio-jepa-asr"
branch_ref = b"refs/heads/" + branch_name.encode()

current_head = repo.refs.read_ref(b"HEAD")
log(f"Current symbolic HEAD: {current_head.decode()}")

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

log(f"Staging {len(files_to_stage)} files on branch {branch_name}...")
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
    log(f"Created commit on {branch_name}: {commit_sha_str}")
    repo.refs[branch_ref] = commit_sha.encode() if isinstance(commit_sha, str) else commit_sha
except Exception as e:
    log(f"Commit note: {e}")

log(f"Branch ref {branch_name} now at: {repo.refs[branch_ref].decode()}")

remote_url = "https://github.com/Celesca/thai-asr-jepa"
log(f"\nAttempting push to {remote_url} (branch {branch_name})...")

try:
    porcelain.push(
        repo,
        remote_location=remote_url,
        refspecs=f"refs/heads/{branch_name}:refs/heads/{branch_name}".encode(),
    )
    log("Push succeeded successfully!")
except Exception as e:
    log(f"Push result: {type(e).__name__}: {e}")

with open("scripts/push_result.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines) + "\n")
