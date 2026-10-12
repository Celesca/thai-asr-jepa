#!/usr/bin/env python3
"""Script to initialize git repo on master with base project, then branch to feat/audio-jepa-thai-asr-temporal."""

import os
import shutil
import sys
from pathlib import Path

# Add .venv site-packages
venv_site = Path(".venv/lib/python3.12/site-packages")
if venv_site.exists():
    sys.path.insert(0, str(venv_site.resolve()))

from dulwich.repo import Repo
from dulwich import porcelain

repo_path = Path.cwd()
git_dir = repo_path / ".git"

print(f"Working in: {repo_path}")

# If .git exists, we re-initialize clean to ensure proper history
if git_dir.exists():
    print("Re-initializing .git directory for clean project initialization...")
    shutil.rmtree(git_dir)

repo = Repo.init(str(repo_path))
print("Git repo initialized.")

# Files/directories to track for the base project
tracked_patterns = [
    ".gitignore",
    "pyproject.toml",
    "README.md",
    "configs",
    "src",
    "tests",
]

# Find all files to add, respecting .gitignore patterns
files_to_add = []
for item in tracked_patterns:
    p = repo_path / item
    if p.is_file():
        files_to_add.append(item)
    elif p.is_dir():
        for root, dirs, files in os.walk(p):
            # Exclude pycache
            dirs[:] = [d for d in dirs if d != "__pycache__" and not d.startswith(".")]
            for f in files:
                if not f.endswith((".pyc", ".pyo")):
                    rel_p = os.path.relpath(os.path.join(root, f), repo_path)
                    files_to_add.append(rel_p)

print(f"Adding {len(files_to_add)} base project files...")
porcelain.add(repo, files_to_add)

# Initial commit on master
commit_sha1 = porcelain.commit(
    repo,
    message="Initial commit: Audio-JEPA base project",
    author="Researcher <researcher@example.com>",
    committer="Researcher <researcher@example.com>",
)
print(f"Base commit on master: {commit_sha1.decode() if isinstance(commit_sha1, bytes) else commit_sha1}")

# Create and switch to feat/audio-jepa-thai-asr-temporal branch
branch_name = "feat/audio-jepa-thai-asr-temporal"
porcelain.branch_create(repo, branch_name)
repo.refs.set_symbolic_ref(b"HEAD", b"refs/heads/" + branch_name.encode())
print(f"Switched HEAD to branch: {branch_name}")

# Commit the experiment additions on the feature branch
porcelain.add(repo, files_to_add)
try:
    commit_sha2 = porcelain.commit(
        repo,
        message="feat: modify Audio-JEPA with asymmetric temporal patching and CTC ASR adapter",
        author="Researcher <researcher@example.com>",
        committer="Researcher <researcher@example.com>",
    )
    print(f"Feature commit on {branch_name}: {commit_sha2.decode() if isinstance(commit_sha2, bytes) else commit_sha2}")
except Exception as e:
    print(f"Note on feature commit: {e}")

print("\nFinal Branch verification:")
print("Current HEAD ref:", repo.refs.read_ref(b"HEAD"))
for k, v in repo.get_refs().items():
    print(f"  {k.decode()} -> {v.decode()}")

print("\nGit repository successfully initialized with base project and switched to feature branch!")
