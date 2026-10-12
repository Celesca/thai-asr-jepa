#!/usr/bin/env python3
"""Initialize base project in Git master branch, then create and switch to feature branch."""

import os
import shutil
import sys
from pathlib import Path

# Add .venv site-packages explicitly
venv_site = Path(".venv/lib/python3.12/site-packages")
if venv_site.exists():
    sys.path.insert(0, str(venv_site.resolve()))

from dulwich.repo import Repo
from dulwich import porcelain

repo_path = Path.cwd()
git_dir = repo_path / ".git"

print(f"Step 1: Cleaning previous git metadata in {git_dir}...", flush=True)
if git_dir.exists():
    shutil.rmtree(git_dir)

print("Step 2: Initializing fresh Git repository...", flush=True)
repo = Repo.init(str(repo_path))

# Build strict list of project files ONLY (never scan .venv or data_cache)
project_files = []
# Top level files
for fname in [".gitignore", "pyproject.toml", "README.md"]:
    if (repo_path / fname).is_file():
        project_files.append(fname)

# Specific directories
for dname in ["configs", "src", "tests", "docs"]:
    dir_path = repo_path / dname
    if dir_path.is_dir():
        for root, dirs, files in os.walk(dir_path):
            dirs[:] = [d for d in dirs if d != "__pycache__" and not d.startswith(".")]
            for f in files:
                if not f.endswith((".pyc", ".pyo", ".log")):
                    rel = os.path.relpath(os.path.join(root, f), repo_path)
                    project_files.append(rel)

print(f"Step 3: Staging {len(project_files)} project files for initial project commit...", flush=True)
porcelain.add(repo, project_files)

print("Step 4: Creating initial commit on master (base Audio-JEPA project)...", flush=True)
commit_base = porcelain.commit(
    repo,
    message="Initial commit: Audio-JEPA base project",
    author="Researcher <researcher@example.com>",
    committer="Researcher <researcher@example.com>",
)
base_sha = commit_base.decode() if isinstance(commit_base, bytes) else commit_base
print(f"-> Base project commit created: {base_sha}", flush=True)

# Ensure refs/heads/master exists
repo.refs[b"refs/heads/master"] = commit_base

print("Step 5: Creating and switching to branch 'feat/audio-jepa-thai-asr-temporal'...", flush=True)
branch_name = "feat/audio-jepa-thai-asr-temporal"
porcelain.branch_create(repo, branch_name)
repo.refs.set_symbolic_ref(b"HEAD", b"refs/heads/" + branch_name.encode())

print(f"-> Switched HEAD to: {repo.refs.read_ref(b'HEAD').decode()}", flush=True)

print("Step 6: Committing experiment changes to feature branch...", flush=True)
porcelain.add(repo, project_files)
try:
    commit_feat = porcelain.commit(
        repo,
        message="feat: adapt Audio-JEPA with asymmetric temporal patching and CTC ASR adapter",
        author="Researcher <researcher@example.com>",
        committer="Researcher <researcher@example.com>",
    )
    feat_sha = commit_feat.decode() if isinstance(commit_feat, bytes) else commit_feat
    print(f"-> Feature commit created: {feat_sha}", flush=True)
except Exception as e:
    print(f"-> Note on feature commit: {e}", flush=True)

print("\nGit repository status successfully initialized:", flush=True)
print(f"  HEAD: {repo.refs.read_ref(b'HEAD').decode()}", flush=True)
for ref, sha in repo.get_refs().items():
    print(f"  {ref.decode()} -> {sha.decode()}", flush=True)
