"""Git repository inspection utilities.

Safely retrieves Git commit hash and dirty status, handling environments
where Git CLI is missing or the project is untracked.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import TypedDict


class GitInfo(TypedDict):
    """Git commit metadata."""
    commit_hash: str
    is_dirty: bool
    branch: str
    status_note: str


def get_git_info(repo_path: str | Path | None = None) -> GitInfo:
    """Return Git commit hash, dirty status, and branch.

    Args:
        repo_path: Path to the repository root. If None, uses current working directory.

    Returns:
        GitInfo dictionary with commit_hash, is_dirty, branch, and status_note.
    """
    if repo_path is None:
        repo_path = Path.cwd()
    else:
        repo_path = Path(repo_path)

    try:
        # Try running git rev-parse
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            commit_hash = res.stdout.strip()
            # Check dirty
            dirty_res = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(repo_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            is_dirty = bool(dirty_res.stdout.strip())
            branch_res = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=str(repo_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            branch = branch_res.stdout.strip() if branch_res.returncode == 0 else "unknown"
            return {
                "commit_hash": commit_hash,
                "is_dirty": is_dirty,
                "branch": branch,
                "status_note": "ok",
            }
    except Exception:
        pass

    # Direct filesystem check of .git directory
    git_dir = repo_path / ".git"
    if git_dir.is_dir():
        try:
            head_file = git_dir / "HEAD"
            if head_file.exists():
                head_content = head_file.read_text().strip()
                if head_content.startswith("ref:"):
                    ref_path = head_content[4:].strip()
                    if ref_path.startswith("refs/heads/"):
                        branch = ref_path[len("refs/heads/") :]
                    else:
                        branch = ref_path
                    ref_file = git_dir / ref_path
                    commit_hash = ref_file.read_text().strip() if ref_file.exists() else "unknown"
                else:
                    commit_hash = head_content
                    branch = "detached"
                return {
                    "commit_hash": commit_hash,
                    "is_dirty": True,  # Cannot determine dirty status without git CLI
                    "branch": branch,
                    "status_note": "parsed from .git files without git cli",
                }
        except Exception:
            pass

    return {
        "commit_hash": "untracked",
        "is_dirty": True,
        "branch": "unknown",
        "status_note": "git not initialized or CLI unavailable",
    }
