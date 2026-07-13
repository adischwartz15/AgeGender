"""Small, dependency-light provenance helpers -- git commit SHA and
core dependency versions -- shared by every script/manifest that records
"what produced this artifact" (transfer-learning run manifests, the locked
split manifest, per-sample prediction export manifests, calibration
artifacts). Centralized here instead of the same ~15 lines copy-pasted into
each script.
"""

from __future__ import annotations

import platform
import subprocess

from src.utils.config import REPO_ROOT


def git_commit_sha() -> str | None:
    """The current git HEAD commit SHA, or ``None`` if unavailable (not a
    git checkout, or ``git`` isn't on PATH) -- never raises."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def dependency_versions() -> dict[str, str | None]:
    """Python/PyTorch/CUDA/timm versions actually in use -- ``timm`` is
    reported as ``None`` (never an error) when the optional transfer
    dependency isn't installed, since every core (from-scratch) path must
    keep working without it."""
    import torch

    versions: dict[str, str | None] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    try:
        import timm

        versions["timm"] = timm.__version__
    except ImportError:
        versions["timm"] = None
    return versions
