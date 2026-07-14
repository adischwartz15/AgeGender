"""Tests for the VOLO transfer-learning extension that must pass with ``timm``
entirely absent -- the mandatory "timm stays optional" constraint.

Tests that require a real ``timm`` install live in
``tests/test_pretrained_volo_with_timm.py`` instead (gated with
``pytest.importorskip("timm")`` at that *file's* top, so only that file is
skipped where the optional extra isn't installed -- putting the skip
mid-file here would abort collection of this whole module instead).
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _tiny_volo_config(pretrained: bool = False) -> dict:
    return {
        "model": {
            "family": "pretrained_volo",
            "volo": {"model_id": "volo_d1_224", "pretrained": pretrained, "pretrained_source": "imagenet1k"},
            "adapters": {"enabled": True, "bottleneck_ratio": 4, "dropout": 0.1},
            "age_head": {"hidden_dim": 16, "dropout": 0.1, "age_min": 0, "age_max": 120},
            "gender_head": {"hidden_dim": 16, "dropout": 0.1, "num_classes": 2},
            "loss_balancing": {
                "mode": "learned_uncertainty",
                "learned_uncertainty": {"init_log_var_age": 0.0, "init_log_var_gender": 0.0},
            },
        }
    }


def _iter_module_level_import_names(py_file: Path) -> set[str]:
    """Return the set of top-level (module-scope, not inside a function/class
    body) imported module names in ``py_file``."""
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:  # tree.body only -- module scope, not nested defs
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_core_project_imports_with_timm_absent():
    """Static check: no file under src/ or scripts/ other than
    src/models/pretrained_volo.py imports 'timm' at module scope (a
    module-scope import would break every core experiment when timm isn't
    installed, even if that file is never used)."""
    offenders = []
    for base in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
        for py_file in base.rglob("*.py"):
            if py_file.name == "pretrained_volo.py":
                continue
            if "timm" in _iter_module_level_import_names(py_file):
                offenders.append(str(py_file.relative_to(REPO_ROOT)))
    assert not offenders, f"module-scope 'import timm' found outside pretrained_volo.py: {offenders}"


def test_core_modules_actually_import_with_timm_blocked():
    """Dynamic check complementing the static one above: in a *fresh*
    subprocess (so this can't pollute the current test session's module
    identities the way monkeypatching sys.modules + importlib.reload would),
    block 'import timm' from ever succeeding and import every core entry
    point plus this project's own optional-dependency guard."""
    script = (
        "import sys\n"
        "class _BlockedTimm:\n"
        "    def find_module(self, name, path=None):\n"
        "        return self if name == 'timm' or name.startswith('timm.') else None\n"
        "    def load_module(self, name):\n"
        "        raise ImportError(\"blocked for test\")\n"
        "sys.meta_path.insert(0, _BlockedTimm())\n"
        "import src.models.multitask_model, src.models.backbone_factory, src.training.trainer\n"
        "import src.training.stages, src.evaluation.comparison, src.evaluation.metrics\n"
        "import src.inference.artifacts, src.models.pretrained_volo\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=REPO_ROOT, capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_selecting_volo_without_timm_raises_clear_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "timm", None)
    from src.models.pretrained_volo import MissingTimmError, PretrainedVOLOFaceOnlyMultiTask

    with pytest.raises(MissingTimmError) as excinfo:
        PretrainedVOLOFaceOnlyMultiTask(_tiny_volo_config())
    assert "requirements-transfer.txt" in str(excinfo.value)


def test_hollow_timm_module_treated_as_missing(monkeypatch):
    """A stray/partial install can leave an importable 'timm' with none of
    the real library's attributes (e.g. an empty namespace-package
    directory left behind by an interrupted install) -- this must be
    treated identically to timm being genuinely absent, not surface as a
    confusing AttributeError deep inside model construction."""
    import types

    hollow_timm = types.ModuleType("timm")  # no create_model, no list_models, ...
    monkeypatch.setitem(sys.modules, "timm", hollow_timm)
    from src.models.pretrained_volo import MissingTimmError, PretrainedVOLOFaceOnlyMultiTask

    with pytest.raises(MissingTimmError) as excinfo:
        PretrainedVOLOFaceOnlyMultiTask(_tiny_volo_config())
    assert "requirements-transfer.txt" in str(excinfo.value)


def test_config_validation_rejects_pretrained_source_outside_imagenet_allowlist():
    from src.models.pretrained_volo import (
        ALLOWED_PRETRAINED_SOURCES, PretrainedSourceNotAllowedError, validate_pretrained_source,
    )

    for bad_source in ("mivolo", "utkface", "mivolo_utkface_finetuned", ""):
        with pytest.raises(PretrainedSourceNotAllowedError):
            validate_pretrained_source(bad_source)
    for good_source in ALLOWED_PRETRAINED_SOURCES:
        validate_pretrained_source(good_source)  # must not raise


def test_pretrained_source_validated_before_any_timm_call(monkeypatch):
    """A disallowed pretrained_source must be rejected before the model ever
    reaches timm.create_model -- i.e. before any weight download is even
    attempted, not just before it 'succeeds'."""
    monkeypatch.setitem(sys.modules, "timm", None)
    from src.models.pretrained_volo import PretrainedSourceNotAllowedError, PretrainedVOLOFaceOnlyMultiTask

    config = _tiny_volo_config()
    config["model"]["volo"]["pretrained_source"] = "mivolo"
    with pytest.raises(PretrainedSourceNotAllowedError):
        PretrainedVOLOFaceOnlyMultiTask(config)
