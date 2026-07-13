"""Notebook-level checks for the T8 execution-flag cells added to both
notebooks: the pretrained-ResNet-18/50 bridge baseline and the
non-parametric (raw/PCA + frozen-backbone) baselines.

Mirrors tests/test_transfer_learning_notebook.py's approach -- statically
parses and validates the real, shipped .ipynb files rather than
reimplementing their logic (which could drift from what actually ships).
Parametrized over both notebooks since the two extensions were added to
both, with only the Colab-specific Drive-sync calls differing.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
COLAB_PATH = REPO_ROOT / "notebooks" / "train_evaluate_colab.ipynb"
KAGGLE_PATH = REPO_ROOT / "notebooks" / "train_evaluate_kaggle.ipynb"
NOTEBOOK_PATHS = [COLAB_PATH, KAGGLE_PATH]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _markdown_headers(nb: dict) -> list[tuple[int, str]]:
    out = []
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "markdown":
            continue
        text = "".join(c["source"]).strip()
        if text.startswith("#"):
            out.append((i, text.splitlines()[0]))
    return out


def _code_cell_containing(nb: dict, needle: str) -> str:
    return next(
        "".join(c["source"]) for c in nb["cells"]
        if c["cell_type"] == "code" and needle in "".join(c["source"])
    )


@pytest.mark.parametrize("path", NOTEBOOK_PATHS, ids=lambda p: p.name)
class TestNonParametricBaselinesCell:
    def test_section_exists_unnumbered_and_before_archive(self, path):
        nb = _load(path)
        headers = _markdown_headers(nb)
        matches = [(i, h) for i, h in headers if h.startswith("### Non-Parametric Baselines")]
        assert len(matches) == 1
        section_idx = matches[0][0]
        archive_idx = next(i for i, h in headers if h.startswith("## 18."))
        supplementary_idx = next(i for i, h in headers if h.startswith("## Supplementary Experiment"))
        assert supplementary_idx < section_idx < archive_idx

    def test_toggle_defaults_off(self, path):
        nb = _load(path)
        code = _code_cell_containing(nb, "RUN_NONPARAMETRIC_BASELINES")
        assert "RUN_NONPARAMETRIC_BASELINES = False" in code

    def test_reuses_repository_scripts_not_duplicated_logic(self, path):
        nb = _load(path)
        code = _code_cell_containing(nb, "RUN_NONPARAMETRIC_BASELINES")
        assert "tune_nonparametric_baselines.py" in code
        assert "evaluate_nonparametric_baselines.py" in code

    def test_code_is_syntactically_valid(self, path):
        nb = _load(path)
        code = _code_cell_containing(nb, "RUN_NONPARAMETRIC_BASELINES")
        ast.parse(code)


@pytest.mark.parametrize("path", NOTEBOOK_PATHS, ids=lambda p: p.name)
class TestPretrainedResNetCell:
    def test_section_exists_unnumbered_and_before_archive(self, path):
        nb = _load(path)
        headers = _markdown_headers(nb)
        matches = [(i, h) for i, h in headers if h.startswith("### Pretrained ResNet")]
        assert len(matches) == 1
        section_idx = matches[0][0]
        archive_idx = next(i for i, h in headers if h.startswith("## 18."))
        supplementary_idx = next(i for i, h in headers if h.startswith("## Supplementary Experiment"))
        assert supplementary_idx < section_idx < archive_idx

    def test_toggle_defaults_off(self, path):
        nb = _load(path)
        code = _code_cell_containing(nb, "RUN_PRETRAINED_RESNET_EXTENSION")
        assert "RUN_PRETRAINED_RESNET_EXTENSION = False" in code

    def test_default_family_is_resnet18_required_not_resnet50_optional(self, path):
        nb = _load(path)
        code = _code_cell_containing(nb, "RUN_PRETRAINED_RESNET_EXTENSION")
        assert 'PRETRAINED_RESNET_FAMILY = "pretrained_resnet18"' in code
        assert "pretrained_resnet50" in code  # mentioned as the optional alternative

    def test_reuses_run_transfer_learning_script_via_model_family_flag(self, path):
        nb = _load(path)
        code = _code_cell_containing(nb, "RUN_PRETRAINED_RESNET_EXTENSION")
        assert "run_transfer_learning.py" in code
        assert "--model-family" in code

    def test_reuses_canonical_seeds_not_hardcoded(self, path):
        nb = _load(path)
        code = _code_cell_containing(nb, "RUN_PRETRAINED_RESNET_EXTENSION")
        assert "TRANSFER_SEEDS" in code
        assert "42,43,44" not in code

    def test_reuses_resume_and_skip_completed_flags(self, path):
        nb = _load(path)
        code = _code_cell_containing(nb, "RUN_PRETRAINED_RESNET_EXTENSION")
        for flag in ("--resume", "--skip-completed", "--persistent-root", "--storage-root"):
            assert flag in code, f"missing {flag!r}"

    def test_code_is_syntactically_valid(self, path):
        nb = _load(path)
        code = _code_cell_containing(nb, "RUN_PRETRAINED_RESNET_EXTENSION")
        ast.parse(code)


def test_colab_new_cells_sync_to_drive_after_each_extension():
    """Only Colab has sync_after_phase (Drive persistence) -- Kaggle
    persists automatically via /kaggle/working, see test below."""
    nb = _load(COLAB_PATH)
    nonparam_code = _code_cell_containing(nb, "RUN_NONPARAMETRIC_BASELINES")
    resnet_code = _code_cell_containing(nb, "RUN_PRETRAINED_RESNET_EXTENSION")
    assert "sync_after_phase(" in nonparam_code
    assert "sync_after_phase(" in resnet_code
    assert "--sync-after-epoch" in resnet_code


def test_kaggle_new_cells_do_not_reference_colab_only_drive_sync():
    """sync_after_phase and --sync-after-epoch are Colab-only symbols (the
    former is defined in a Colab-only cell that checks IN_COLAB; the latter
    is never used by any other Kaggle cell either) -- referencing them in
    the Kaggle notebook would raise NameError / pass a meaningless flag."""
    nb = _load(KAGGLE_PATH)
    nonparam_code = _code_cell_containing(nb, "RUN_NONPARAMETRIC_BASELINES")
    resnet_code = _code_cell_containing(nb, "RUN_PRETRAINED_RESNET_EXTENSION")
    assert "sync_after_phase(" not in nonparam_code
    assert "sync_after_phase(" not in resnet_code
    assert "--sync-after-epoch" not in resnet_code


@pytest.mark.parametrize("path", NOTEBOOK_PATHS, ids=lambda p: p.name)
def test_all_referenced_bootstrap_variables_are_actually_defined(path):
    """The new cells reuse REPO_DIR/RUN_DIR/TRANSFER_SEEDS/LOCAL_TRANSFER_ROOT/
    PERSISTENT_TRANSFER_ROOT/AUTO_RESUME/SKIP_COMPLETED from the existing
    Persistent Transfer-Learning Storage bootstrap cell rather than
    redefining them -- guard against silent drift if that cell is ever
    renamed."""
    nb = _load(path)
    full_source = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    for var in (
        "REPO_DIR", "RUN_DIR", "TRANSFER_SEEDS", "LOCAL_TRANSFER_ROOT",
        "PERSISTENT_TRANSFER_ROOT", "AUTO_RESUME", "SKIP_COMPLETED",
    ):
        assert f"{var} =" in full_source or f"{var}=" in full_source, f"{var} never defined in notebook"


@pytest.mark.parametrize("path", NOTEBOOK_PATHS, ids=lambda p: p.name)
def test_notebook_remains_valid_nbformat(path):
    import warnings

    nbformat = pytest.importorskip("nbformat")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
