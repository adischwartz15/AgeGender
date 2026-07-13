"""Notebook-level checks for the supplementary VOLO transfer-learning section
in notebooks/train_evaluate_kaggle.ipynb. Mirrors
tests/test_transfer_learning_notebook.py's approach for the Colab notebook
-- statically parses and validates the real, shipped .ipynb.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "train_evaluate_kaggle.ipynb"


def _load_notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def test_notebook_json_is_valid():
    nb = _load_notebook()
    assert "cells" in nb
    assert len(nb["cells"]) > 0


def test_supplementary_section_exists_and_is_unnumbered():
    nb = _load_notebook()
    headers = [
        "".join(c["source"]).strip()
        for c in nb["cells"]
        if c["cell_type"] == "markdown" and "".join(c["source"]).strip().startswith("#")
    ]
    matches = [h for h in headers if "Supplementary Experiment" in h and "VOLO" in h]
    assert len(matches) == 1
    header_line = matches[0].splitlines()[0]
    assert header_line.startswith("## Supplementary"), "must not be numbered like the core sections (## N. ...)"


def test_supplementary_section_placed_before_archive_and_does_not_renumber_existing_sections():
    nb = _load_notebook()
    numbered_headers = []
    supplementary_index = None
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "markdown":
            continue
        text = "".join(cell["source"]).strip()
        if text.startswith("## Supplementary Experiment"):
            supplementary_index = i
        elif text.startswith("## ") and text[3].isdigit():
            numbered_headers.append((i, text.splitlines()[0]))

    assert supplementary_index is not None
    expected_numbers = [f"## {n}." for n in range(2, 19)]
    found_numbers = [h for _, h in numbered_headers]
    for expected in expected_numbers:
        assert any(h.startswith(expected) for h in found_numbers), f"missing or renumbered: {expected}"

    archive_index = next(i for i, h in numbered_headers if h.startswith("## 18."))
    assert supplementary_index < archive_index


def test_supplementary_section_code_cells_are_syntactically_valid_python():
    nb = _load_notebook()
    in_section = False
    checked = 0
    for cell in nb["cells"]:
        text = "".join(cell["source"]).strip()
        if cell["cell_type"] == "markdown" and text.startswith("## Supplementary Experiment"):
            in_section = True
            continue
        if cell["cell_type"] == "markdown" and text.startswith("## 18."):
            break
        if in_section and cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            checked += 1
    assert checked >= 10


def test_supplementary_section_toggle_defaults_to_off():
    nb = _load_notebook()
    combined = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "RUN_TRANSFER_LEARNING_EXTENSION = False" in combined


def test_supplementary_section_mentions_required_disclosures():
    nb = _load_notebook()
    intro = next(
        "".join(c["source"]) for c in nb["cells"]
        if c["cell_type"] == "markdown" and "".join(c["source"]).strip().startswith("## Supplementary Experiment")
    )
    for phrase in ("MiVOLO", "ImageNet", "from scratch", "optimizer/training schedule", "Table B"):
        assert phrase in intro, f"missing required disclosure phrase: {phrase!r}"


def test_no_stale_transfer_learning_seeds():
    nb = _load_notebook()
    combined = "\n".join("".join(c["source"]) for c in nb["cells"])
    assert "42,43,44" not in combined
    assert "42, 43, 44" not in combined
    assert "TRANSFER_SEEDS = [42, 123, 2026]" in combined


def test_never_imports_google_colab():
    """This is a Kaggle-only notebook -- must never import google.colab or
    call google.colab.drive.mount(...), even conditionally."""
    nb = _load_notebook()
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not any("colab" in alias.name for alias in node.names), f"cell {i} imports google.colab"
            if isinstance(node, ast.ImportFrom):
                assert not (node.module and "colab" in node.module), f"cell {i} imports from google.colab"
    combined = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "drive.mount(" not in combined


def test_working_root_is_under_kaggle_working():
    nb = _load_notebook()
    combined = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert 'Path("/kaggle/working/AgeGender/transfer_learning")' in combined


def test_bootstrap_cell_defines_required_persistence_variables():
    nb = _load_notebook()
    bootstrap = next(
        "".join(c["source"]) for c in nb["cells"]
        if c["cell_type"] == "code" and "Persistent Transfer-Learning Storage and Resume -- bootstrap cell" in "".join(c["source"])
    )
    for var in (
        "PLATFORM", "TRANSFER_SEEDS", "LOCAL_TRANSFER_ROOT", "PERSISTENT_TRANSFER_ROOT",
        "AUTO_RESUME", "SKIP_COMPLETED", "KAGGLE_RESTORE_SOURCE", "ENABLE_KAGGLE_DRIVE_BACKUP",
    ):
        assert var in bootstrap, f"bootstrap cell missing {var!r}"


def test_kaggle_secrets_names_documented_not_hardcoded():
    """The two required Kaggle Secret names must appear (documenting what
    to configure), but no literal credential value may appear anywhere."""
    nb = _load_notebook()
    combined = "\n".join("".join(c["source"]) for c in nb["cells"])
    assert "GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON" in combined
    assert "GOOGLE_DRIVE_FOLDER_ID" in combined
    # No accidental literal secret assignment, e.g. GOOGLE_DRIVE_FOLDER_ID = "1AbC..."
    assert 'GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON = "' not in combined
    assert 'GOOGLE_DRIVE_FOLDER_ID = "' not in combined


def test_kaggle_archive_excludes_datasets_and_credentials_by_construction():
    """Regression guard: the Kaggle archive cell must call build_summary_archive
    (which excludes dataset images/cache/credential-shaped files by
    construction -- see tests/test_persistent_artifacts.py) rather than a
    hand-rolled zipfile call that could forget an exclusion."""
    nb = _load_notebook()
    archive_cell = next(
        "".join(c["source"]) for c in nb["cells"]
        if c["cell_type"] == "code" and "agegender_transfer_learning_artifacts.zip" in "".join(c["source"])
    )
    assert "build_summary_archive" in archive_cell
    assert "include_best_and_last_checkpoints=True" in archive_cell


def test_run_cell_passes_resume_and_persistence_cli_flags():
    nb = _load_notebook()
    run_cell = next(
        "".join(c["source"]) for c in nb["cells"]
        if c["cell_type"] == "code" and "Table B: multi-seed run" in "".join(c["source"])
    )
    for flag in ("--resume", "--skip-completed", "--persistent-root", "--storage-root"):
        assert flag in run_cell, f"run cell missing {flag!r}"


def test_optional_single_seed_cell_exists():
    nb = _load_notebook()
    combined_code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert any("SINGLE_SEED is not None" in src for src in combined_code)
