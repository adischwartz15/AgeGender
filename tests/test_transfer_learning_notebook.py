"""Notebook-level checks for the supplementary VOLO transfer-learning section
added to notebooks/train_evaluate_colab.ipynb.

Notebook cells aren't natively importable/unit-testable (see
tests/test_notebook_resume_logic.py for the same approach used on the core
notebook's helper cells) -- these checks parse the real, shipped .ipynb and
statically validate it, rather than a reimplementation that could drift.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "train_evaluate_colab.ipynb"


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
    """Must sit before '## 18. Artifact persistence...' and every existing
    numbered header (1-18) must still be present, unchanged, and in order --
    i.e. inserting this section must not have renumbered anything."""
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
    expected_numbers = [f"## {n}." for n in range(2, 19)]  # ## 1. is the top-level title (# ...), not ## 1.
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
    assert checked >= 10  # sanity: the section really has multiple code cells


def test_supplementary_section_toggle_defaults_to_off():
    """VOLO must never run by default -- RUN_TRANSFER_LEARNING_EXTENSION = False."""
    nb = _load_notebook()
    combined = "\n".join(
        "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"
    )
    assert "RUN_TRANSFER_LEARNING_EXTENSION = False" in combined


def test_supplementary_section_mentions_required_disclosures():
    """Cheap regression guard: the required prose disclosures (separate from
    core, ImageNet not UTKFace/MiVOLO, face-only, optimizer/schedule confound)
    are actually present in the intro markdown, not just in this repo's docs."""
    nb = _load_notebook()
    intro = next(
        "".join(c["source"]) for c in nb["cells"]
        if c["cell_type"] == "markdown" and "".join(c["source"]).strip().startswith("## Supplementary Experiment")
    )
    for phrase in ("MiVOLO", "ImageNet", "from scratch", "optimizer/training schedule", "Table B"):
        assert phrase in intro, f"missing required disclosure phrase: {phrase!r}"


# -- Persistence / resume ---------------------------------------------------------


def test_no_stale_transfer_learning_seeds():
    nb = _load_notebook()
    combined = "\n".join("".join(c["source"]) for c in nb["cells"])
    assert "42,43,44" not in combined
    assert "42, 43, 44" not in combined
    assert "TRANSFER_LEARNING_SEEDS" not in combined  # renamed to TRANSFER_SEEDS
    assert "TRANSFER_SEEDS = [42, 123, 2026]" in combined


def test_persistence_and_resume_section_exists_before_run_cell():
    nb = _load_notebook()
    headers = [
        (i, "".join(c["source"]).strip().splitlines()[0])
        for i, c in enumerate(nb["cells"])
        if c["cell_type"] == "markdown" and "".join(c["source"]).strip()
    ]
    persistence_matches = [i for i, h in headers if h.startswith("### Persistent Transfer-Learning Storage and Resume")]
    assert len(persistence_matches) == 1
    persistence_idx = persistence_matches[0]

    supplementary_idx = next(i for i, h in headers if h.startswith("## Supplementary Experiment"))
    table_b_idx = next(
        i for i, c in enumerate(nb["cells"])
        if c["cell_type"] == "code" and "Table B: multi-seed run" in "".join(c["source"])
    )
    assert supplementary_idx < persistence_idx < table_b_idx


def test_bootstrap_cell_defines_required_persistence_variables():
    nb = _load_notebook()
    bootstrap = next(
        "".join(c["source"]) for c in nb["cells"]
        if c["cell_type"] == "code" and "Persistent Transfer-Learning Storage and Resume -- bootstrap cell" in "".join(c["source"])
    )
    for var in (
        "PLATFORM", "TRANSFER_SEEDS", "LOCAL_TRANSFER_ROOT", "PERSISTENT_TRANSFER_ROOT",
        "AUTO_RESUME", "SKIP_COMPLETED", "SYNC_AFTER_EPOCH",
    ):
        assert var in bootstrap, f"bootstrap cell missing {var!r}"
    assert "restore_seed" in bootstrap  # restores from Drive before showing status
    assert "seed_status_report" in bootstrap


def test_run_cell_passes_resume_and_persistence_cli_flags():
    nb = _load_notebook()
    run_cell = next(
        "".join(c["source"]) for c in nb["cells"]
        if c["cell_type"] == "code" and "Table B: multi-seed run" in "".join(c["source"])
    )
    for flag in ("--resume", "--skip-completed", "--sync-after-epoch", "--persistent-root", "--storage-root"):
        assert flag in run_cell, f"run cell missing {flag!r}"


def test_optional_single_seed_cell_exists():
    nb = _load_notebook()
    combined_code = [
        "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"
    ]
    assert any("SINGLE_SEED is not None" in src for src in combined_code)


def test_bootstrap_cell_never_depends_on_undeclared_prior_state():
    """The bootstrap cell must derive LOCAL_TRANSFER_ROOT/PERSISTENT_TRANSFER_ROOT
    itself rather than reusing a stale RUN_DIR-relative path from a dead kernel."""
    nb = _load_notebook()
    bootstrap = next(
        "".join(c["source"]) for c in nb["cells"]
        if c["cell_type"] == "code" and "Persistent Transfer-Learning Storage and Resume -- bootstrap cell" in "".join(c["source"])
    )
    assert "RUN_DIR /" not in bootstrap
    assert 'Path("/content/AgeGender_runtime/transfer_learning")' in bootstrap
