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
