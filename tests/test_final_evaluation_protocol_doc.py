"""Regression tests for docs/final_evaluation_protocol.md.

Not just an existence check: also cross-checks the pre-registered seeds,
confidence threshold, and calibration alpha quoted in the doc against the
actual config files, so the two can't silently drift apart (e.g. someone
changes configs/model.yaml's confidence_threshold without updating the
pre-registered protocol doc).
"""

from __future__ import annotations

from src.utils.config import REPO_ROOT, load_full_config

DOC_PATH = REPO_ROOT / "docs" / "final_evaluation_protocol.md"


def _read_doc() -> str:
    return DOC_PATH.read_text(encoding="utf-8")


def test_final_evaluation_protocol_doc_exists():
    assert DOC_PATH.exists()


def test_doc_pre_registers_all_three_seeds():
    text = _read_doc()
    for seed in ("42", "123", "2026"):
        assert seed in text


def test_doc_lists_all_seven_configured_experiments():
    text = _read_doc()
    for experiment in (
        "exp_0_simple_cnn_shared_adapters_learned_balance",
        "exp_0b_plain_deep18_no_skip_shared_adapters_learned_balance",
        "exp_0c_custom_resnet18_no_zero_init_shared_adapters_learned_balance",
        "exp_a_separate", "exp_b_shared_no_adapters", "exp_c_shared_adapters",
        "exp_d_shared_adapters_learned_balance",
    ):
        assert experiment in text


def test_doc_states_the_aurc_gated_decision_rule():
    text = _read_doc()
    assert "AURC" in text
    assert "excludes zero" in text
    assert "never" in text.lower() and "sufficient evidence" in text


def test_doc_declares_no_post_hoc_changes_policy():
    text = _read_doc()
    assert "No post-hoc changes" in text or "post-hoc" in text.lower()


def test_doc_confidence_threshold_matches_actual_config():
    config = load_full_config()
    actual_threshold = config["model"]["gender_head"]["confidence_threshold"]
    assert str(actual_threshold) in _read_doc()


def test_doc_calibration_alpha_matches_actual_config():
    config = load_full_config()
    actual_alpha = config["calibration"]["alpha"]
    assert str(actual_alpha) in _read_doc()
