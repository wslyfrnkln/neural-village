import numpy as np
import pandas as pd
import pytest
from neural_playback.report_card import (
    compute_trait_scores,
    format_report_card,
    generate_report_card,
)
from neural_playback.roi_mapping import load_roi_map


def _fake_timeseries(value: float = 1.0) -> pd.DataFrame:
    """Create a fake ROI timeseries DataFrame for testing."""
    roi_map = load_roi_map()
    roi_names = [r["roi_name"] for r in roi_map]
    data = {name: np.full(10, value, dtype=np.float32) for name in roi_names}
    return pd.DataFrame(data)


def test_compute_trait_scores_returns_all_traits():
    df = _fake_timeseries(1.0)
    scores = compute_trait_scores(df)
    trait_names = {r["trait"] for r in load_roi_map()}
    assert isinstance(scores, dict)
    for trait in trait_names:
        assert trait in scores, f"Missing trait: {trait}"


def test_scores_are_in_range():
    df = _fake_timeseries(1.0)
    scores = compute_trait_scores(df)
    for trait, score in scores.items():
        assert 0.0 <= score <= 10.0, f"{trait} score {score} out of range"


def test_zero_activation_gives_low_score():
    df = _fake_timeseries(0.0)
    scores = compute_trait_scores(df)
    for trait, score in scores.items():
        assert score < 1.0, f"{trait} score {score} should be near zero for zero activation"


def test_format_report_card_contains_all_traits():
    df = _fake_timeseries(1.0)
    scores = compute_trait_scores(df)
    report = format_report_card(scores, track_name="TEST TRACK")
    trait_names = {r["trait"] for r in load_roi_map()}
    for trait in trait_names:
        assert trait in report, f"Missing trait in report: {trait}"


def test_format_report_card_contains_disclaimer():
    df = _fake_timeseries(1.0)
    scores = compute_trait_scores(df)
    report = format_report_card(scores)
    assert "predicted neural engagement" in report


def test_generate_report_card_returns_tuple():
    preds = np.random.rand(8, 100).astype(np.float32)
    labels = np.zeros(100, dtype=np.int32)
    labels[:20] = 34
    scores, report = generate_report_card(preds, labels, track_name="TEST")
    assert isinstance(scores, dict)
    assert isinstance(report, str)
    assert len(scores) >= 5
