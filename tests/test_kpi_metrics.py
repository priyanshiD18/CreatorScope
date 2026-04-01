"""
Unit tests for Part 1: KPI metrics
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.data_loader import generate_synthetic_channels, generate_synthetic_videos
from src.kpi_metrics import (
    compute_engagement_rate,
    compute_consistency_score,
    compute_breakout_ratio,
    gini_coefficient,
    compute_hhi,
    full_kpi_report,
)


@pytest.fixture(scope="module")
def sample_data():
    channels = generate_synthetic_channels(n=50, seed=0)
    videos   = generate_synthetic_videos(channels, n_per_channel=10, seed=0)
    return channels, videos


def test_engagement_rate_bounds(sample_data):
    _, videos = sample_data
    er = compute_engagement_rate(videos)
    assert (er.dropna() >= 0).all(), "Engagement rate must be non-negative"
    # Most should be <= 1 (likes+comments rarely exceed views)
    assert (er.dropna() <= 5).all(), "Suspiciously high engagement rate"


def test_consistency_score_positive(sample_data):
    _, videos = sample_data
    cs = compute_consistency_score(videos)
    assert "consistency_score" in cs.columns
    valid = cs["consistency_score"].dropna()
    assert (valid >= 0).all()


def test_gini_perfect_equality():
    vals = np.ones(100)
    assert abs(gini_coefficient(vals)) < 1e-6


def test_gini_maximum_inequality():
    vals = np.zeros(99)
    vals = np.append(vals, 100.0)
    g = gini_coefficient(vals)
    assert g > 0.9


def test_hhi_range(sample_data):
    channels, _ = sample_data
    hhi = compute_hhi(channels, "category")
    assert 0 <= hhi <= 10_000, f"HHI out of range: {hhi}"


def test_full_kpi_report_keys(sample_data):
    channels, videos = sample_data
    report = full_kpi_report(channels, videos)
    for key in ["gini_coefficient", "hhi_categories", "total_channels", "total_videos"]:
        assert key in report, f"Missing key: {key}"


def test_breakout_ratio_bounds(sample_data):
    _, videos = sample_data
    br = compute_breakout_ratio(videos)
    assert (br >= 0).all() and (br <= 1).all()
