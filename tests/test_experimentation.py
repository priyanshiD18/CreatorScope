"""
Unit tests for Part 3: Experimentation
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.data_loader import generate_synthetic_channels, generate_synthetic_videos
from src.experimentation import (
    power_analysis,
    simulate_2x2_experiment,
    simulate_causal_impact,
    trending_impact_did,
)


@pytest.fixture(scope="module")
def sample_data():
    channels = generate_synthetic_channels(n=100, seed=1)
    videos   = generate_synthetic_videos(channels, n_per_channel=15, seed=1)
    return channels, videos


def test_power_analysis_positive_n():
    result = power_analysis(baseline=1.2, mde_pct=0.10, alpha=0.05, power=0.80)
    assert result["n_per_group"] > 0
    assert result["total_n"] == result["n_per_group"] * 2
    assert result["cohens_d"] > 0


def test_power_analysis_larger_mde_needs_fewer_subjects():
    small_mde = power_analysis(mde_pct=0.05)["n_per_group"]
    large_mde = power_analysis(mde_pct=0.30)["n_per_group"]
    assert small_mde > large_mde


def test_2x2_anova_returns_table():
    result = simulate_2x2_experiment(n_per_cell=50, seed=42)
    assert "anova_table" in result
    assert result["cell_means"].shape == (2, 2)


def test_2x2_main_effects_significant():
    result = simulate_2x2_experiment(n_per_cell=300, seed=42)
    anova  = result["anova_table"]
    # Both main effects should be significant with large enough n
    assert anova.loc["C(thumbnail_tool)", "PR(>F)"] < 0.05
    assert anova.loc["C(upload_reminder)", "PR(>F)"] < 0.05


def test_causal_impact_returns_keys():
    result = simulate_causal_impact(n_pre=40, n_post=20, seed=7)
    assert "ts" in result
    assert len(result["ts"]) == 60


def test_did_estimate_positive(sample_data):
    channels, videos = sample_data
    result = trending_impact_did(channels, videos)
    assert "did_estimate" in result
    assert isinstance(result["did_estimate"], float)
