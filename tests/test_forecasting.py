"""
Unit tests for Part 4: Forecasting & Predictive Models
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.data_loader import generate_synthetic_channels, generate_synthetic_videos
from src.models.forecasting import (
    make_channel_timeseries,
    detect_anomalies,
    build_churn_features,
    train_churn_model,
    build_video_features,
    train_video_predictor,
)


@pytest.fixture(scope="module")
def sample_data():
    channels = generate_synthetic_channels(n=80, seed=99)
    videos   = generate_synthetic_videos(channels, n_per_channel=20, seed=99)
    return channels, videos


def test_timeseries_non_negative(sample_data):
    _, videos = sample_data
    ch_id = videos["channel_id"].iloc[0]
    ts = make_channel_timeseries(videos, ch_id, freq="W")
    assert (ts >= 0).all()
    assert len(ts) > 0


def test_anomaly_detection_columns(sample_data):
    _, videos = sample_data
    result = detect_anomalies(videos)
    assert "is_anomaly" in result.columns
    assert "anomaly_type" in result.columns
    assert result["is_anomaly"].dtype == bool


def test_anomaly_contamination_rate(sample_data):
    _, videos = sample_data
    result = detect_anomalies(videos)
    rate = result["is_anomaly"].mean()
    # Should be a small fraction, not everything
    assert 0 < rate < 0.5


def test_churn_features_has_label(sample_data):
    channels, videos = sample_data
    feat = build_churn_features(channels, videos)
    assert "churned" in feat.columns
    assert feat["churned"].isin([0, 1]).all()


def test_churn_model_auc_above_chance(sample_data):
    channels, videos = sample_data
    feat   = build_churn_features(channels, videos)
    if feat["churned"].nunique() < 2:
        pytest.skip("Not enough churn variety in synthetic sample")
    result = train_churn_model(feat)
    assert result["gbm_auc"] > 0.5, f"AUC below chance: {result['gbm_auc']}"


def test_video_predictor_mape(sample_data):
    channels, videos = sample_data
    feat   = build_video_features(channels, videos)
    result = train_video_predictor(feat)
    # MAPE should be finite and not absurd
    assert np.isfinite(result["mape"])
    assert result["mape"] < 10  # 1000% MAPE would indicate a bug
