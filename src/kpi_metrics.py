"""
kpi_metrics.py
--------------
Part 1: Creator Ecosystem KPI Framework
Computes all creator-side and platform-side KPIs defined in the blueprint.
"""

import numpy as np
import pandas as pd
import scipy.stats  # gini implemented manually below


# ─────────────────────────────────────────────────────────────────────────────
# Creator-side KPIs
# ─────────────────────────────────────────────────────────────────────────────

def compute_engagement_rate(videos: pd.DataFrame) -> pd.Series:
    """(Likes + Comments) / Views per video."""
    denom = videos["views"].replace(0, np.nan)
    return (videos["likes"] + videos["comment_count"]) / denom


def compute_consistency_score(videos: pd.DataFrame) -> pd.DataFrame:
    """
    Per-channel consistency score.
    Lower upload-gap stddev → more consistent → higher score.
    Returns DataFrame with channel_id, consistency_score, avg_gap_days.
    """
    df = videos.sort_values(["channel_id", "publish_time"]).copy()
    df["gap_days"] = df.groupby("channel_id")["publish_time"].diff().dt.total_seconds() / 86400

    stats = (
        df.groupby("channel_id")["gap_days"]
        .agg(avg_gap_days="mean", std_gap_days="std")
        .reset_index()
    )
    # Consistency = 1 / std (higher std → less consistent)
    stats["consistency_score"] = 1.0 / stats["std_gap_days"].replace(0, np.nan)
    return stats


def compute_breakout_ratio(videos: pd.DataFrame) -> pd.Series:
    """
    Per video: 1 if views > 2x channel average, else 0.
    Aggregate to channel level → breakout ratio.
    """
    channel_avg = videos.groupby("channel_id")["views"].transform("mean")
    videos = videos.copy()
    videos["is_breakout"] = (videos["views"] > 2 * channel_avg).astype(int)
    return videos.groupby("channel_id")["is_breakout"].mean().rename("breakout_ratio")


def compute_view_sub_ratio(channels: pd.DataFrame, videos: pd.DataFrame) -> pd.Series:
    """Average views per video ÷ subscriber count."""
    avg_views = videos.groupby("channel_id")["views"].mean()
    subs = channels.set_index("channel_id")["subscribers"]
    return (avg_views / subs.replace(0, np.nan)).rename("view_sub_ratio")


def compute_growth_velocity(channel_daily: pd.DataFrame,
                             window_days: int = 30) -> pd.DataFrame:
    """
    Subscriber gain rate over rolling window.
    channel_daily must have: channel_id, metric_date, subs_day
    """
    df = channel_daily.sort_values(["channel_id", "metric_date"])
    df["subs_rolling"] = (
        df.groupby("channel_id")["subs_day"]
        .transform(lambda x: x.rolling(window_days, min_periods=1).sum())
    )
    return df[["channel_id", "metric_date", "subs_rolling"]].rename(
        columns={"subs_rolling": f"growth_velocity_{window_days}d"}
    )


# ─────────────────────────────────────────────────────────────────────────────
# Platform-side KPIs
# ─────────────────────────────────────────────────────────────────────────────

def compute_creator_retention(videos: pd.DataFrame,
                               thresholds: list = [30, 90, 180]) -> pd.DataFrame:
    """
    D30/D90/D180 creator retention: % of channels still uploading after N days.
    """
    first = videos.groupby("channel_id")["publish_time"].min().rename("first_upload")
    last = videos.groupby("channel_id")["publish_time"].max().rename("last_upload")
    df = pd.concat([first, last], axis=1)
    df["active_days"] = (df["last_upload"] - df["first_upload"]).dt.days

    for t in thresholds:
        df[f"retained_d{t}"] = (df["active_days"] >= t)

    retention_rates = {f"d{t}_retention_pct": df[f"retained_d{t}"].mean() * 100
                       for t in thresholds}
    return df, retention_rates


def compute_activation_rate(videos: pd.DataFrame, min_videos: int = 5,
                             window_days: int = 30) -> float:
    """
    % of new channels that upload ≥ min_videos in first window_days.
    """
    first = videos.groupby("channel_id")["publish_time"].min()
    def activated(grp):
        ch = grp.name
        start = first[ch]
        return (grp["publish_time"] <= start + pd.Timedelta(days=window_days)).sum() >= min_videos

    result = videos.groupby("channel_id").apply(activated)
    return result.mean() * 100


def compute_hhi(channels: pd.DataFrame, col: str = "category") -> float:
    """
    Herfindahl-Hirschman Index for category concentration.
    HHI = sum of squared market share percentages.
    Score near 0 = diverse; near 10,000 = monopoly.
    """
    shares = channels[col].value_counts(normalize=True)
    return float((shares ** 2).sum() * 10_000)


def compute_trending_accessibility(trending: pd.DataFrame,
                                    channels: pd.DataFrame,
                                    small_threshold: int = 100_000) -> float:
    """
    % of trending video slots going to channels with < small_threshold subscribers.
    """
    merged = trending.merge(
        channels[["channel_id", "subscribers"]], on="channel_id", how="left"
    )
    small = (merged["subscribers"] < small_threshold).sum()
    return small / len(merged) * 100


def gini_coefficient(values: np.ndarray) -> float:
    """
    Compute Gini coefficient of an array of non-negative values.
    0 = perfect equality, 1 = maximum inequality.
    """
    values = np.sort(values[values > 0])
    n = len(values)
    if n == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * (index * values).sum() / (n * values.sum())) - (n + 1) / n)


# ─────────────────────────────────────────────────────────────────────────────
# Full KPI report
# ─────────────────────────────────────────────────────────────────────────────

def full_kpi_report(channels: pd.DataFrame, videos: pd.DataFrame) -> dict:
    """Compute all KPIs and return as a dict for dashboard consumption."""
    videos = videos.copy()
    videos["engagement_rate"] = compute_engagement_rate(videos)
    consistency = compute_consistency_score(videos)
    breakout = compute_breakout_ratio(videos)
    view_sub = compute_view_sub_ratio(channels, videos)
    _, retention_rates = compute_creator_retention(videos)
    activation = compute_activation_rate(videos)
    hhi = compute_hhi(channels, "category")
    gini = gini_coefficient(channels["total_views"].fillna(0).values)

    # Channel size tier engagement
    tier_engagement = (
        videos.groupby(
            pd.cut(
                videos.merge(channels[["channel_id", "subscribers"]], on="channel_id")["subscribers"],
                bins=[0, 10_000, 100_000, 1_000_000, 10_000_000, np.inf],
                labels=["Nano", "Micro", "Mid", "Macro", "Mega"],
            )
        )["engagement_rate"]
        .mean()
        .to_dict()
    )

    return {
        "total_channels": len(channels),
        "total_videos": len(videos),
        "median_engagement_rate": float(videos["engagement_rate"].median()),
        "gini_coefficient": round(gini, 4),
        "hhi_categories": round(hhi, 1),
        "activation_rate_pct": round(activation, 2),
        "retention_rates": retention_rates,
        "tier_engagement": tier_engagement,
    }

# platform KPIs: creator retention D30/D90/D180, HHI, activation rate

# platform KPIs: creator retention D30/D90/D180, HHI, activation rate
