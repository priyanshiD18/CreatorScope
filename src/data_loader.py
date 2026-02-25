"""
data_loader.py
--------------
Loads Kaggle datasets into the PostgreSQL database and provides
a unified DataFrame interface for all analysis notebooks.

Datasets expected in data/raw/:
  - trending_videos.csv   (YouTube Trending Video Dataset)
  - global_yt_stats.csv   (Global YouTube Statistics 2023)
  - video_analytics.csv   (Ken Jee or similar per-video analytics)
"""

import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROC_DIR = Path(__file__).parent.parent / "data" / "processed"

DB_URL = os.environ.get("DB_URL", "postgresql://cs_user:cs_pass@localhost:5432/creatorscope")


def get_engine():
    return create_engine(DB_URL)


# ─────────────────────────────────────────────────────────────────────────────
# Load & normalize raw CSVs
# ─────────────────────────────────────────────────────────────────────────────

def load_global_yt_stats(path: Path = None) -> pd.DataFrame:
    """Load Global YouTube Statistics 2023 (top 1000 channels)."""
    path = path or RAW_DIR / "global_yt_stats.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")

    rename = {
        "Youtuber": "channel_name",
        "subscribers": "subscribers",
        "video views": "total_views",
        "uploads": "total_videos",
        "Country": "country",
        "channel_type": "category",
        "created_year": "created_year",
        "created_month": "created_month",
        "lowest_monthly_earnings": "earnings_min",
        "highest_monthly_earnings": "earnings_max",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Synthetic channel_id from name
    df["channel_id"] = df["channel_name"].str.lower().str.replace(r"\W+", "_", regex=True)

    # Estimate weekly uploads: total_videos / (years_active * 52)
    current_year = 2024
    df["years_active"] = (current_year - df["created_year"].fillna(current_year - 3)).clip(lower=1)
    df["avg_weekly_uploads"] = df["total_videos"] / (df["years_active"] * 52)

    return df


def load_trending_videos(path: Path = None) -> pd.DataFrame:
    """Load YouTube Trending Video Dataset."""
    path = path or RAW_DIR / "trending_videos.csv"
    df = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["publish_time", "trending_date"])

    rename = {
        "video_id": "video_id",
        "title": "title",
        "channel_title": "channel_name",
        "category_id": "category",
        "publish_time": "publish_time",
        "trending_date": "trending_date",
        "views": "views",
        "likes": "likes",
        "dislikes": "dislikes",
        "comment_count": "comment_count",
        "tags": "tags",
        "description": "description",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Synthetic channel_id
    if "channel_id" not in df.columns:
        df["channel_id"] = df["channel_name"].str.lower().str.replace(r"\W+", "_", regex=True)

    # Engagement rate
    df["engagement_rate"] = np.where(
        df["views"] > 0,
        (df["likes"] + df["comment_count"]) / df["views"],
        0,
    )

    return df


def load_video_analytics(path: Path = None) -> pd.DataFrame:
    """Load per-video creator-studio analytics (Ken Jee style)."""
    path = path or RAW_DIR / "video_analytics.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")

    rename = {
        "Video": "title",
        "Video publish time": "publish_time",
        "Views": "views",
        "Likes": "likes",
        "Comments": "comment_count",
        "Subscribers": "subscribers_gained",
        "Impressions": "impressions",
        "Impressions click-through rate (%)": "ctr",
        "Average view duration": "avg_view_duration",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    if "publish_time" in df.columns:
        df["publish_time"] = pd.to_datetime(df["publish_time"], errors="coerce")

    df["video_id"] = df.get("video_id", pd.Series(range(len(df)))).astype(str)
    df["channel_id"] = "ken_jee"

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data generator (when Kaggle files are not present)
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_channels(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic channel data for development/testing."""
    rng = np.random.default_rng(seed)
    categories = ["Entertainment", "Music", "Gaming", "Education", "Tech",
                  "Sports", "News", "Lifestyle", "Comedy", "Science"]
    countries = ["US", "IN", "BR", "GB", "MX", "CA", "DE", "FR", "JP", "KR"]

    # Power-law subscriber distribution
    subscribers = (rng.pareto(1.5, n) * 10_000 + 1_000).astype(int)
    created_year = rng.integers(2008, 2024, n)
    years_active = (2024 - created_year).clip(min=1)
    total_videos = (rng.exponential(200, n) + 10).astype(int)

    df = pd.DataFrame({
        "channel_id": [f"ch_{i:05d}" for i in range(n)],
        "channel_name": [f"Creator_{i}" for i in range(n)],
        "category": rng.choice(categories, n),
        "country": rng.choice(countries, n),
        "subscribers": subscribers,
        "total_views": (subscribers * rng.uniform(50, 500, n)).astype(int),
        "total_videos": total_videos,
        "avg_weekly_uploads": total_videos / (years_active * 52),
        "created_year": created_year,
        "created_month": rng.integers(1, 13, n),
        "earnings_min": subscribers * 0.0001,
        "earnings_max": subscribers * 0.0005,
    })
    return df


def generate_synthetic_videos(channels: pd.DataFrame, n_per_channel: int = 20,
                               seed: int = 42) -> pd.DataFrame:
    """Generate synthetic video records for each channel."""
    rng = np.random.default_rng(seed)
    records = []
    for _, ch in channels.iterrows():
        n = max(1, int(rng.poisson(n_per_channel)))
        base_views = ch["total_views"] / max(ch["total_videos"], 1)
        for i in range(n):
            days_back = rng.integers(1, 365 * max(1, 2024 - ch["created_year"]))
            views = max(0, int(rng.lognormal(np.log(base_views + 1), 1.2)))
            likes = max(0, int(views * rng.beta(2, 20)))
            comments = max(0, int(views * rng.beta(1, 50)))
            records.append({
                "video_id": f"{ch['channel_id']}_v{i:04d}",
                "channel_id": ch["channel_id"],
                "title": f"Video {i} by {ch['channel_name']}",
                "publish_time": pd.Timestamp("2024-01-01") - pd.Timedelta(days=int(days_back)),
                "category": ch["category"],
                "views": views,
                "likes": likes,
                "dislikes": max(0, int(likes * 0.05)),
                "comment_count": comments,
                "duration_seconds": int(rng.uniform(60, 1800)),
                "tags": "",
                "description_len": int(rng.uniform(50, 500)),
            })
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Ingest into PostgreSQL
# ─────────────────────────────────────────────────────────────────────────────

def ingest_to_db(channels: pd.DataFrame, videos: pd.DataFrame):
    engine = get_engine()
    channel_cols = ["channel_id", "channel_name", "category", "country", "subscribers",
                    "total_views", "total_videos", "avg_weekly_uploads",
                    "created_year", "created_month", "earnings_min", "earnings_max"]
    video_cols = ["video_id", "channel_id", "title", "publish_time", "category",
                  "views", "likes", "dislikes", "comment_count",
                  "duration_seconds", "tags", "description_len"]

    channels[channel_cols].drop_duplicates("channel_id").to_sql(
        "channels", engine, if_exists="append", index=False, method="multi"
    )
    videos[video_cols].drop_duplicates("video_id").to_sql(
        "videos", engine, if_exists="append", index=False, method="multi"
    )
    print(f"Ingested {len(channels)} channels and {len(videos)} videos.")


def load_from_db(query: str) -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point: initialise project data
# ─────────────────────────────────────────────────────────────────────────────

def init_data(use_synthetic: bool = False):
    """
    Load data from Kaggle CSVs (if present) or generate synthetic data,
    then ingest into PostgreSQL.
    """
    if use_synthetic or not (RAW_DIR / "global_yt_stats.csv").exists():
        print("Using synthetic data (Kaggle CSVs not found).")
        channels = generate_synthetic_channels(n=500)
        videos = generate_synthetic_videos(channels, n_per_channel=30)
    else:
        channels = load_global_yt_stats()
        videos = load_trending_videos()

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    channels.to_parquet(PROC_DIR / "channels.parquet", index=False)
    videos.to_parquet(PROC_DIR / "videos.parquet", index=False)

    ingest_to_db(channels, videos)
    return channels, videos


if __name__ == "__main__":
    init_data()
