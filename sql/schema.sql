-- CreatorScope PostgreSQL Schema
-- Matches Kaggle datasets: YouTube Trending, Global YouTube Statistics 2023,
-- Ken Jee YouTube Data

CREATE TABLE IF NOT EXISTS channels (
    channel_id        TEXT PRIMARY KEY,
    channel_name      TEXT,
    category          TEXT,
    country           TEXT,
    subscribers       BIGINT,
    total_views       BIGINT,
    total_videos      INT,
    avg_weekly_uploads FLOAT,
    created_year      INT,
    created_month     INT,
    earnings_min      FLOAT,
    earnings_max      FLOAT,
    loaded_at         TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS videos (
    video_id          TEXT PRIMARY KEY,
    channel_id        TEXT REFERENCES channels(channel_id),
    title             TEXT,
    publish_time      TIMESTAMP,
    category          TEXT,
    views             BIGINT,
    likes             BIGINT,
    dislikes          BIGINT,
    comment_count     BIGINT,
    duration_seconds  INT,
    tags              TEXT,
    description_len   INT,
    loaded_at         TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS trending_snapshots (
    snapshot_id       SERIAL PRIMARY KEY,
    video_id          TEXT REFERENCES videos(video_id),
    trending_date     DATE,
    country           TEXT,
    rank_on_day       INT
);

-- Ken Jee-style per-video analytics (creator studio level)
CREATE TABLE IF NOT EXISTS video_analytics (
    analytics_id      SERIAL PRIMARY KEY,
    video_id          TEXT REFERENCES videos(video_id),
    channel_id        TEXT REFERENCES channels(channel_id),
    impressions       BIGINT,
    ctr               FLOAT,       -- click-through rate
    avg_view_duration FLOAT,       -- seconds
    subscribers_gained INT,
    subscribers_lost   INT,
    snapshot_date     DATE
);

-- Derived: daily channel metrics (aggregated from videos)
CREATE TABLE IF NOT EXISTS channel_daily_metrics (
    metric_id         SERIAL PRIMARY KEY,
    channel_id        TEXT REFERENCES channels(channel_id),
    metric_date       DATE,
    views_day         BIGINT,
    subs_day          INT,
    videos_published  INT,
    avg_engagement    FLOAT
);

-- Indexes for analytical queries
CREATE INDEX idx_videos_channel ON videos(channel_id);
CREATE INDEX idx_videos_publish ON videos(publish_time);
CREATE INDEX idx_trending_date  ON trending_snapshots(trending_date);
CREATE INDEX idx_analytics_channel ON video_analytics(channel_id);
CREATE INDEX idx_daily_channel ON channel_daily_metrics(channel_id, metric_date);
