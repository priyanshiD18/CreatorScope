-- =============================================================
-- Part 2: Creator-level derived metrics (used in Python models)
-- =============================================================

-- Compute engagement rate, consistency score, breakout ratio per channel
CREATE OR REPLACE VIEW vw_creator_features AS
WITH video_stats AS (
    SELECT
        channel_id,
        COUNT(*)                                               AS total_videos,
        AVG(views)                                             AS avg_views,
        AVG(CASE WHEN views > 0 THEN (likes + comment_count)::FLOAT / views END)
                                                               AS avg_engagement_rate,
        STDDEV(EXTRACT(EPOCH FROM
            (publish_time - LAG(publish_time) OVER (PARTITION BY channel_id ORDER BY publish_time))
        ) / 86400)                                             AS upload_gap_stddev,  -- consistency (lower = more consistent)
        AVG(EXTRACT(EPOCH FROM
            (publish_time - LAG(publish_time) OVER (PARTITION BY channel_id ORDER BY publish_time))
        ) / 86400)                                             AS avg_upload_gap_days,
        SUM(CASE WHEN views > 2 * AVG(views) OVER (PARTITION BY channel_id) THEN 1 ELSE 0 END)::FLOAT
            / NULLIF(COUNT(*), 0)                             AS breakout_ratio
    FROM videos
    GROUP BY channel_id
)
SELECT
    c.channel_id,
    c.channel_name,
    c.category,
    c.country,
    c.subscribers,
    c.total_views,
    c.avg_weekly_uploads,
    v.total_videos,
    v.avg_views,
    v.avg_engagement_rate,
    v.upload_gap_stddev                                        AS consistency_score_raw,
    v.avg_upload_gap_days,
    v.breakout_ratio,
    -- Consistency score: inverted stddev (higher = more consistent)
    CASE WHEN v.upload_gap_stddev > 0
         THEN 1.0 / v.upload_gap_stddev
         ELSE NULL
    END                                                        AS consistency_score,
    -- View-to-subscriber ratio
    CASE WHEN c.subscribers > 0
         THEN v.avg_views / c.subscribers
         ELSE NULL
    END                                                        AS view_sub_ratio,
    c.created_year,
    c.created_month
FROM channels c
JOIN video_stats v USING (channel_id);


-- Creator retention: still uploading after N days
-- Usage: call with a given cohort start date
CREATE OR REPLACE VIEW vw_creator_retention AS
WITH first_upload AS (
    SELECT channel_id, MIN(publish_time) AS first_video_date
    FROM videos
    GROUP BY channel_id
),
latest_upload AS (
    SELECT channel_id, MAX(publish_time) AS last_video_date
    FROM videos
    GROUP BY channel_id
)
SELECT
    f.channel_id,
    f.first_video_date,
    l.last_video_date,
    DATE_PART('day', l.last_video_date - f.first_video_date) AS active_days,
    CASE WHEN DATE_PART('day', l.last_video_date - f.first_video_date) >= 30  THEN TRUE ELSE FALSE END AS retained_d30,
    CASE WHEN DATE_PART('day', l.last_video_date - f.first_video_date) >= 90  THEN TRUE ELSE FALSE END AS retained_d90,
    CASE WHEN DATE_PART('day', l.last_video_date - f.first_video_date) >= 180 THEN TRUE ELSE FALSE END AS retained_d180,
    -- New creator activation: 5+ videos in first 30 days
    (SELECT COUNT(*) FROM videos v
     WHERE v.channel_id = f.channel_id
       AND v.publish_time BETWEEN f.first_video_date AND f.first_video_date + INTERVAL '30 days'
    ) AS videos_in_first_30_days
FROM first_upload f
JOIN latest_upload l USING (channel_id);
