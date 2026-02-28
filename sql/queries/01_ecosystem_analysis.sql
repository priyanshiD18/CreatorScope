-- =============================================================
-- Part 1: Ecosystem KPI Queries
-- =============================================================

-- 1. Median upload frequency by category
SELECT
    category,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY avg_weekly_uploads) AS median_uploads_per_week,
    COUNT(*)                                                          AS channel_count
FROM channels
GROUP BY category
ORDER BY median_uploads_per_week DESC;


-- 2. Power-law: what % of total views do top 1% of channels capture?
WITH ranked AS (
    SELECT
        channel_id,
        total_views,
        NTILE(100) OVER (ORDER BY total_views) AS percentile
    FROM channels
    WHERE total_views IS NOT NULL
),
top1 AS (
    SELECT SUM(total_views) AS top1_views FROM ranked WHERE percentile = 100
),
total AS (
    SELECT SUM(total_views) AS all_views FROM channels
)
SELECT
    ROUND(100.0 * top1.top1_views / total.all_views, 2) AS pct_views_captured_by_top1pct
FROM top1, total;


-- 3. Engagement rate by channel size tier
SELECT
    CASE
        WHEN subscribers < 10000         THEN 'Nano (<10K)'
        WHEN subscribers < 100000        THEN 'Micro (10K-100K)'
        WHEN subscribers < 1000000       THEN 'Mid (100K-1M)'
        WHEN subscribers < 10000000      THEN 'Macro (1M-10M)'
        ELSE                                  'Mega (10M+)'
    END AS tier,
    COUNT(*)                              AS channel_count,
    ROUND(AVG(v.engagement_rate)::NUMERIC, 4) AS avg_engagement_rate
FROM channels c
JOIN LATERAL (
    SELECT
        channel_id,
        CASE WHEN SUM(views) > 0
             THEN SUM(likes + comment_count)::FLOAT / SUM(views)
             ELSE 0
        END AS engagement_rate
    FROM videos
    WHERE channel_id = c.channel_id
    GROUP BY channel_id
) v ON TRUE
GROUP BY tier
ORDER BY MIN(c.subscribers);


-- 4. Which categories have the highest new-creator entry rate?
-- Proxy: channels created in the last 2 years as % of all channels in that category
SELECT
    category,
    COUNT(*) FILTER (WHERE created_year >= EXTRACT(YEAR FROM NOW()) - 2) AS new_channels,
    COUNT(*)                                                               AS total_channels,
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE created_year >= EXTRACT(YEAR FROM NOW()) - 2)
        / NULLIF(COUNT(*), 0), 1
    )                                                                      AS new_entry_pct
FROM channels
GROUP BY category
HAVING COUNT(*) >= 10
ORDER BY new_entry_pct DESC;


-- 5. Gini coefficient of views (manual calculation)
-- Lower = more equal; higher = more concentrated
WITH ordered AS (
    SELECT
        total_views,
        ROW_NUMBER() OVER (ORDER BY total_views) AS i,
        COUNT(*) OVER ()                          AS n,
        SUM(total_views) OVER ()                  AS total
    FROM channels
    WHERE total_views > 0
)
SELECT
    ROUND(
        (2 * SUM(i * total_views) / (n * total) - (n + 1)::FLOAT / n)::NUMERIC,
        4
    ) AS gini_coefficient
FROM ordered
GROUP BY n, total;


-- 6. Geographic distribution of top creators (by subscribers)
SELECT
    country,
    COUNT(*)           AS channel_count,
    SUM(subscribers)   AS total_subscribers,
    AVG(subscribers)   AS avg_subscribers
FROM channels
WHERE subscribers IS NOT NULL
GROUP BY country
ORDER BY total_subscribers DESC
LIMIT 20;

# added Gini coefficient SQL and geographic distribution query

# added Gini coefficient SQL and geographic distribution query
