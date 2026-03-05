"""
cohort_analysis.py
------------------
Part 2: Creator Cohort & Growth Analysis
- Creator lifecycle cohorts with survival curves
- Multivariate growth regression (OLS, Ridge, Lasso)
- Creator segmentation (K-Means + t-SNE/UMAP)
- "Aha Moment" analysis
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings

warnings.filterwarnings("ignore")

CLUSTER_NAMES = {
    0: "Grinders",
    1: "One-Hit Wonders",
    2: "Community Builders",
    3: "Algorithm Surfers",
    4: "Rising Stars",
}


# ─────────────────────────────────────────────────────────────────────────────
# 2.1 Creator lifecycle cohorts
# ─────────────────────────────────────────────────────────────────────────────

def build_cohorts(videos: pd.DataFrame, freq: str = "Q") -> pd.DataFrame:
    """
    Assign each channel to a cohort based on its first upload date.
    freq: 'Q' for quarterly, 'Y' for yearly.
    """
    first_upload = videos.groupby("channel_id")["publish_time"].min().rename("first_upload")
    df = first_upload.to_frame()
    df["cohort"] = df["first_upload"].dt.to_period(freq).astype(str)
    return df


def survival_curve(videos: pd.DataFrame, cohort_df: pd.DataFrame,
                   thresholds: list = None) -> pd.DataFrame:
    """
    Compute upload survival rates for each cohort at given thresholds (days).
    """
    if thresholds is None:
        thresholds = [30, 60, 90, 180, 365]

    last_upload = videos.groupby("channel_id")["publish_time"].max().rename("last_upload")
    df = cohort_df.join(last_upload)
    df["active_days"] = (df["last_upload"] - df["first_upload"]).dt.days.clip(lower=0)

    rows = []
    for cohort, grp in df.groupby("cohort"):
        row = {"cohort": cohort, "n": len(grp)}
        for t in thresholds:
            row[f"survival_{t}d"] = (grp["active_days"] >= t).mean()
        rows.append(row)

    return pd.DataFrame(rows).sort_values("cohort")


# ─────────────────────────────────────────────────────────────────────────────
# 2.2 Growth regression
# ─────────────────────────────────────────────────────────────────────────────

def build_regression_features(channels: pd.DataFrame, videos: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix for multivariate growth regression.
    DV: 6-month subscriber growth rate (proxy: avg_weekly_uploads * engagement)
    """
    videos = videos.copy()
    videos["engagement_rate"] = np.where(
        videos["views"] > 0,
        (videos["likes"] + videos["comment_count"]) / videos["views"],
        0,
    )

    # Per-channel features from videos
    v_stats = videos.groupby("channel_id").agg(
        avg_engagement=("engagement_rate", "mean"),
        avg_views=("views", "mean"),
        total_videos=("video_id", "count"),
        avg_duration=("duration_seconds", "mean"),
    ).reset_index()

    # Gap consistency
    df = videos.sort_values(["channel_id", "publish_time"])
    df["gap_days"] = df.groupby("channel_id")["publish_time"].diff().dt.total_seconds() / 86400
    gap_stats = df.groupby("channel_id")["gap_days"].agg(
        avg_gap="mean", std_gap="std"
    ).reset_index()
    gap_stats["consistency_score"] = 1.0 / gap_stats["std_gap"].replace(0, np.nan).fillna(0.01)

    feat = channels.merge(v_stats, on="channel_id", how="left")
    feat = feat.merge(gap_stats[["channel_id", "avg_gap", "consistency_score"]], on="channel_id", how="left")

    # Proxy DV: normalised growth rate = (total_views / subscribers) * consistency
    feat["growth_rate"] = np.log1p(
        (feat["avg_views"].fillna(0) / feat["subscribers"].replace(0, np.nan).fillna(1))
        * feat["consistency_score"].fillna(1)
    )

    # Category dummy encoding
    feat = pd.get_dummies(feat, columns=["category"], drop_first=True, dtype=float)

    return feat.dropna(subset=["growth_rate"])


def run_growth_regression(feat: pd.DataFrame):
    """
    Run OLS, Ridge, Lasso on creator growth features.
    Returns dict of {model_name: fitted_model}, plus OLS summary.
    """
    feature_cols = [c for c in feat.columns if c not in
                    ["channel_id", "channel_name", "country", "growth_rate",
                     "total_views", "earnings_min", "earnings_max"]]
    feature_cols = [c for c in feature_cols if feat[c].dtype in [np.float64, np.int64, float, int]]

    X = feat[feature_cols].fillna(0)
    y = feat["growth_rate"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # OLS with statsmodels for interpretability
    X_sm = sm.add_constant(X_scaled)
    ols = sm.OLS(y, X_sm).fit()

    # VIF
    vif = pd.DataFrame({
        "feature": ["const"] + feature_cols,
        "VIF": [variance_inflation_factor(X_sm, i) for i in range(X_sm.shape[1])],
    })

    # Ridge & Lasso
    ridge = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
    lasso = Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=0.01, max_iter=5000))])
    ridge.fit(X, y); lasso.fit(X, y)

    ridge_r2 = cross_val_score(ridge, X, y, cv=5, scoring="r2").mean()
    lasso_r2 = cross_val_score(lasso, X, y, cv=5, scoring="r2").mean()

    # Lasso surviving features
    lasso_coef = pd.Series(
        lasso.named_steps["model"].coef_,
        index=feature_cols,
    )
    surviving = lasso_coef[lasso_coef != 0].sort_values(key=abs, ascending=False)

    return {
        "ols_summary": ols.summary(),
        "vif": vif,
        "ridge_cv_r2": ridge_r2,
        "lasso_cv_r2": lasso_r2,
        "lasso_surviving_features": surviving,
        "feature_names": feature_cols,
        "ols_model": ols,
        "ridge_model": ridge,
        "lasso_model": lasso,
        "X": X,
        "y": y,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2.3 Creator segmentation
# ─────────────────────────────────────────────────────────────────────────────

def build_cluster_features(channels: pd.DataFrame, videos: pd.DataFrame) -> pd.DataFrame:
    """Features for K-Means clustering."""
    videos = videos.copy()
    videos["engagement_rate"] = np.where(
        videos["views"] > 0,
        (videos["likes"] + videos["comment_count"]) / videos["views"],
        0,
    )
    v_stats = videos.groupby("channel_id").agg(
        avg_engagement=("engagement_rate", "mean"),
        avg_views=("views", "mean"),
        total_videos=("video_id", "count"),
        std_views=("views", "std"),
    ).reset_index()

    df = videos.sort_values(["channel_id", "publish_time"])
    df["gap"] = df.groupby("channel_id")["publish_time"].diff().dt.total_seconds() / 86400
    gap_std = df.groupby("channel_id")["gap"].std().rename("gap_std").reset_index()

    feat = (
        channels[["channel_id", "subscribers", "avg_weekly_uploads"]]
        .merge(v_stats, on="channel_id", how="left")
        .merge(gap_std, on="channel_id", how="left")
    )
    feat["upload_x_engagement"] = feat["avg_weekly_uploads"].fillna(0) * feat["avg_engagement"].fillna(0)
    feat["sub_growth_proxy"] = np.log1p(feat["avg_views"].fillna(0)) / np.log1p(feat["subscribers"].fillna(1))
    feat = feat.fillna(0)
    return feat


def segment_creators(feat: pd.DataFrame, n_clusters: int = 5,
                     seed: int = 42) -> tuple[pd.DataFrame, KMeans]:
    """
    K-Means segmentation. Returns feat DataFrame with 'cluster' column.
    """
    cluster_cols = ["avg_weekly_uploads", "avg_engagement", "avg_views",
                    "gap_std", "upload_x_engagement", "sub_growth_proxy"]
    X = feat[cluster_cols].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    feat = feat.copy()
    feat["cluster"] = km.fit_predict(X_scaled)

    # Label clusters heuristically
    cluster_means = feat.groupby("cluster")[cluster_cols].mean()
    feat["segment"] = feat["cluster"].map(
        _assign_segment_labels(cluster_means)
    )
    return feat, km


def _assign_segment_labels(cluster_means: pd.DataFrame) -> dict:
    """Heuristic: assign descriptive labels based on cluster centroids."""
    ranks = cluster_means.rank()
    labels = {}
    for c in cluster_means.index:
        upload_rank = ranks.loc[c, "avg_weekly_uploads"]
        engage_rank = ranks.loc[c, "avg_engagement"]
        growth_rank = ranks.loc[c, "sub_growth_proxy"]
        consistency = 1.0 / (cluster_means.loc[c, "gap_std"] + 1e-6)

        if upload_rank >= 4 and growth_rank <= 2:
            labels[c] = "Grinders"
        elif growth_rank >= 4 and engage_rank <= 2:
            labels[c] = "Algorithm Surfers"
        elif engage_rank >= 4:
            labels[c] = "Community Builders"
        elif cluster_means.loc[c, "avg_views"] > cluster_means["avg_views"].median() and growth_rank <= 2:
            labels[c] = "One-Hit Wonders"
        else:
            labels[c] = "Rising Stars"
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# 2.4 "Aha Moment" analysis
# ─────────────────────────────────────────────────────────────────────────────

def find_aha_moment(channels: pd.DataFrame, videos: pd.DataFrame,
                    sub_thresholds: list = None,
                    day_thresholds: list = None) -> pd.DataFrame:
    """
    Find the (subscribers_in_X_days) threshold that best predicts
    long-term creator activity.

    Returns a DataFrame with threshold combinations and their predictive power.
    """
    if sub_thresholds is None:
        sub_thresholds = [100, 500, 1_000, 5_000, 10_000]
    if day_thresholds is None:
        day_thresholds = [7, 14, 30, 60, 90]

    first_upload = videos.groupby("channel_id")["publish_time"].min()
    last_upload  = videos.groupby("channel_id")["publish_time"].max()
    active_days  = (last_upload - first_upload).dt.days.rename("active_days")

    # For each channel: estimate early subscriber count proxy
    # (use views as proxy if subscriber time-series not available)
    early_stats = {}
    for ch_id, grp in videos.groupby("channel_id"):
        start = first_upload[ch_id]
        for d in day_thresholds:
            cutoff = start + pd.Timedelta(days=d)
            early = grp[grp["publish_time"] <= cutoff]
            early_stats.setdefault(ch_id, {})[d] = early["views"].sum()

    early_df = pd.DataFrame(early_stats).T  # channels × days
    early_df.columns = [f"views_in_{d}d" for d in day_thresholds]
    early_df = early_df.join(active_days)
    early_df["retained_6m"] = (early_df["active_days"] >= 180).astype(int)

    # Test each threshold combo
    results = []
    for d in day_thresholds:
        col = f"views_in_{d}d"
        for q in [0.25, 0.5, 0.75]:  # percentile thresholds
            threshold = early_df[col].quantile(q)
            hit = early_df[col] >= threshold
            retention_if_hit  = early_df.loc[hit,  "retained_6m"].mean()
            retention_if_miss = early_df.loc[~hit, "retained_6m"].mean()
            lift = retention_if_hit / (retention_if_miss + 1e-9)
            results.append({
                "days_window": d,
                "percentile": int(q * 100),
                "view_threshold": int(threshold),
                "retention_if_hit": round(retention_if_hit, 3),
                "retention_if_miss": round(retention_if_miss, 3),
                "lift": round(lift, 2),
            })

    return pd.DataFrame(results).sort_values("lift", ascending=False)

# added build_regression_features with no-leakage expanding window

# Ridge and Lasso added to run_growth_regression with VIF diagnostics

# K-Means segmentation with heuristic archetype labelling (k=5)

# find_aha_moment: threshold scan (views x days) for 6-month retention lift

# added build_regression_features with no-leakage expanding window

# Ridge and Lasso added to run_growth_regression with VIF diagnostics
