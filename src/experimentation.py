"""
experimentation.py
------------------
Part 3: A/B Testing, Causal Inference, Power Analysis
- Trending algorithm impact (DiD + propensity score matching)
- Full A/B experiment design document
- Multivariate 2×2 ANOVA simulation
- CausalImpact wrapper
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_ind, chi2_contingency, f_oneway
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.power import TTestIndPower
import warnings

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 3.1 Trending impact: DiD + Propensity Score Matching
# ─────────────────────────────────────────────────────────────────────────────

def trending_impact_did(channels: pd.DataFrame, videos: pd.DataFrame,
                        trending_channel_ids: list = None) -> dict:
    """
    Difference-in-differences estimate of the causal effect of trending
    on subscriber growth (proxied by view volume).

    If trending_channel_ids is None, we simulate by treating the top
    5% view-getters in the first half as 'treated'.
    """
    videos = videos.sort_values("publish_time").copy()
    midpoint = videos["publish_time"].median()

    # Pre/post view counts per channel
    pre  = videos[videos["publish_time"] < midpoint].groupby("channel_id")["views"].mean().rename("pre_views")
    post = videos[videos["publish_time"] >= midpoint].groupby("channel_id")["views"].mean().rename("post_views")
    df = pd.concat([pre, post], axis=1).dropna()

    # Treatment: top 5% by pre-period views (proxy for getting on trending)
    if trending_channel_ids is None:
        threshold = df["pre_views"].quantile(0.95)
        df["treated"] = (df["pre_views"] >= threshold).astype(int)
    else:
        df["treated"] = df.index.isin(trending_channel_ids).astype(int)

    df["log_pre"]  = np.log1p(df["pre_views"])
    df["log_post"] = np.log1p(df["post_views"])

    # OLS DiD: log_post ~ treated + log_pre (pre-trend control)
    model = smf.ols("log_post ~ treated + log_pre", data=df).fit()

    treated_post   = df.loc[df["treated"] == 1, "log_post"].mean()
    treated_pre    = df.loc[df["treated"] == 1, "log_pre"].mean()
    control_post   = df.loc[df["treated"] == 0, "log_post"].mean()
    control_pre    = df.loc[df["treated"] == 0, "log_pre"].mean()
    did_estimate   = (treated_post - treated_pre) - (control_post - control_pre)

    return {
        "did_estimate": round(did_estimate, 4),
        "treated_count": int(df["treated"].sum()),
        "control_count": int((df["treated"] == 0).sum()),
        "ols_summary": model.summary(),
        "treated_coef": model.params.get("treated", np.nan),
        "treated_pval": model.pvalues.get("treated", np.nan),
    }


def propensity_score_matching(channels: pd.DataFrame, videos: pd.DataFrame) -> pd.DataFrame:
    """
    Simple propensity score matching on channel features.
    Treated = channels that appear in trending snapshots.
    Returns matched DataFrame with ATT estimate.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    videos = videos.copy()
    videos["engagement_rate"] = np.where(
        videos["views"] > 0,
        (videos["likes"] + videos["comment_count"]) / videos["views"], 0
    )
    ch_stats = videos.groupby("channel_id").agg(
        avg_views=("views", "mean"),
        avg_engagement=("engagement_rate", "mean"),
        video_count=("video_id", "count"),
    ).reset_index()

    df = channels.merge(ch_stats, on="channel_id", how="left").fillna(0)

    # Simulate treatment: top-quintile engagement gets trending exposure
    df["treated"] = (df["avg_engagement"] >= df["avg_engagement"].quantile(0.8)).astype(int)

    feature_cols = ["subscribers", "avg_weekly_uploads", "avg_views", "avg_engagement", "video_count"]
    X = StandardScaler().fit_transform(df[feature_cols].fillna(0))
    lr = LogisticRegression(max_iter=500)
    lr.fit(X, df["treated"])
    df["propensity"] = lr.predict_proba(X)[:, 1]

    # Nearest-neighbour matching (greedy, 1:1)
    treated = df[df["treated"] == 1].copy()
    control = df[df["treated"] == 0].copy()
    matched_pairs = []
    used = set()
    for _, t_row in treated.iterrows():
        diffs = (control["propensity"] - t_row["propensity"]).abs()
        diffs = diffs[~diffs.index.isin(used)]
        if len(diffs) == 0:
            continue
        best_idx = diffs.idxmin()
        used.add(best_idx)
        matched_pairs.append({
            "treated_views": t_row["avg_views"],
            "control_views": control.loc[best_idx, "avg_views"],
            "pscore_treated": t_row["propensity"],
            "pscore_control": control.loc[best_idx, "propensity"],
        })

    matched_df = pd.DataFrame(matched_pairs)
    att = (matched_df["treated_views"] - matched_df["control_views"]).mean()
    t_stat, p_val = ttest_ind(matched_df["treated_views"], matched_df["control_views"])

    matched_df.attrs["ATT"] = att
    matched_df.attrs["p_value"] = p_val
    return matched_df


# ─────────────────────────────────────────────────────────────────────────────
# 3.2 Power analysis
# ─────────────────────────────────────────────────────────────────────────────

def power_analysis(baseline: float = 1.2, mde_pct: float = 0.10,
                   alpha: float = 0.05, power: float = 0.80,
                   std_estimate: float = 0.8) -> dict:
    """
    Calculate required sample size for the upload-frequency experiment.
    """
    effect_size = (baseline * mde_pct) / std_estimate  # Cohen's d
    analysis = TTestIndPower()
    n_per_group = analysis.solve_power(
        effect_size=effect_size, alpha=alpha, power=power, alternative="two-sided"
    )
    return {
        "baseline_uploads_per_week": baseline,
        "minimum_detectable_effect": baseline * mde_pct,
        "cohens_d": round(effect_size, 3),
        "n_per_group": int(np.ceil(n_per_group)),
        "total_n": int(np.ceil(n_per_group)) * 2,
        "alpha": alpha,
        "power": power,
    }


def experiment_runtime_days(n_total: int, daily_active_creators: int = 5_000,
                             allocation_pct: float = 1.0) -> int:
    """Estimate days needed to recruit n_total creators into the experiment."""
    return int(np.ceil(n_total / (daily_active_creators * allocation_pct)))


# ─────────────────────────────────────────────────────────────────────────────
# 3.3 Multivariate 2×2 ANOVA simulation
# ─────────────────────────────────────────────────────────────────────────────

def simulate_2x2_experiment(n_per_cell: int = 200, seed: int = 42) -> dict:
    """
    Simulate a 2×2 factorial experiment:
      Factor A: Thumbnail suggestion tool (0=off, 1=on)
      Factor B: Upload reminder notification (0=off, 1=on)
      DV: Weekly views per video
    """
    rng = np.random.default_rng(seed)
    base_views = 5_000
    a_effect   = 800    # +16%
    b_effect   = 600    # +12%
    ab_synergy = 400    # interaction bonus

    def cell(a, b, n):
        mu = base_views + a * a_effect + b * b_effect + a * b * ab_synergy
        return rng.normal(mu, 1500, n)

    a0b0 = cell(0, 0, n_per_cell)
    a1b0 = cell(1, 0, n_per_cell)
    a0b1 = cell(0, 1, n_per_cell)
    a1b1 = cell(1, 1, n_per_cell)

    df = pd.DataFrame({
        "views": np.concatenate([a0b0, a1b0, a0b1, a1b1]),
        "thumbnail_tool": [0]*n_per_cell + [1]*n_per_cell + [0]*n_per_cell + [1]*n_per_cell,
        "upload_reminder": [0]*n_per_cell*2 + [1]*n_per_cell*2,
    })

    # Two-way ANOVA via OLS
    model = smf.ols("views ~ C(thumbnail_tool) * C(upload_reminder)", data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    cell_means = df.groupby(["thumbnail_tool", "upload_reminder"])["views"].mean().unstack()

    return {
        "anova_table": anova_table,
        "cell_means": cell_means,
        "ols_summary": model.summary(),
        "data": df,
        "main_effect_A": float(a_effect),
        "main_effect_B": float(b_effect),
        "interaction": float(ab_synergy),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3.4 CausalImpact wrapper
# ─────────────────────────────────────────────────────────────────────────────

def simulate_causal_impact(n_pre: int = 60, n_post: int = 30, seed: int = 42) -> dict:
    """
    Simulate a time-series with a structural break and estimate
    causal impact using Bayesian structural time-series.

    Falls back to a manual DiD if causalimpact is not installed.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n_pre + n_post, freq="D")

    # Treated series: algorithm change on day n_pre causes +20% lift
    treated_pre  = rng.normal(1_000, 80, n_pre)
    treated_post = rng.normal(1_200, 80, n_post)  # causal lift
    y = np.concatenate([treated_pre, treated_post])

    # Control (unaffected category)
    control = rng.normal(950, 75, n_pre + n_post)

    ts = pd.DataFrame({"y": y, "control": control}, index=dates)

    try:
        from causalimpact import CausalImpact
        ci = CausalImpact(ts, [dates[0], dates[n_pre - 1]], [dates[n_pre], dates[-1]])
        summary = ci.summary()
        report  = ci.summary(output="report")
        return {"method": "CausalImpact", "summary": summary, "report": report, "ts": ts}
    except ImportError:
        # Manual DiD fallback
        pre_mean  = treated_pre.mean()
        post_mean = treated_post.mean()
        ctrl_pre  = control[:n_pre].mean()
        ctrl_post = control[n_pre:].mean()
        did = (post_mean - pre_mean) - (ctrl_post - ctrl_pre)
        t, p = ttest_ind(treated_post, treated_pre)
        return {
            "method": "DiD fallback (causalimpact not installed)",
            "did_estimate": round(did, 2),
            "pct_lift": round(did / pre_mean * 100, 2),
            "t_statistic": round(t, 3),
            "p_value": round(p, 4),
            "ts": ts,
        }

# propensity_score_matching: greedy 1:1 nearest-neighbour on channel features

# power_analysis and experiment_runtime_days added

# simulate_2x2_experiment: factorial design with thumbnail tool x upload reminder

# simulate_causal_impact: CausalImpact wrapper with DiD fallback if not installed

# propensity_score_matching: greedy 1:1 nearest-neighbour on channel features

# power_analysis and experiment_runtime_days added
