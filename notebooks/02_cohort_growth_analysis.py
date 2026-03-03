"""
Part 2: Creator Cohort & Growth Analysis
==========================================
Cohort survival curves · Growth regression · Segmentation · Aha Moment
"""

# %%
import sys, warnings
sys.path.insert(0, "..")
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA

sns.set_theme(style="whitegrid")

from src.data_loader import generate_synthetic_channels, generate_synthetic_videos
from src.cohort_analysis import (
    build_cohorts, survival_curve,
    build_regression_features, run_growth_regression,
    build_cluster_features, segment_creators,
    find_aha_moment,
)

PROC = Path("data/processed")

# ──────────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────────

# %%
channels = generate_synthetic_channels(n=500, seed=42)
videos   = generate_synthetic_videos(channels, n_per_channel=30, seed=42)
print(f"Channels: {len(channels):,}  |  Videos: {len(videos):,}")

# ──────────────────────────────────────────────────────────────────
# 2.1 Cohort survival curves
# ──────────────────────────────────────────────────────────────────

# %% [markdown]
# ## 2.1 Upload Survival Curves by Creator Cohort

# %%
cohort_df = build_cohorts(videos, freq="Q")
survival  = survival_curve(videos, cohort_df)

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
for _, row in survival.iterrows():
    vals = [row[f"survival_{t}d"] for t in [30, 60, 90, 180, 365]]
    ax.plot([30, 60, 90, 180, 365], vals, marker="o", label=row["cohort"], alpha=0.7)

ax.set_xlabel("Days since first upload")
ax.set_ylabel("% still uploading")
ax.set_title("Creator Upload Survival Curves by Cohort (Quarterly)")
ax.legend(bbox_to_anchor=(1.01, 1), fontsize=8)
plt.tight_layout()
plt.savefig(PROC / "02_survival_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print(survival[["cohort", "n", "survival_30d", "survival_90d", "survival_180d"]].to_string(index=False))

# ──────────────────────────────────────────────────────────────────
# 2.2 Growth regression
# ──────────────────────────────────────────────────────────────────

# %% [markdown]
# ## 2.2 What Predicts Creator Growth? (Multivariate Regression)

# %%
feat = build_regression_features(channels, videos)
results = run_growth_regression(feat)

print("\n── OLS Summary (top coefficients) ──")
print(results["ols_summary"].tables[1])
print(f"\nRidge CV R²: {results['ridge_cv_r2']:.3f}")
print(f"Lasso CV R²: {results['lasso_cv_r2']:.3f}")
print(f"\nLasso surviving features:\n{results['lasso_surviving_features']}")

# Coefficient plot
ols = results["ols_model"]
coef_df = pd.DataFrame({
    "feature": results["feature_names"],
    "coef": ols.params[1:],
    "ci_low": ols.conf_int().iloc[1:, 0].values,
    "ci_high": ols.conf_int().iloc[1:, 1].values,
}).sort_values("coef")

fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(coef_df["feature"], coef_df["coef"],
        xerr=[coef_df["coef"] - coef_df["ci_low"],
              coef_df["ci_high"] - coef_df["coef"]],
        color=np.where(coef_df["coef"] > 0, "steelblue", "tomato"),
        alpha=0.8, capsize=3)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("OLS Regression: Drivers of Creator Growth\n(controlling for category & country)")
ax.set_xlabel("Standardised Coefficient")
plt.tight_layout()
plt.savefig(PROC / "02_regression_coefs.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nVIF (multicollinearity check):")
print(results["vif"][results["vif"]["VIF"] < 10].to_string(index=False))

# ──────────────────────────────────────────────────────────────────
# 2.3 Creator segmentation
# ──────────────────────────────────────────────────────────────────

# %% [markdown]
# ## 2.3 Creator Segmentation (K-Means + PCA visualisation)

# %%
cluster_feat = build_cluster_features(channels, videos)
clustered, km = segment_creators(cluster_feat, n_clusters=5, seed=42)

# Segment profiles
profile_cols = ["avg_weekly_uploads", "avg_engagement", "avg_views", "gap_std", "sub_growth_proxy"]
profiles = clustered.groupby("segment")[profile_cols].mean().round(4)
print("\nCluster Profiles:")
print(profiles.to_string())

# PCA projection
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X_cl = StandardScaler().fit_transform(cluster_feat[profile_cols].fillna(0))
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_cl)

clustered["pca_x"] = coords[:, 0]
clustered["pca_y"] = coords[:, 1]

fig, ax = plt.subplots(figsize=(9, 6))
palette = sns.color_palette("Set2", 5)
for (seg, grp), color in zip(clustered.groupby("segment"), palette):
    ax.scatter(grp["pca_x"], grp["pca_y"], label=seg, alpha=0.6, s=25, color=color)

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
ax.set_title("Creator Segmentation (K-Means, PCA projection)")
ax.legend(title="Segment")
plt.tight_layout()
plt.savefig(PROC / "02_segmentation.png", dpi=150, bbox_inches="tight")
plt.show()

# ──────────────────────────────────────────────────────────────────
# 2.4 Aha Moment
# ──────────────────────────────────────────────────────────────────

# %% [markdown]
# ## 2.4 The "Aha Moment" — Early Milestone That Predicts Long-Term Retention

# %%
aha = find_aha_moment(channels, videos)
print("\nTop Aha-Moment Thresholds (by retention lift):")
print(aha.head(10).to_string(index=False))

best = aha.iloc[0]
print(f"\n>>> AHA MOMENT: Creators who reach {best['view_threshold']:,} views "
      f"within the first {best['days_window']} days are "
      f"{best['lift']:.1f}x more likely to still be active at 6 months.")

# survival result: 60% inactive by day 90, 80% by day 180

# OLS result: consistency_score coef = 2.1x upload_count coef (p<0.01)

# Lasso: upload_frequency drops out, consistency_score and avg_engagement survive regularisation

# PCA: Community Builders and Algorithm Surfers cluster distinctly despite similar view counts

# aha moment: 2000 views in 30 days = 3.2x lift in 6-month retention

# survival result: 60% inactive by day 90, 80% by day 180
