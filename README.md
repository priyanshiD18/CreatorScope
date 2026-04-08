# CreatorScope
### Understanding What Makes Content Creators Grow on Digital Platforms
*A Product Analytics Case Study*

---

> "The creator economy is projected to approach $480B+ by 2027 (Goldman Sachs Research, 2023) but we still don't have
> great answers to basic questions: Does upload frequency actually matter? What separates
> a creator who grows from one who stalls? If a platform changes its recommendation
> algorithm, how do you measure the causal impact on creator growth?
> I built CreatorScope to explore these questions with data."

---

## What This Project Demonstrates

| Skill | Where |
|---|---|
| A/B Testing & Power Analysis | [Part 3](notebooks/03_ab_testing_causal.py) — full experiment design, 2×2 ANOVA |
| Causal Inference | [Part 3](notebooks/03_ab_testing_causal.py) — propensity score matching, DiD, CausalImpact |
| KPI Framework Design | [Part 1](notebooks/01_kpi_framework.py) — two-sided platform metrics |
| Regression & Statistical Modeling | [Part 2](notebooks/02_cohort_growth_analysis.py) — OLS, Ridge, Lasso, VIF |
| Cohort Analysis | [Part 2](notebooks/02_cohort_growth_analysis.py) — survival curves, lifecycle analysis |
| Time-Series Forecasting | [Part 4](notebooks/04_forecasting.py) — ARIMA, Prophet, LSTM |
| Predictive Modeling | [Part 4](notebooks/04_forecasting.py) — XGBoost + SHAP, churn model |
| Anomaly Detection | [Part 4](notebooks/04_forecasting.py) — Isolation Forest, Z-score |
| Creator Segmentation | [Part 2](notebooks/02_cohort_growth_analysis.py) — K-Means, PCA |
| SQL Analysis | [sql/queries/](sql/queries/) — ecosystem KPIs, power-law, Gini |
| Automated Dashboard | [dashboard/app.py](dashboard/app.py) — Streamlit, 5 interactive tabs |
| Product Recommendations | [docs/product_insights.md](docs/product_insights.md) — written analysis |

---

## Project Architecture

```
CreatorScope/
│
├── Part 1: Creator Ecosystem KPIs       → notebooks/01_kpi_framework.py
├── Part 2: Cohort & Growth Analysis     → notebooks/02_cohort_growth_analysis.py
├── Part 3: Experimentation & Causal     → notebooks/03_ab_testing_causal.py
├── Part 4: Growth Forecasting & ML      → notebooks/04_forecasting.py
├── Part 5: Interactive Dashboard        → dashboard/app.py
├── Part 6: Product Insights Doc         → docs/product_insights.md
│
├── src/
│   ├── data_loader.py        # Kaggle dataset ingestion + synthetic data
│   ├── kpi_metrics.py        # Creator & platform KPI computations
│   ├── cohort_analysis.py    # Cohort survival, regression, segmentation, aha-moment
│   ├── experimentation.py    # A/B testing, power analysis, DiD, CausalImpact
│   └── models/
│       └── forecasting.py    # ARIMA, Prophet, LSTM, XGBoost, churn, anomaly
│
├── sql/
│   ├── schema.sql            # PostgreSQL schema (channels, videos, analytics)
│   └── queries/
│       ├── 01_ecosystem_analysis.sql
│       └── 02_creator_metrics.sql
│
├── docker-compose.yml        # PostgreSQL + Streamlit
├── Dockerfile
└── .github/workflows/ci.yml  # Lint → test → docker build
```

---

## Quick Start

### Option A — Docker (recommended)

```bash
git clone https://github.com/YOUR_USERNAME/creatorscope
cd creatorscope
docker-compose up
```

Dashboard → http://localhost:8501

### Option B — Local

```bash
pip install -r requirements.txt

# Optional: add Kaggle data to data/raw/ (see Data section below)
# Without it, synthetic data is used automatically

# Run any analysis part
python notebooks/01_kpi_framework.py
python notebooks/02_cohort_growth_analysis.py
python notebooks/03_ab_testing_causal.py
python notebooks/04_forecasting.py

# Launch dashboard
streamlit run dashboard/app.py
```

---

## Data

All datasets are freely available on Kaggle. Place CSVs in `data/raw/`:

| File | Kaggle Dataset | Use |
|---|---|---|
| `global_yt_stats.csv` | Global YouTube Statistics 2023 | Channel-level analysis |
| `trending_videos.csv` | YouTube Trending Video Dataset | Trending impact, engagement |
| `video_analytics.csv` | Ken Jee YouTube Data | Per-video creator analytics |

**Without Kaggle data:** the project auto-generates realistic synthetic data
(500 channels, 15,000 videos with power-law distributions) so everything runs out of the box.

---

## Key Findings

### The "Aha Moment"
Creators who reach ~2,000 views within their first 30 days are **3.2× more likely**
to still be active at 6 months — a platform lever for early-stage creator investment.

### Consistency Beats Frequency
Upload consistency score has **2.1× the regression coefficient** of raw upload count
for 6-month growth (after controlling for category and audience size).

### Trending is a Catalyst, Not a Solution
Propensity score matching shows +34% view lift post-trending, but the effect decays
to baseline within 14–18 days for creators who don't convert exposure into community.

### Community Builders Win Long-Term
Despite lower raw views, "Community Builders" (high engagement, consistent posting)
show **45% lower churn** than "Algorithm Surfers" — arguing for engagement-weighted
platform metrics over raw view counts.

### Churn is Predictable 30+ Days Early (AUC = 0.87)
Gradient Boosting churn model with SHAP explanations surfaces `gap_trend` and
`engagement_trend` as the dominant signals — enabling proactive creator interventions.

---

## Dashboard (5 Tabs)

| Tab | What It Shows |
|---|---|
| Ecosystem Overview | Lorenz curve, Gini, category leaderboard, geographic map, tier engagement |
| Creator Deep Dive | Growth trajectory, segment membership, anomaly-flagged videos |
| Experiment Toolkit | Interactive power analysis — input effect size, get sample size + runtime |
| Growth Forecast | ARIMA / Prophet forecast with confidence bands + seasonality decomposition |
| Churn Risk Radar | Ranked churn table, risk factor breakdown, intervention playbook |

---

## Product Insights

Full written analysis: **[docs/product_insights.md](docs/product_insights.md)**

Includes KPI framework, 5 major findings with product implications, 3 recommended
A/B experiments, and identified data gaps.

---

## Stack

`Python` · `PostgreSQL` · `pandas` · `scikit-learn` · `XGBoost` · `statsmodels`
· `Prophet` · `PyTorch (LSTM)` · `SHAP` · `Streamlit` · `Plotly` · `Docker`
· `GitHub Actions CI`
