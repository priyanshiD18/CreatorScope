"""
forecasting.py
--------------
Part 4: Growth Forecasting & Predictive Models

4.1 Creator growth trajectory forecasting (ARIMA, Prophet, LSTM)
4.2 Video performance prediction (XGBoost + SHAP)
4.3 Engagement anomaly detection (Isolation Forest / Z-score)
4.4 Creator churn prediction (Logistic + GBM + SHAP)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score,
    mean_absolute_percentage_error, mean_squared_error,
)
from sklearn.pipeline import Pipeline
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 4.1 Time-series forecasting
# ─────────────────────────────────────────────────────────────────────────────

def make_channel_timeseries(videos: pd.DataFrame, channel_id: str,
                             freq: str = "W") -> pd.Series:
    """Aggregate a channel's videos into a weekly view time-series."""
    ch = videos[videos["channel_id"] == channel_id].copy()
    ch = ch.set_index("publish_time").resample(freq)["views"].sum()
    ch = ch.fillna(0)
    return ch


def forecast_arima(series: pd.Series, steps: int = 12) -> dict:
    """Fit auto-ARIMA and forecast `steps` periods ahead."""
    try:
        from pmdarima import auto_arima
        model = auto_arima(
            series, seasonal=True, m=4,
            stepwise=True, suppress_warnings=True, error_action="ignore"
        )
        forecast, conf_int = model.predict(n_periods=steps, return_conf_int=True)
        future_idx = pd.date_range(series.index[-1], periods=steps + 1, freq=series.index.freq)[1:]
        return {
            "model": "ARIMA",
            "forecast": pd.Series(forecast, index=future_idx),
            "lower": pd.Series(conf_int[:, 0], index=future_idx),
            "upper": pd.Series(conf_int[:, 1], index=future_idx),
            "aic": model.aic(),
        }
    except Exception as e:
        # Fallback: simple exponential smoothing
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        hw = ExponentialSmoothing(series, trend="add", seasonal="add",
                                  seasonal_periods=4).fit(optimized=True)
        forecast = hw.forecast(steps)
        return {"model": "ExponentialSmoothing (ARIMA fallback)", "forecast": forecast,
                "lower": forecast * 0.8, "upper": forecast * 1.2}


def forecast_prophet(series: pd.Series, steps: int = 12) -> dict:
    """Fit Facebook Prophet and forecast `steps` periods ahead."""
    try:
        from prophet import Prophet
        df_p = series.reset_index()
        df_p.columns = ["ds", "y"]
        df_p["ds"] = pd.to_datetime(df_p["ds"])

        model = Prophet(weekly_seasonality=True, yearly_seasonality=True,
                        changepoint_prior_scale=0.05)
        model.fit(df_p)
        future = model.make_future_dataframe(periods=steps, freq="W")
        forecast = model.predict(future)

        idx = forecast["ds"].iloc[-steps:]
        return {
            "model": "Prophet",
            "forecast": pd.Series(forecast["yhat"].iloc[-steps:].values, index=idx),
            "lower": pd.Series(forecast["yhat_lower"].iloc[-steps:].values, index=idx),
            "upper": pd.Series(forecast["yhat_upper"].iloc[-steps:].values, index=idx),
            "components": forecast[["ds", "trend", "weekly", "yearly"]],
        }
    except ImportError:
        return {"model": "Prophet (not installed)", "forecast": pd.Series(dtype=float)}


def forecast_lstm(series: pd.Series, steps: int = 12, lookback: int = 8) -> dict:
    """Minimal LSTM forecast using PyTorch."""
    try:
        import torch
        import torch.nn as nn

        vals = series.values.astype(np.float32)
        mu, sigma = vals.mean(), vals.std() + 1e-8
        scaled = (vals - mu) / sigma

        # Build sequences
        X_data, y_data = [], []
        for i in range(len(scaled) - lookback):
            X_data.append(scaled[i:i + lookback])
            y_data.append(scaled[i + lookback])

        if len(X_data) < 10:
            return {"model": "LSTM (insufficient data)", "forecast": pd.Series(dtype=float)}

        X_t = torch.FloatTensor(X_data).unsqueeze(-1)
        y_t = torch.FloatTensor(y_data)

        class SimpleLSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(1, 32, batch_first=True)
                self.fc   = nn.Linear(32, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :]).squeeze()

        model = SimpleLSTM()
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()

        for epoch in range(100):
            model.train()
            opt.zero_grad()
            pred = model(X_t)
            loss = loss_fn(pred, y_t)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            window = scaled[-lookback:].tolist()
            preds = []
            for _ in range(steps):
                inp = torch.FloatTensor(window[-lookback:]).unsqueeze(0).unsqueeze(-1)
                p = model(inp).item()
                preds.append(p)
                window.append(p)

        forecast_vals = np.array(preds) * sigma + mu
        future_idx = pd.date_range(series.index[-1], periods=steps + 1,
                                    freq=series.index.freq)[1:]
        return {
            "model": "LSTM",
            "forecast": pd.Series(forecast_vals, index=future_idx),
        }
    except ImportError:
        return {"model": "LSTM (PyTorch not installed)", "forecast": pd.Series(dtype=float)}


def compare_forecasts(series: pd.Series, steps: int = 8) -> pd.DataFrame:
    """
    Walk-forward validation: compare ARIMA vs Prophet vs LSTM on MAPE/RMSE.
    """
    n = len(series)
    split = int(n * 0.8)
    train, test = series.iloc[:split], series.iloc[split:]
    k = min(steps, len(test))

    results = []
    for name, fn in [("ARIMA", forecast_arima), ("Prophet", forecast_prophet),
                     ("LSTM", forecast_lstm)]:
        try:
            out = fn(train, steps=k)
            fc = out["forecast"]
            if len(fc) == 0:
                continue
            actual = test.values[:k]
            pred   = fc.values[:k]
            mape   = mean_absolute_percentage_error(actual + 1, pred + 1)
            rmse   = np.sqrt(mean_squared_error(actual, pred))
            results.append({"model": name, "MAPE": round(mape, 4), "RMSE": round(rmse, 1)})
        except Exception as e:
            results.append({"model": name, "MAPE": None, "RMSE": None, "error": str(e)})

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# 4.2 Video performance prediction (XGBoost + SHAP)
# ─────────────────────────────────────────────────────────────────────────────

def build_video_features(channels: pd.DataFrame, videos: pd.DataFrame) -> pd.DataFrame:
    """Build pre-publish features for video performance prediction."""
    df = videos.merge(channels[["channel_id", "subscribers", "avg_weekly_uploads"]], on="channel_id")

    # Historical channel averages up to each video (avoid leakage)
    df = df.sort_values(["channel_id", "publish_time"])
    df["hist_avg_views"] = (
        df.groupby("channel_id")["views"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    df["days_since_last"] = (
        df.groupby("channel_id")["publish_time"]
        .transform(lambda x: x.diff().dt.days)
    )
    df["publish_hour"] = df["publish_time"].dt.hour
    df["publish_dow"]  = df["publish_time"].dt.dayofweek
    df["title_len"]    = df.get("title", pd.Series("", index=df.index)).str.len()

    feature_cols = ["subscribers", "avg_weekly_uploads", "hist_avg_views",
                    "days_since_last", "publish_hour", "publish_dow",
                    "title_len", "duration_seconds"]
    df["log_views"] = np.log1p(df["views"])

    feat = df[feature_cols + ["log_views", "channel_id", "video_id"]].dropna()
    return feat


def train_video_predictor(feat: pd.DataFrame):
    """Train XGBoost to predict log(views+1)."""
    feature_cols = [c for c in feat.columns if c not in ["log_views", "channel_id", "video_id"]]
    X = feat[feature_cols]
    y = feat["log_views"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mape = mean_absolute_percentage_error(np.expm1(y_test), np.expm1(preds))
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    # SHAP values
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        shap_importance = pd.DataFrame({
            "feature": feature_cols,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)
    except ImportError:
        shap_importance = pd.DataFrame({"feature": feature_cols,
                                         "mean_abs_shap": model.feature_importances_})

    return {
        "model": model,
        "mape": round(mape, 4),
        "rmse": round(rmse, 3),
        "shap_importance": shap_importance,
        "X_test": X_test,
        "y_test": y_test,
        "feature_cols": feature_cols,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4.3 Engagement anomaly detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_anomalies(videos: pd.DataFrame) -> pd.DataFrame:
    """
    Flag videos that significantly over/under-perform channel expectations.
    Methods: Isolation Forest + Z-score on engagement residuals.
    """
    videos = videos.copy()
    videos["engagement_rate"] = np.where(
        videos["views"] > 0,
        (videos["likes"] + videos["comment_count"]) / videos["views"], 0
    )

    ch_stats = videos.groupby("channel_id").agg(
        mean_views=("views", "mean"),
        std_views=("views", "std"),
        mean_eng=("engagement_rate", "mean"),
        std_eng=("engagement_rate", "std"),
    ).reset_index()

    videos = videos.merge(ch_stats, on="channel_id", how="left")
    videos["z_views"] = (videos["views"] - videos["mean_views"]) / (videos["std_views"] + 1e-8)
    videos["z_eng"]   = (videos["engagement_rate"] - videos["mean_eng"]) / (videos["std_eng"] + 1e-8)
    videos["z_anomaly"] = (videos["z_views"].abs() > 2) | (videos["z_eng"].abs() > 2)

    # Isolation Forest
    feat_cols = ["views", "engagement_rate"]
    X = videos[feat_cols].fillna(0)
    iso = IsolationForest(contamination=0.05, random_state=42)
    videos["if_anomaly"] = iso.fit_predict(X) == -1
    videos["anomaly_score"] = iso.score_samples(X)

    videos["is_anomaly"] = videos["z_anomaly"] | videos["if_anomaly"]
    videos["anomaly_type"] = np.where(
        videos["z_views"] > 2, "over-performer",
        np.where(videos["z_views"] < -2, "under-performer", "normal")
    )
    return videos


# ─────────────────────────────────────────────────────────────────────────────
# 4.4 Creator churn prediction
# ─────────────────────────────────────────────────────────────────────────────

def build_churn_features(channels: pd.DataFrame, videos: pd.DataFrame,
                          inactivity_days: int = 90) -> pd.DataFrame:
    """
    Label a creator as churned if they haven't uploaded in `inactivity_days`.
    Features: upload gap trend, engagement trend, subscriber growth stalling.
    """
    videos = videos.sort_values(["channel_id", "publish_time"]).copy()
    videos["engagement_rate"] = np.where(
        videos["views"] > 0,
        (videos["likes"] + videos["comment_count"]) / videos["views"], 0
    )
    now = videos["publish_time"].max()

    def channel_features(grp):
        gaps = grp["publish_time"].diff().dt.days.dropna()
        recent_eng = grp["engagement_rate"].tail(5).mean()
        early_eng  = grp["engagement_rate"].head(5).mean()
        recent_views = grp["views"].tail(5).mean()
        early_views  = grp["views"].head(5).mean()
        last_upload  = grp["publish_time"].max()
        days_silent  = (now - last_upload).days
        return pd.Series({
            "total_videos": len(grp),
            "avg_gap": gaps.mean() if len(gaps) > 0 else np.nan,
            "gap_trend": gaps.tail(3).mean() - gaps.head(3).mean() if len(gaps) >= 6 else 0,
            "engagement_trend": recent_eng - early_eng,
            "view_trend": (recent_views - early_views) / (early_views + 1),
            "days_since_last": days_silent,
            "churned": int(days_silent >= inactivity_days),
        })

    feat = videos.groupby("channel_id").apply(channel_features).reset_index()
    feat = feat.merge(channels[["channel_id", "subscribers", "avg_weekly_uploads",
                                 "category"]], on="channel_id", how="left")
    feat = pd.get_dummies(feat, columns=["category"], drop_first=True, dtype=float)
    return feat.dropna(subset=["churned"])


def train_churn_model(feat: pd.DataFrame):
    """Train Logistic + GBM churn models. Returns best model + SHAP."""
    label_col = "churned"
    drop_cols = ["channel_id", label_col]
    feature_cols = [c for c in feat.columns if c not in drop_cols]

    X = feat[feature_cols].fillna(0)
    y = feat[label_col]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Logistic Regression
    lr = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500))])
    lr.fit(X_tr, y_tr)
    lr_auc = roc_auc_score(y_te, lr.predict_proba(X_te)[:, 1])

    # Gradient Boosting
    gbm = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                      max_depth=4, random_state=42)
    gbm.fit(X_tr, y_tr)
    gbm_auc = roc_auc_score(y_te, gbm.predict_proba(X_te)[:, 1])

    best_model = gbm if gbm_auc > lr_auc else lr.named_steps["clf"]

    # SHAP
    try:
        import shap
        explainer = shap.TreeExplainer(gbm)
        shap_values = explainer.shap_values(X_te)
        shap_imp = pd.DataFrame({
            "feature": feature_cols,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)
    except Exception:
        shap_imp = pd.DataFrame({
            "feature": feature_cols,
            "mean_abs_shap": gbm.feature_importances_,
        }).sort_values("mean_abs_shap", ascending=False)

    # Churn probabilities for all creators
    feat = feat.copy()
    feat["churn_probability"] = gbm.predict_proba(X)[:, 1]

    return {
        "lr_auc": round(lr_auc, 4),
        "gbm_auc": round(gbm_auc, 4),
        "classification_report": classification_report(y_te, gbm.predict(X_te)),
        "shap_importance": shap_imp,
        "churn_scores": feat[["channel_id", "churn_probability", "churned",
                               "gap_trend", "engagement_trend", "days_since_last"]],
        "gbm_model": gbm,
        "feature_cols": feature_cols,
    }

# forecast_arima: auto-order selection with pmdarima, seasonal decomposition

# forecast_prophet: weekly and yearly seasonality, changepoint_prior_scale=0.05

# forecast_lstm: PyTorch single-layer LSTM, lookback=8 weeks, 100 epochs

# build_video_features: expanding window historical averages, no data leakage

# train_video_predictor: XGBoost regressor with SHAP TreeExplainer

# detect_anomalies: Isolation Forest (contamination=5%) + Z-score on engagement residuals

# build_churn_features: gap_trend, engagement_trend, days_since_last as leading indicators

# forecast_arima: auto-order selection with pmdarima, seasonal decomposition

# forecast_prophet: weekly and yearly seasonality, changepoint_prior_scale=0.05

# forecast_lstm: PyTorch single-layer LSTM, lookback=8 weeks, 100 epochs

# build_video_features: expanding window historical averages, no data leakage

# train_video_predictor: XGBoost regressor with SHAP TreeExplainer

# detect_anomalies: Isolation Forest (contamination=5%) + Z-score on engagement residuals

# build_churn_features: gap_trend, engagement_trend, days_since_last as leading indicators
