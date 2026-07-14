"""Baseline forecasting models: naive, moving average, and seasonal naive.

Each model forecasts a fixed horizon from a single origin (the last training
date) using only training history — no holdout values are referenced, so a
lag shorter than the horizon is handled by tiling the last observed cycle
forward rather than reading into the future.
"""

import numpy as np
from xgboost import XGBRegressor

from features import FEATURE_COLS


def train_xgboost(train_df, params):
    """Fit an XGBoost regressor on the engineered feature table."""
    model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=-1,
        **params,
    )
    model.fit(train_df[FEATURE_COLS], train_df["sales"])
    return model


def tile_forecast(history, horizon, lag):
    """Repeat the last `lag` observed values to cover `horizon` future steps."""
    pattern = np.asarray(history)[-lag:]
    reps = int(np.ceil(horizon / lag))
    return np.tile(pattern, reps)[:horizon]


def naive_weekly_forecast(history, horizon=28, lag=7):
    """Same weekday last week, tiled across the horizon."""
    return tile_forecast(history, horizon, lag)


def moving_average_forecast(history, horizon=28, window=28):
    """Flat forecast at the trailing `window`-day mean."""
    avg = np.asarray(history)[-window:].mean()
    return np.full(horizon, avg)


def seasonal_naive_forecast(history, horizon=28, lag=364, fallback_lag=28):
    """Same period last year (lag=364), falling back to lag=28 for short histories."""
    history = np.asarray(history)
    use_lag = lag if len(history) >= lag else fallback_lag
    return tile_forecast(history, horizon, use_lag)
