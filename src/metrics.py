"""Forecast accuracy metrics: RMSE, MAE, and WAPE."""

import numpy as np


def rmse(y_true, y_pred):
    """Root mean squared error."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    """Mean absolute error."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def wape(y_true, y_pred):
    """Weighted absolute percentage error: sum(|error|) / sum(|actual|).

    Scale-aware and robust to zero-sales days, unlike MAPE which is undefined
    when actuals are zero. Primary metric for this project.
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true).sum()
    if denom == 0:
        return float("nan")
    return float(np.abs(y_true - y_pred).sum() / denom)
