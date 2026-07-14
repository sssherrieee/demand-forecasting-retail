"""Feature engineering for the XGBoost demand model: lags, rolling stats,
calendar/price passthroughs, categorical encoding, and recursive forecasting.
"""

import numpy as np
import pandas as pd

LAGS = [7, 14, 28]
ROLL_WINDOWS = [7, 28]
MAX_WINDOW = max(LAGS + ROLL_WINDOWS)

FEATURE_COLS = [
    "lag_7", "lag_14", "lag_28",
    "roll_mean_7", "roll_std_7", "roll_mean_28", "roll_std_28",
    "wday", "month", "is_event", "snap", "sell_price", "price_change_flag",
    "store_enc", "item_enc",
]


def add_lag_rolling_features(df):
    """Add lag and rolling mean/std sales features per store-item series.

    Rolling stats are computed on sales shifted by 1 day, so a row never
    sees its own day's actual value.
    """
    df = df.sort_values(["store_id", "item_id", "date"]).copy()
    grp = df.groupby(["store_id", "item_id"])["sales"]
    for lag in LAGS:
        df[f"lag_{lag}"] = grp.shift(lag)
    for window in ROLL_WINDOWS:
        df[f"roll_mean_{window}"] = grp.transform(lambda s: s.shift(1).rolling(window).mean())
        df[f"roll_std_{window}"] = grp.transform(lambda s: s.shift(1).rolling(window).std())
    return df


def encode_categoricals(df, store_map=None, item_map=None):
    """Label-encode store_id and item_id. Pass in maps fit on training data to encode new data consistently."""
    df = df.copy()
    if store_map is None:
        store_map = {v: i for i, v in enumerate(sorted(df["store_id"].unique()))}
    if item_map is None:
        item_map = {v: i for i, v in enumerate(sorted(df["item_id"].unique()))}
    df["store_enc"] = df["store_id"].map(store_map)
    df["item_enc"] = df["item_id"].map(item_map)
    return df, store_map, item_map


def build_training_table(df, store_map=None, item_map=None):
    """Build the supervised feature table from a long sales dataframe.

    Drops rows without a full lag/rolling history (the first MAX_WINDOW days
    of each series).
    """
    df = add_lag_rolling_features(df)
    df, store_map, item_map = encode_categoricals(df, store_map, item_map)
    df["is_event"] = df["is_event"].astype(int)
    df["price_change_flag"] = df["price_change_flag"].astype(int)
    df = df.dropna(subset=FEATURE_COLS)
    return df, store_map, item_map


def iterative_forecast(model, sales_history_wide, exog_df, store_map, item_map, horizon_dates):
    """Recursively forecast `horizon_dates` day by day.

    Each day's lag/rolling features are built from actual history plus
    previously predicted days, since the model needs lag_7/lag_14 values
    that fall inside the forecast horizon itself.

    sales_history_wide: DataFrame indexed by date, columns=(store_id, item_id),
        containing at least the last MAX_WINDOW days of known sales.
    exog_df: long dataframe with calendar/price columns for the horizon dates.
    Returns: long dataframe of store_id, item_id, date, sales_pred.
    """
    history = sales_history_wide.copy()
    series_cols = history.columns
    store_arr = np.array([c[0] for c in series_cols])
    item_arr = np.array([c[1] for c in series_cols])
    store_enc = np.array([store_map[s] for s in store_arr])
    item_enc = np.array([item_map[i] for i in item_arr])

    preds = []
    for d in horizon_dates:
        exog_d = (
            exog_df[exog_df["date"] == d]
            .set_index(["store_id", "item_id"])
            .loc[list(series_cols)]
        )

        X = pd.DataFrame({
            "lag_7": history.iloc[-7].values,
            "lag_14": history.iloc[-14].values,
            "lag_28": history.iloc[-28].values,
            "roll_mean_7": history.iloc[-7:].mean().values,
            "roll_std_7": history.iloc[-7:].std().values,
            "roll_mean_28": history.iloc[-28:].mean().values,
            "roll_std_28": history.iloc[-28:].std().values,
            "wday": exog_d["wday"].values,
            "month": exog_d["month"].values,
            "is_event": exog_d["is_event"].astype(int).values,
            "snap": exog_d["snap"].values,
            "sell_price": exog_d["sell_price"].values,
            "price_change_flag": exog_d["price_change_flag"].astype(int).values,
            "store_enc": store_enc,
            "item_enc": item_enc,
        })[FEATURE_COLS]

        pred = np.clip(model.predict(X), 0, None)
        preds.append(pd.DataFrame({
            "store_id": store_arr, "item_id": item_arr, "date": d, "sales_pred": pred,
        }))
        history.loc[d] = pred

    return pd.concat(preds, ignore_index=True)
