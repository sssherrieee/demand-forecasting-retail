"""Build the modeling table for the FOODS_3 / CA_1+TX_1 demand forecasting scope.

Filters the M5 dataset down to the top 150 SKUs (by total units sold) in
category FOODS_3 across stores CA_1 and TX_1, melts the wide sales format
into a tidy long table, and joins calendar and price data.
"""

import numpy as np
import pandas as pd

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

DEPT = "FOODS_3"
STORES = ["CA_1", "TX_1"]
STORE_SNAP_COL = {"CA_1": "snap_CA", "TX_1": "snap_TX"}
START_DATE = "2013-01-01"
N_SKUS = 150


def load_raw(raw_dir=RAW_DIR):
    """Load the three raw M5 CSVs."""
    sales = pd.read_csv(f"{raw_dir}/sales_train_evaluation.csv")
    calendar = pd.read_csv(f"{raw_dir}/calendar.csv", parse_dates=["date"])
    prices = pd.read_csv(f"{raw_dir}/sell_prices.csv")
    return sales, calendar, prices


def get_scope_day_columns(calendar, sales, start_date=START_DATE):
    """Return the d_* columns present in `sales` whose calendar date is on/after start_date."""
    in_scope = calendar.loc[calendar["date"] >= start_date, "d"]
    return [d for d in in_scope.tolist() if d in sales.columns]


def get_top_skus(sales, calendar, dept=DEPT, stores=STORES, start_date=START_DATE, n=N_SKUS):
    """Return the top-n item_ids by total units sold within scope."""
    day_cols = get_scope_day_columns(calendar, sales, start_date)
    scoped = sales[(sales["dept_id"] == dept) & (sales["store_id"].isin(stores))]
    totals = scoped.groupby("item_id")[day_cols].sum().sum(axis=1)
    return totals.sort_values(ascending=False).head(n).index.tolist()


def melt_to_long(sales, calendar, item_ids, stores=STORES, start_date=START_DATE):
    """Melt wide d_* sales columns into a tidy date | store_id | item_id | sales table."""
    day_cols = get_scope_day_columns(calendar, sales, start_date)
    scoped = sales[sales["item_id"].isin(item_ids) & sales["store_id"].isin(stores)]
    id_cols = ["item_id", "store_id"]
    long_df = scoped[id_cols + day_cols].melt(
        id_vars=id_cols, var_name="d", value_name="sales"
    )
    date_map = calendar.set_index("d")["date"]
    long_df["date"] = long_df["d"].map(date_map)
    return long_df.drop(columns="d")


def add_calendar_features(df, calendar):
    """Join weekday, month, event, and SNAP features from the calendar table."""
    cal_cols = [
        "date", "wm_yr_wk", "weekday", "wday", "month", "year",
        "event_name_1", "event_type_1", "event_name_2", "event_type_2",
        "snap_CA", "snap_TX",
    ]
    df = df.merge(calendar[cal_cols], on="date", how="left")
    df["snap"] = df.apply(lambda r: r[STORE_SNAP_COL[r["store_id"]]], axis=1)
    df = df.drop(columns=["snap_CA", "snap_TX"])
    df["is_event"] = df["event_name_1"].notna() | df["event_name_2"].notna()
    return df


def add_price_features(df, prices):
    """Join weekly sell prices, forward-fill to daily, and flag price changes."""
    df = df.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    df = df.sort_values(["store_id", "item_id", "date"])
    df["sell_price"] = df.groupby(["store_id", "item_id"])["sell_price"].ffill().bfill()
    prev_price = df.groupby(["store_id", "item_id"])["sell_price"].shift(1)
    df["price_change_flag"] = (df["sell_price"] != prev_price) & prev_price.notna()
    return df


def run_quality_checks(df):
    """Print data-quality diagnostics: missingness, duplicates, date gaps, zero-sales rate."""
    print("--- Data quality checks ---")
    print("Missing values per column:")
    missing = df.isna().sum()
    print(missing[missing > 0] if missing.any() else "  none")

    dupes = df.duplicated(subset=["date", "store_id", "item_id"]).sum()
    print(f"Duplicate (date, store_id, item_id) rows: {dupes}")

    n_series = df.groupby(["store_id", "item_id"]).ngroups
    expected_days = df["date"].nunique()
    counts = df.groupby(["store_id", "item_id"]).size()
    incomplete = (counts != expected_days).sum()
    print(f"Series: {n_series}, expected days/series: {expected_days}, "
          f"series with missing dates: {incomplete}")

    zero_days = (df["sales"] == 0).sum()
    print(f"Zero-sales days: {zero_days} ({zero_days / len(df):.1%} of rows)")


def main():
    """Build the modeling table end to end and save it to data/processed/."""
    print("Loading raw data...")
    sales, calendar, prices = load_raw()

    print(f"Selecting top {N_SKUS} SKUs in {DEPT} across {STORES} from {START_DATE}...")
    top_skus = get_top_skus(sales, calendar)

    print("Melting to long format...")
    df = melt_to_long(sales, calendar, top_skus)

    print("Joining calendar features...")
    df = add_calendar_features(df, calendar)

    print("Joining price features...")
    df = add_price_features(df, prices)

    run_quality_checks(df)

    df = df.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)
    out_path = f"{PROCESSED_DIR}/modeling_table.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved {out_path} — {len(df):,} rows, {df.shape[1]} columns")


if __name__ == "__main__":
    main()
