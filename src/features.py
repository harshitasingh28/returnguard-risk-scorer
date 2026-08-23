"""Feature engineering for the return-risk scorer.

All features here are available AT CHECKOUT TIME (before the order ships),
which is the only point a merchant can actually act on this prediction.
`past_return_rate` / `past_order_count` are computed upstream in the data
generator using strictly-prior orders only (no leakage).
"""
import pandas as pd

NUMERIC_FEATURES = [
    "order_value", "discount_pct", "num_items", "hour_of_day",
    "is_weekend", "shipping_billing_mismatch", "device_risk_score",
    "distance_km", "past_return_rate", "past_order_count",
    "account_age_days", "is_first_time_buyer",
]
CATEGORICAL_FEATURES = ["category", "payment_method"]
TARGET = "is_returned"
TIME_COL = "order_date"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[TIME_COL])
    # audit-only ground-truth columns (segment/latent risk/true probability) must
    # NEVER be used as model features — they exist purely to verify the
    # dataset's realism and are dropped immediately here.
    audit_cols = [c for c in df.columns if c.startswith("_audit_")]
    if audit_cols:
        df = df.drop(columns=audit_cols)
    return df


def time_based_split(df: pd.DataFrame, train_frac=0.70, val_frac=0.15):
    """Split by TIME, not randomly — the model must be evaluated on FUTURE
    orders relative to training, which is how it will actually be used in
    production. Random splitting would leak future patterns into training.
    """
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = df.iloc[:n_train]
    val = df.iloc[n_train:n_train + n_val]
    test = df.iloc[n_train + n_val:]
    return train, val, test


def encode(df: pd.DataFrame, cat_maps=None, fit=False):
    """One-hot encode categoricals. If fit=True, learn categories from this
    df and return the maps; else apply given maps (val/test must reuse
    train's encoding to avoid leakage / unseen-category mismatches).
    """
    out = df[NUMERIC_FEATURES].copy()
    if cat_maps is None:
        cat_maps = {}
    for col in CATEGORICAL_FEATURES:
        if fit:
            cats = sorted(df[col].unique().tolist())
            cat_maps[col] = cats
        else:
            cats = cat_maps[col]
        for c in cats:
            out[f"{col}_{c}"] = (df[col] == c).astype(int)
    return out, cat_maps
