import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np
import pandas as pd
from src.features import time_based_split, encode
from src.evaluate import fp_cost, fn_cost, expected_total_cost, find_best_threshold


def test_time_based_split_is_chronological_and_non_overlapping():
    dates = pd.date_range("2026-01-01", periods=100, freq="D")
    df = pd.DataFrame({"order_date": dates, "x": range(100)})
    train, val, test = time_based_split(df, 0.6, 0.2)
    assert train["order_date"].max() <= val["order_date"].min()
    assert val["order_date"].max() <= test["order_date"].min()
    assert len(train) + len(val) + len(test) == len(df)


def test_encode_reuses_train_categories_for_val_test():
    train = pd.DataFrame({
        "order_value": [10, 20], "discount_pct": [0, 0], "num_items": [1, 1],
        "hour_of_day": [1, 1], "is_weekend": [0, 0], "shipping_billing_mismatch": [0, 0],
        "device_risk_score": [0.1, 0.1], "distance_km": [1, 1], "past_return_rate": [0, 0],
        "past_order_count": [0, 0], "account_age_days": [0, 0], "is_first_time_buyer": [1, 1],
        "category": ["apparel", "electronics"], "payment_method": ["card", "upi"],
    })
    X_train, cat_maps = encode(train, fit=True)
    unseen = train.copy()
    unseen.loc[0, "category"] = "apparel"  # category present in train, should encode fine
    X_unseen, _ = encode(unseen, cat_maps=cat_maps, fit=False)
    assert list(X_train.columns) == list(X_unseen.columns)


def test_cost_functions_are_positive_and_monotonic_in_order_value():
    values = np.array([100.0, 1000.0])
    fp = fp_cost(values)
    fn = fn_cost(values)
    assert (fp > 0).all() and (fn > 0).all()
    assert fp[1] > fp[0]  # higher order value -> higher friction cost
    assert fn[1] > fn[0]  # higher order value -> higher loss cost


def test_expected_total_cost_zero_when_no_errors():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.0, 0.0, 1.0, 1.0])  # perfect separation
    order_values = np.array([100.0, 100.0, 100.0, 100.0])
    cost, n_fp, n_fn = expected_total_cost(y_true, y_prob, order_values, threshold=0.5)
    assert n_fp == 0 and n_fn == 0
    assert cost == 0.0


def test_find_best_threshold_picks_lower_cost_over_naive():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=500)
    y_prob = np.clip(y_true * 0.6 + rng.normal(0, 0.2, size=500), 0, 1)  # informative score
    order_values = rng.uniform(100, 1000, size=500)
    best, sweep = find_best_threshold(y_true, y_prob, order_values)
    naive_cost, _, _ = expected_total_cost(y_true, y_prob, order_values, threshold=1.1)  # flags nothing
    assert best["expected_cost"] <= naive_cost


def test_explainer_scales_input_before_scoring_logreg():
    """Regression test for a real bug caught during audit: explain_case
    computed X_scaled but then called predict_proba on the UNSCALED input,
    silently producing near-zero risk scores for every case. This test
    would have caught it: a case engineered to be obviously high-risk
    (high past_return_rate, mismatch, high device risk) must score
    meaningfully above a case engineered to be obviously low-risk.
    """
    from src.explain import RiskExplainer
    from src.features import encode
    import json as _json

    with open(PROJECT_ROOT / "models" / "cat_maps.json") as f:
        cat_maps = _json.load(f)

    low_risk = pd.DataFrame([{
        "order_value": 500, "discount_pct": 0, "num_items": 1, "hour_of_day": 12,
        "is_weekend": 0, "shipping_billing_mismatch": 0, "device_risk_score": 0.02,
        "distance_km": 20, "past_return_rate": 0.0, "past_order_count": 20,
        "account_age_days": 900, "is_first_time_buyer": 0,
        "category": "grocery", "payment_method": "upi",
    }])
    high_risk = pd.DataFrame([{
        "order_value": 500, "discount_pct": 60, "num_items": 5, "hour_of_day": 3,
        "is_weekend": 1, "shipping_billing_mismatch": 1, "device_risk_score": 0.95,
        "distance_km": 20, "past_return_rate": 0.8, "past_order_count": 1,
        "account_age_days": 2, "is_first_time_buyer": 0,
        "category": "apparel", "payment_method": "cod",
    }])

    explainer = RiskExplainer()
    X_low, _ = encode(low_risk, cat_maps=cat_maps, fit=False)
    X_high, _ = encode(high_risk, cat_maps=cat_maps, fit=False)
    low_score = explainer.explain_case(X_low)["risk_score_pct"]
    high_score = explainer.explain_case(X_high)["risk_score_pct"]

    assert high_score > low_score, (
        f"High-risk case ({high_score}%) should score above low-risk case "
        f"({low_score}%) — if this fails, scaling is likely broken again."
    )
    assert high_score > 5.0, "High-risk case should not collapse to ~0% (scaling bug symptom)"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
