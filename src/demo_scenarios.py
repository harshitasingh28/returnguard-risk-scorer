"""
Selects real examples from the held-out TEST set (not synthetically
hand-crafted "obviously risky" inputs) to demonstrate the model honestly:
one low-risk, one medium-risk, one high-risk case, based on the model's
own calibrated output distribution. Each is labeled with its TRUE outcome
so the demo doesn't overstate accuracy.
"""
import sys, json
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import pandas as pd
from src.features import encode, TARGET
from src.explain import RiskExplainer

MODEL_DIR = str(PROJECT_ROOT / "models")


def main():
    test = pd.read_csv(f"{MODEL_DIR}/split_test.csv", parse_dates=["order_date"])
    with open(f"{MODEL_DIR}/cat_maps.json") as f:
        cat_maps = json.load(f)
    X_test, _ = encode(test, cat_maps=cat_maps, fit=False)

    explainer = RiskExplainer()
    feature_cols = ["order_value", "discount_pct", "num_items", "hour_of_day",
                     "is_weekend", "shipping_billing_mismatch", "device_risk_score",
                     "distance_km", "past_return_rate", "past_order_count",
                     "account_age_days", "is_first_time_buyer", "category", "payment_method"]
    scores = []
    for _, row in test.iterrows():
        row_df = pd.DataFrame([row[feature_cols]])
        X_row, _ = encode(row_df, cat_maps=cat_maps, fit=False)
        scores.append(explainer.explain_case(X_row)["risk_score_pct"] / 100)
    test = test.reset_index(drop=True)
    test["risk_score"] = scores

    low = test.loc[test["risk_score"].idxmin()]
    high = test.loc[test["risk_score"].idxmax()]
    median_score = test["risk_score"].median()
    med = test.iloc[(test["risk_score"] - median_score).abs().idxmin()]
    actual_positives = test[test[TARGET] == 1]
    caught = actual_positives.loc[actual_positives["risk_score"].idxmax()]

    scenarios = []
    for label, row in [("low_risk_example", low), ("medium_risk_example", med),
                        ("highest_scored_example", high), ("true_positive_caught", caught)]:
        row_df = pd.DataFrame([row[["order_value", "discount_pct", "num_items", "hour_of_day",
                                     "is_weekend", "shipping_billing_mismatch", "device_risk_score",
                                     "distance_km", "past_return_rate", "past_order_count",
                                     "account_age_days", "is_first_time_buyer", "category",
                                     "payment_method"]]])
        X_row, _ = encode(row_df, cat_maps=cat_maps, fit=False)
        explanation = explainer.explain_case(X_row)
        scenarios.append({
            "scenario": label,
            "true_outcome": "RETURNED" if row["is_returned"] == 1 else "NOT_RETURNED",
            "input": row_df.iloc[0].to_dict(),
            "model_output": explanation,
        })
        print(f"\n--- {label} ---")
        print(f"True outcome: {'RETURNED' if row['is_returned']==1 else 'NOT_RETURNED'}")
        print(f"Predicted risk: {explanation['risk_score_pct']}% ({explanation['risk_category']})")
        for r in explanation["top_reasons"]:
            print(f"  - {r['reason']} (SHAP contribution: {r['contribution']})")

    with open(PROJECT_ROOT / "reports" / "demo_scenarios.json", "w") as f:
        json.dump(scenarios, f, indent=2, default=str)
    print("\nSaved to reports/demo_scenarios.json")


if __name__ == "__main__":
    main()
