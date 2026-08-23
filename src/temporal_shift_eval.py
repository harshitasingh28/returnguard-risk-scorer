"""
Temporal-shift robustness experiment (diagnostic only).

This is DELIBERATELY SEPARATE from the production train/val/test pipeline
in src/train.py and src/evaluate.py. It answers a different question than
the standard time-based split: "if we train on an early window and deploy
against a LATER window with a real time gap (not just the next 15% of
chronologically-sorted rows), does performance hold up?"

Design:
  - EARLY window: first 60% of orders by date  -> used to fit a fresh model
  - BUFFER (skipped): middle 20% of orders     -> ensures a genuine gap
    between train and shift-test periods, not adjacent rows
  - LATE window: last 20% of orders by date     -> shift-test set, never
    used for anything else in this experiment

Rules enforced by construction:
  1. The LATE window is NOT used to select a threshold. The threshold
     applied here is read directly from reports/evaluation_report.json
     (the production model's validation-selected threshold) and used
     AS-IS, frozen.
  2. The LATE window is NOT used to tune hyperparameters. The model here
     uses the exact same architecture/hyperparameters as the production
     winner (Logistic Regression, class_weight="balanced") -- nothing is
     adjusted based on how well it does on the LATE window.
  3. This experiment's results are NOT fed back into src/train.py or
     src/evaluate.py, and do NOT change the production model, its
     threshold, or any file under models/. It writes only to
     reports/evaluation_report.json under a clearly separate top-level
     key, temporal_shift_robustness.
  4. Category encoding is fit on the EARLY window only, exactly as
     production does with its own train split -- the LATE window reuses
     those category maps, so any category that appears only in the LATE
     window would be silently dropped (this is checked and reported).
"""
import sys, json
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    precision_score, recall_score, f1_score, average_precision_score,
    roc_auc_score, confusion_matrix,
)
from src.features import load_data, encode, TARGET, TIME_COL

DATA_PATH = str(PROJECT_ROOT / "data" / "return_risk_orders.csv")
REPORT_PATH = str(PROJECT_ROOT / "reports" / "evaluation_report.json")

FN_COST_RATE = 0.35
FP_FLAT_COST = 45.0
FP_MARGIN_RATE = 0.06


def fp_cost(v):
    return FP_FLAT_COST + FP_MARGIN_RATE * v


def fn_cost(v):
    return FN_COST_RATE * v


def expected_cost(y_true, y_pred, order_values):
    fp_mask = (y_pred == 1) & (y_true == 0)
    fn_mask = (y_pred == 0) & (y_true == 1)
    return float(fp_cost(order_values[fp_mask]).sum() + fn_cost(order_values[fn_mask]).sum())


def main():
    df = load_data(DATA_PATH)  # already drops _audit_* columns
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    n = len(df)

    early = df.iloc[: int(n * 0.60)]
    # buffer: df.iloc[int(n*0.60):int(n*0.80)] -- intentionally unused, creates a real time gap
    late = df.iloc[int(n * 0.80):]

    print(f"EARLY (train) window: {len(early)} orders, {early[TIME_COL].min().date()} to {early[TIME_COL].max().date()}")
    print(f"LATE (shift-test) window: {len(late)} orders, {late[TIME_COL].min().date()} to {late[TIME_COL].max().date()}")
    print(f"Buffer skipped: {n - len(early) - len(late)} orders between the two windows\n")

    # Sub-split EARLY into an inner train/calibration slice (chronologically,
    # same discipline as the production pipeline) so this fresh model gets
    # calibrated exactly the way the production model was -- calibrating on
    # EARLY-internal data only, never on LATE. Comparing an uncalibrated raw
    # score against a threshold selected for a CALIBRATED model would be an
    # apples-to-oranges bug, not a genuine drift finding.
    n_early = len(early)
    early_fit = early.iloc[: int(n_early * 0.80)]
    early_calib = early.iloc[int(n_early * 0.80):]

    X_early_fit, cat_maps = encode(early_fit, fit=True)
    X_early_calib, _ = encode(early_calib, cat_maps=cat_maps, fit=False)
    X_late, _ = encode(late, cat_maps=cat_maps, fit=False)
    y_early_fit, y_early_calib, y_late = early_fit[TARGET].values, early_calib[TARGET].values, late[TARGET].values
    ov_late = late["order_value"].values

    # check for categories present in LATE but not EARLY (would be silently
    # dropped by encode() reusing EARLY's cat_maps -- report this honestly)
    dropped_categories = {}
    for col in ["category", "payment_method"]:
        late_cats = set(late[col].unique())
        early_cats = set(cat_maps[col])
        missing = late_cats - early_cats
        if missing:
            dropped_categories[col] = sorted(missing)

    scaler = StandardScaler()
    X_early_fit_s = scaler.fit_transform(X_early_fit)
    X_early_calib_s = scaler.transform(X_early_calib)
    X_late_s = scaler.transform(X_late)

    # SAME architecture/hyperparameters as the production winner (logreg),
    # same calibration procedure (sigmoid, fit on a held-out slice).
    # Nothing here is tuned against the LATE window.
    base_model = LogisticRegression(max_iter=2000, class_weight="balanced")
    base_model.fit(X_early_fit_s, y_early_fit)
    model = CalibratedClassifierCV(FrozenEstimator(base_model), method="sigmoid")
    model.fit(X_early_calib_s, y_early_calib)

    # Frozen threshold: read from the ALREADY-DECIDED production report.
    # This experiment does not select or adjust a threshold of its own.
    with open(REPORT_PATH) as f:
        existing_report = json.load(f)
    winner = existing_report["winning_model"]
    frozen_threshold = next(m["selected_threshold"] for m in existing_report["models"] if m["model"] == winner)

    y_prob_late = model.predict_proba(X_late_s)[:, 1]
    y_pred_late = (y_prob_late >= frozen_threshold).astype(int)

    cm = confusion_matrix(y_late, y_pred_late, labels=[0, 1]).tolist()
    cost = expected_cost(y_late, y_pred_late, ov_late)
    naive_cost = expected_cost(y_late, np.zeros_like(y_pred_late), ov_late)

    result = {
        "purpose": "Diagnostic robustness check only. Does NOT affect the production model, "
                   "threshold, or any file under models/. Not used for tuning.",
        "methodology": "Fresh LogisticRegression trained on the EARLY window's inner fit-slice "
                        "(80% of EARLY), calibrated (sigmoid) on EARLY's remaining inner calib-slice "
                        "(20% of EARLY) -- mirroring the production pipeline's own calibration "
                        "discipline. The LATE window is touched only once, for final scoring, exactly "
                        "as the production test set is.",
        "early_window": {"n": int(len(early)), "date_start": str(early[TIME_COL].min().date()),
                          "date_end": str(early[TIME_COL].max().date()),
                          "inner_fit_n": int(len(early_fit)), "inner_calib_n": int(len(early_calib)),
                          "positive_rate": round(float(early[TARGET].mean()), 4)},
        "buffer_skipped_n": int(n - len(early) - len(late)),
        "late_window_shift_test": {"n": int(len(late)), "date_start": str(late[TIME_COL].min().date()),
                                    "date_end": str(late[TIME_COL].max().date()),
                                    "positive_rate": round(float(y_late.mean()), 4)},
        "categories_in_late_not_in_early": dropped_categories or None,
        "frozen_threshold_used": frozen_threshold,
        "frozen_threshold_source": f"production winning model ({winner}), selected on the ORIGINAL validation split -- reused as-is here, not re-tuned",
        "precision": round(float(precision_score(y_late, y_pred_late, zero_division=0)), 4),
        "recall": round(float(recall_score(y_late, y_pred_late, zero_division=0)), 4),
        "f1": round(float(f1_score(y_late, y_pred_late, zero_division=0)), 4),
        "pr_auc": round(float(average_precision_score(y_late, y_prob_late)), 4),
        "roc_auc": round(float(roc_auc_score(y_late, y_prob_late)), 4),
        "confusion_matrix": {"tn": cm[0][0], "fp": cm[0][1], "fn": cm[1][0], "tp": cm[1][1]},
        "expected_cost_inr": round(cost, 2),
        "naive_approve_all_cost_inr": round(naive_cost, 2),
        "cost_reduction_pct": round(100 * (1 - cost / naive_cost), 2) if naive_cost > 0 else None,
    }

    print("=== TEMPORAL-SHIFT RESULT (frozen threshold, no tuning) ===")
    for k, v in result.items():
        print(f"  {k}: {v}")

    # comparison to the ORIGINAL standard test-set result for the same model,
    # for context only -- not used to adjust anything
    original_test = next(m["test_metrics_held_out"] for m in existing_report["models"] if m["model"] == winner)
    result["comparison_to_original_held_out_test"] = {
        "original_precision": original_test["precision"], "original_recall": original_test["recall"],
        "original_cost_reduction_pct": original_test["cost_reduction_pct"],
        "note": "Original test set is the next 15% of chronologically-sorted rows after training; "
                "this experiment uses a stricter, more distant later window with a skipped buffer.",
    }

    existing_report["temporal_shift_robustness"] = result
    with open(REPORT_PATH, "w") as f:
        json.dump(existing_report, f, indent=2)
    print(f"\nAppended to {REPORT_PATH} under 'temporal_shift_robustness'. "
          f"Production model/threshold/artifacts were NOT modified.")


if __name__ == "__main__":
    main()
