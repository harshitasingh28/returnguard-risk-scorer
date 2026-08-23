import sys, json, joblib
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.features import load_data, time_based_split, encode, TARGET

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
import xgboost as xgb
import numpy as np

DATA_PATH = str(PROJECT_ROOT / "data" / "return_risk_orders.csv")
MODEL_DIR = str(PROJECT_ROOT / "models")


def main():
    df = load_data(DATA_PATH)
    train, val, test = time_based_split(df)
    print(f"Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")
    print(f"Train pos rate: {train[TARGET].mean():.4f}  "
          f"Val pos rate: {val[TARGET].mean():.4f}  "
          f"Test pos rate: {test[TARGET].mean():.4f}")

    X_train, cat_maps = encode(train, fit=True)
    X_val, _ = encode(val, cat_maps=cat_maps, fit=False)
    X_test, _ = encode(test, cat_maps=cat_maps, fit=False)
    y_train, y_val, y_test = train[TARGET].values, val[TARGET].values, test[TARGET].values
    feature_names = X_train.columns.tolist()

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    # ---- Model 1: Logistic Regression (interpretable baseline) ----
    logreg = LogisticRegression(max_iter=2000, class_weight="balanced")
    logreg.fit(X_train_s, y_train)

    # ---- Model 2: Random Forest ----
    rf = RandomForestClassifier(
        n_estimators=400, max_depth=8, min_samples_leaf=20,
        class_weight="balanced_subsample", random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    # ---- Model 3: XGBoost (production candidate) ----
    pos, neg = y_train.sum(), len(y_train) - y_train.sum()
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=5,
        reg_lambda=2.0, scale_pos_weight=neg / max(pos, 1),
        eval_metric="aucpr", random_state=42,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # calibrate every model on VALIDATION ONLY (never train, never test)
    logreg_cal = CalibratedClassifierCV(FrozenEstimator(logreg), method="sigmoid")
    logreg_cal.fit(X_val_s, y_val)
    rf_cal = CalibratedClassifierCV(FrozenEstimator(rf), method="sigmoid")
    rf_cal.fit(X_val, y_val)
    xgb_cal = CalibratedClassifierCV(FrozenEstimator(xgb_model), method="sigmoid")
    xgb_cal.fit(X_val, y_val)

    joblib.dump(scaler, f"{MODEL_DIR}/scaler.joblib")
    joblib.dump(logreg_cal, f"{MODEL_DIR}/model_logreg_calibrated.joblib")
    joblib.dump(rf_cal, f"{MODEL_DIR}/model_rf_calibrated.joblib")
    joblib.dump(xgb_cal, f"{MODEL_DIR}/model_xgb_calibrated.joblib")
    xgb_model.save_model(f"{MODEL_DIR}/production_xgb.json")  # raw booster, needed for SHAP TreeExplainer
    with open(f"{MODEL_DIR}/cat_maps.json", "w") as f:
        json.dump(cat_maps, f)
    with open(f"{MODEL_DIR}/feature_names.json", "w") as f:
        json.dump(feature_names, f)

    train.to_csv(f"{MODEL_DIR}/split_train.csv", index=False)
    val.to_csv(f"{MODEL_DIR}/split_val.csv", index=False)
    test.to_csv(f"{MODEL_DIR}/split_test.csv", index=False)

    print("Training complete. 3 calibrated models saved (logreg, rf, xgb).")


if __name__ == "__main__":
    main()
