import sys, json
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd

from src.features import encode, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from src.explain import RiskExplainer

app = FastAPI(title="AI Risk Manager — Return-Risk Scorer", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

with open(PROJECT_ROOT / "models" / "cat_maps.json") as f:
    CAT_MAPS = json.load(f)

explainer = RiskExplainer()

# decision thresholds pulled from the evaluation report (val-selected, cost-optimal)
with open(PROJECT_ROOT / "reports" / "evaluation_report.json") as f:
    _report = json.load(f)
_winner = _report["winning_model"]
PROD_THRESHOLD = next(m["selected_threshold"] for m in _report["models"] if m["model"] == _winner)
LOW_MED_BOUNDARY = PROD_THRESHOLD * 0.6  # softer boundary for a 3-tier UX (LOW / MEDIUM / HIGH)


class OrderInput(BaseModel):
    order_value: float = Field(..., gt=0)
    discount_pct: float = Field(0, ge=0, le=100)
    num_items: int = Field(1, ge=1)
    hour_of_day: int = Field(12, ge=0, le=23)
    is_weekend: int = Field(0, ge=0, le=1)
    shipping_billing_mismatch: int = Field(0, ge=0, le=1)
    device_risk_score: float = Field(0.1, ge=0, le=1)
    distance_km: float = Field(50, ge=0)
    past_return_rate: float = Field(0.15, ge=0, le=1)
    past_order_count: int = Field(0, ge=0)
    account_age_days: int = Field(0, ge=0)
    is_first_time_buyer: int = Field(1, ge=0, le=1)
    category: str
    payment_method: str


@app.get("/health")
def health():
    return {"status": "ok", "production_threshold": PROD_THRESHOLD}


@app.post("/score")
def score_order(order: OrderInput):
    row = pd.DataFrame([order.dict()])
    if order.category not in CAT_MAPS["category"]:
        raise HTTPException(400, f"Unknown category. Valid: {CAT_MAPS['category']}")
    if order.payment_method not in CAT_MAPS["payment_method"]:
        raise HTTPException(400, f"Unknown payment_method. Valid: {CAT_MAPS['payment_method']}")

    X, _ = encode(row, cat_maps=CAT_MAPS, fit=False)
    result = explainer.explain_case(X)

    decision = "BLOCK_MANUAL_REVIEW" if result["risk_score_pct"] / 100 >= PROD_THRESHOLD else (
        "VERIFY" if result["risk_score_pct"] / 100 >= LOW_MED_BOUNDARY else "APPROVE")

    return {
        **result,
        "decision": decision,
        "threshold_used": PROD_THRESHOLD,
        "note": "Synthetic-data demo model (v2, noisy latent-segment generator) — see README for evaluation methodology and limitations.",
    }


@app.get("/model-info")
def model_info():
    return {
        "winning_model": _winner,
        "global_feature_importance": explainer.global_feature_importance(),
        "model_benchmark": _report["models"],
        "rule_based_baseline": _report["rule_based_baseline"],
        "review_capacity": _report["review_capacity"],
    }
