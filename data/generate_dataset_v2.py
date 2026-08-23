"""
Synthetic Return-Risk dataset generator — v2 (post-audit).

WHY V2 EXISTS: v1 generated the label as a direct logistic function of the
same features fed to the model (z = w1*f1 + w2*f2 + ...). A tree model can
partially reverse-engineer that formula, which produced suspiciously clean
feature-importance recovery. That is a credibility weakness, not an ML
achievement. V2 fixes this by generating behavior from a LATENT, PARTIALLY
UNOBSERVED customer process with genuine noise, so the observed features
are only a noisy, indirect signal of true risk — the way real behavioral
data actually works.

METHODOLOGY:
1. Each customer belongs to one of 5 latent SEGMENTS (unobserved to the
   model) with different behavioral tendencies. Segments are not a
   feature — the model never sees segment membership directly, only its
   noisy downstream effects on order features.
2. Within a segment, individual customers still vary (segment sets a
   distribution, not a fixed value) — so segment membership alone doesn't
   determine outcome.
3. The return outcome for each order is drawn from a Bernoulli process
   whose true probability depends on a NONLINEAR, NOISY combination of
   latent segment + observed features + an idiosyncratic per-order shock
   — deliberately including irreducible randomness (e.g. "customer just
   changed their mind," "product didn't fit," which no feature captures).
4. Legitimate customers CAN look risky on a given order (e.g. a low-risk
   segment customer who happens to buy a discounted item that doesn't fit)
   and risky customers CAN look clean on a given order — enforced by
   capping how much any single observed feature can move the latent risk,
   and by a substantial noise term.
5. All customer-level aggregates (past_return_rate etc.) are computed
   using STRICTLY PRIOR orders only (unchanged from v1 — this was already
   leakage-free and is preserved).

This is still synthetic and still stated as such. The change is that the
label-generating process is now a plausible simulation of noisy real-world
behavior rather than a recoverable formula over the exact model inputs.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG = np.random.default_rng(7)  # new seed — this is a genuinely new dataset, not a re-tune of v1

N_CUSTOMERS = 6000
N_ORDERS = 22000
START_DATE = datetime(2026, 1, 1)
WINDOW_DAYS = 210

CATEGORIES = ["apparel", "electronics", "beauty", "home", "grocery", "footwear"]
PAYMENT_METHODS = ["card", "upi", "netbanking", "wallet", "cod"]

# 5 latent customer segments — NOT observed by the model. Each has a
# behavioral tendency (mean latent risk) and a spread (how much individual
# customers within the segment vary), so segment membership alone does not
# determine outcome — there's real overlap between segments.
SEGMENTS = [
    {"name": "loyal_low_risk",     "share": 0.45, "risk_mean": -1.6, "risk_sd": 0.5, "order_rate": 1.2},
    {"name": "occasional_normal",  "share": 0.28, "risk_mean": -0.9, "risk_sd": 0.6, "order_rate": 0.7},
    {"name": "price_sensitive",    "share": 0.12, "risk_mean": -0.3, "risk_sd": 0.7, "order_rate": 1.0},
    {"name": "frequent_returner",  "share": 0.10, "risk_mean": 0.5,  "risk_sd": 0.8, "order_rate": 1.4},
    {"name": "high_risk_abuser",   "share": 0.05, "risk_mean": 1.3,  "risk_sd": 0.9, "order_rate": 1.6},
]


def assign_segments(n):
    shares = [s["share"] for s in SEGMENTS]
    idx = RNG.choice(len(SEGMENTS), size=n, p=shares)
    return idx


def make_customers(n):
    seg_idx = assign_segments(n)
    latent_risk = np.array([
        RNG.normal(SEGMENTS[i]["risk_mean"], SEGMENTS[i]["risk_sd"]) for i in seg_idx
    ])
    order_rate = np.array([SEGMENTS[i]["order_rate"] for i in seg_idx]) * RNG.lognormal(0, 0.4, size=n)
    account_age_start = RNG.integers(0, 900, size=n)
    return pd.DataFrame({
        "customer_id": np.arange(n),
        "segment_idx": seg_idx,          # kept for audit/analysis only — NEVER fed to the model
        "latent_risk": latent_risk,       # unobserved ground truth propensity
        "order_rate": order_rate,
        "account_age_at_start": account_age_start,
    })


def simulate():
    customers = make_customers(N_CUSTOMERS)
    cust_lookup = customers.set_index("customer_id")

    order_weights = customers["order_rate"].values
    order_weights = order_weights / order_weights.sum()
    customer_ids = RNG.choice(N_CUSTOMERS, size=N_ORDERS, p=order_weights)

    days_offset = RNG.integers(0, WINDOW_DAYS, size=N_ORDERS)
    order_dates = [START_DATE + timedelta(days=int(d)) for d in days_offset]

    df = pd.DataFrame({"customer_id": customer_ids, "order_date": order_dates}) \
        .sort_values(["customer_id", "order_date"]).reset_index(drop=True)
    df["order_id"] = np.arange(len(df))

    # Category/payment preferences correlate WEAKLY with segment (realistic:
    # risky customers aren't uniformly distributed across categories) but
    # with heavy overlap, not a clean split.
    latent_risk_per_order = cust_lookup.loc[df["customer_id"].values, "latent_risk"].values
    cat_probs_base = np.array([0.28, 0.22, 0.14, 0.16, 0.10, 0.10])  # apparel..footwear
    df["category"] = [
        RNG.choice(CATEGORIES, p=_tilt(cat_probs_base, [0.12, -0.05, 0.0, -0.05, -0.08, 0.06], lr))
        for lr in latent_risk_per_order
    ]
    pay_probs_base = np.array([0.42, 0.28, 0.10, 0.08, 0.12])  # card,upi,netbanking,wallet,cod
    df["payment_method"] = [
        RNG.choice(PAYMENT_METHODS, p=_tilt(pay_probs_base, [-0.05, -0.03, -0.02, 0.03, 0.07], lr))
        for lr in latent_risk_per_order
    ]

    df["order_value"] = np.round(np.exp(RNG.normal(6.5, 0.7, size=len(df))), 2)
    df["discount_pct"] = np.clip(RNG.normal(15, 12, size=len(df)) + 3 * np.clip(latent_risk_per_order, 0, None),
                                  0, 80).round(1)
    df["num_items"] = RNG.poisson(1.8, size=len(df)) + 1
    df["hour_of_day"] = RNG.integers(0, 24, size=len(df))
    df["is_weekend"] = (RNG.random(len(df)) < 0.28).astype(int)
    # shipping/billing mismatch: weakly correlated with risk + pure noise (legit customers travel, gift-ship etc.)
    mismatch_p = np.clip(0.04 + 0.03 * np.clip(latent_risk_per_order, 0, None), 0.02, 0.35)
    df["shipping_billing_mismatch"] = (RNG.random(len(df)) < mismatch_p).astype(int)
    # device risk score: noisy proxy, only loosely tied to true latent risk
    device_base = np.clip(RNG.beta(1.3, 6.0, size=len(df)), 0, 1)
    device_signal = np.clip(0.15 * np.clip(latent_risk_per_order, 0, None), 0, 0.4)
    df["device_risk_score"] = np.clip(device_base + device_signal + RNG.normal(0, 0.06, len(df)), 0, 1).round(3)
    df["distance_km"] = np.round(np.clip(RNG.normal(180, 140, size=len(df)), 1, 2500), 1)

    # ---- sequential generation: past aggregates + noisy Bernoulli label ----
    past_return_rate = np.zeros(len(df))
    past_order_count = np.zeros(len(df), dtype=int)
    account_age_days = np.zeros(len(df), dtype=int)
    labels = np.zeros(len(df), dtype=int)
    true_p = np.zeros(len(df))  # ground-truth probability, kept for audit only (not a feature)

    running_returns, running_orders = {}, {}
    acct_start = cust_lookup.loc[df["customer_id"].values, "account_age_at_start"].values

    for i in range(len(df)):
        cid = df.at[i, "customer_id"]
        n_past = running_orders.get(cid, 0)
        past_ret = running_returns.get(cid, [])
        prr = (sum(past_ret) / len(past_ret)) if past_ret else 0.13  # cold-start prior ~ population base rate
        past_return_rate[i] = prr
        past_order_count[i] = n_past
        account_age_days[i] = acct_start[i] + (df.at[i, "order_date"] - START_DATE).days

        lr = latent_risk_per_order[i]
        # NONLINEAR, capped, noisy combination — no single feature can push the
        # decision deterministically; genuine idiosyncratic shock per order.
        observed_signal = (
            0.9 * np.tanh(prr * 2.5)                                  # capped, not linear
            + 0.7 * df.at[i, "shipping_billing_mismatch"]
            + 0.5 * df.at[i, "device_risk_score"]
            + 0.10 * np.tanh(df.at[i, "discount_pct"] / 25)
            + 0.15 * (1 if n_past == 0 else 0)
            - 0.0004 * min(account_age_days[i], 2000)
        )
        idiosyncratic_shock = RNG.normal(0, 1.15)  # "changed their mind", "didn't fit" — irreducible randomness
        z = -1.95 + 0.55 * lr + 0.55 * observed_signal + idiosyncratic_shock
        p = 1 / (1 + np.exp(-z))
        p = float(np.clip(p, 0.01, 0.9))
        true_p[i] = p
        label = 1 if RNG.random() < p else 0
        labels[i] = label

        running_orders[cid] = n_past + 1
        running_returns.setdefault(cid, []).append(label)

    df["past_return_rate"] = past_return_rate.round(3)
    df["past_order_count"] = past_order_count
    df["account_age_days"] = account_age_days
    df["is_first_time_buyer"] = (df["past_order_count"] == 0).astype(int)
    df["is_returned"] = labels

    # audit-only columns (ground truth, NOT used as model features — dropped before training)
    df["_audit_segment"] = cust_lookup.loc[df["customer_id"].values, "segment_idx"].values
    df["_audit_latent_risk"] = latent_risk_per_order
    df["_audit_true_probability"] = true_p

    return df


def _tilt(base_probs, tilts, latent_risk):
    """Shift a categorical distribution slightly based on latent risk, then
    renormalize. `tilts` are the per-category shift at latent_risk=+1;
    scaled down for negative/near-zero latent risk. Heavy overlap remains
    because the tilt is small relative to base_probs.
    """
    factor = np.clip(latent_risk, -1, 1.5) / 1.5
    p = base_probs + np.array(tilts) * max(factor, 0) * 0.5
    p = np.clip(p, 0.01, None)
    return p / p.sum()


if __name__ == "__main__":
    from pathlib import Path
    df = simulate()
    out_path = Path(__file__).resolve().parent / "return_risk_orders.csv"
    df.to_csv(out_path, index=False)
    print(f"Rows: {len(df)}  Positive rate: {df['is_returned'].mean():.4f}")
    print("\nSegment composition (audit-only, not a model feature):")
    print(df.groupby("_audit_segment")["is_returned"].agg(["mean", "count"]))
    print(f"\nCorrelation of true_probability with is_returned: "
          f"{np.corrcoef(df['_audit_true_probability'], df['is_returned'])[0,1]:.3f}  "
          f"(should be well below 1.0 — confirms label isn't deterministic)")
