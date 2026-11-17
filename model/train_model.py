"""
CredX - Alternative Credit Risk Model (training script)

Young users (students, young professionals, freelancers) often have NO traditional
credit history. This script trains a logistic-regression model that predicts the
probability of default (missing a repayment for 90+ days within 12 months) from
ALTERNATIVE data only: rent & bill payments, savings habits, income stability,
gig-platform reputation, BNPL usage, overdrafts and spending mix.

Because real alternative-credit data is private, we generate a realistic synthetic
dataset (n=20,000) whose risk relationships follow published findings
(e.g. on-time rent and savings buffers reduce default; overdrafts, high DTI and
BNPL stacking increase it). The trained weights are exported to
model.json and executed in the browser by the web app.

Run:  python model/train_model.py
"""
import json, pathlib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss

rng = np.random.default_rng(42)
N = 20_000

FEATURES = [
    ("rent_ontime_pct",       "Rent paid on time (%)"),
    ("rent_history_months",   "Months of rent history"),
    ("bills_ontime_pct",      "Phone/utility bills on time (%)"),
    ("savings_rate_pct",      "Share of income saved (%)"),
    ("emergency_fund_months", "Emergency fund (months of expenses)"),
    ("income_volatility_pct", "Income volatility (%)"),
    ("income_history_months", "Months of steady income"),
    ("debt_to_income_pct",    "Debt payments / income (%)"),
    ("bnpl_active",           "Active Buy-Now-Pay-Later plans"),
    ("overdrafts_6m",         "Overdrafts in last 6 months"),
    ("discretionary_pct",     "Non-essential spending (%)"),
    ("platform_rating",       "Gig platform / peer rating (0-5)"),
]

def simulate(n):
    profile = rng.choice([0, 1, 2], size=n, p=[0.4, 0.35, 0.25])  # student, employee, freelancer
    X = {}
    X["rent_ontime_pct"]       = np.clip(rng.beta(8, 1.5, n) * 100, 0, 100)
    X["rent_history_months"]   = np.clip(rng.gamma(2.0, 9, n), 0, 72)
    X["bills_ontime_pct"]      = np.clip(0.6 * X["rent_ontime_pct"] + 0.4 * rng.beta(8, 1.5, n) * 100, 0, 100)
    X["savings_rate_pct"]      = np.clip(rng.normal(10, 8, n) + (profile == 1) * 4, 0, 50)
    X["emergency_fund_months"] = np.clip(X["savings_rate_pct"] / 6 + rng.gamma(1.2, 0.8, n), 0, 12)
    X["income_volatility_pct"] = np.clip(np.where(profile == 2, rng.normal(45, 18, n),
                                          np.where(profile == 0, rng.normal(25, 12, n), rng.normal(10, 6, n))), 0, 100)
    X["income_history_months"] = np.clip(rng.gamma(2.0, 10, n) - (profile == 0) * 8, 0, 72)
    X["debt_to_income_pct"]    = np.clip(rng.gamma(2.0, 9, n) + (profile == 0) * 6, 0, 90)
    X["bnpl_active"]           = rng.poisson(np.where(profile == 0, 1.4, 0.8))
    X["overdrafts_6m"]         = rng.poisson(np.clip(1.6 - X["emergency_fund_months"] * 0.25, 0.1, None))
    X["discretionary_pct"]     = np.clip(rng.normal(32, 12, n), 5, 80)
    X["platform_rating"]       = np.where(profile == 2, np.clip(rng.normal(4.4, 0.45, n), 1, 5), 4.5)

    # "true" latent default risk (unknown to the model)
    z = (-2.4
         - 0.045 * (X["rent_ontime_pct"] - 85)
         - 0.012 * X["rent_history_months"]
         - 0.025 * (X["bills_ontime_pct"] - 85)
         - 0.05  * X["savings_rate_pct"]
         - 0.22  * X["emergency_fund_months"]
         + 0.022 * X["income_volatility_pct"]
         - 0.015 * X["income_history_months"]
         + 0.045 * X["debt_to_income_pct"]
         + 0.28  * X["bnpl_active"]
         + 0.38  * X["overdrafts_6m"]
         + 0.02  * (X["discretionary_pct"] - 30)
         - 0.6   * (X["platform_rating"] - 4.5)
         + rng.normal(0, 0.6, n))
    p = 1 / (1 + np.exp(-z))
    y = rng.binomial(1, p)
    M = np.column_stack([X[f] for f, _ in FEATURES])
    return M, y

X, y = simulate(N)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=7, stratify=y)
scaler = StandardScaler().fit(X_tr)
clf = LogisticRegression(max_iter=2000, C=1.0).fit(scaler.transform(X_tr), y_tr)

proba = clf.predict_proba(scaler.transform(X_te))[:, 1]
metrics = {
    "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
    "default_rate": round(float(y.mean()), 4),
    "auc": round(float(roc_auc_score(y_te, proba)), 4),
    "accuracy": round(float(accuracy_score(y_te, proba > 0.5)), 4),
    "brier": round(float(brier_score_loss(y_te, proba)), 4),
}
print("Test metrics:", metrics)

out = {
    "name": "CredX Alternative Credit Risk Model",
    "version": "1.0.0",
    "type": "logistic_regression",
    "target": "P(default within 12 months)",
    "features": [{"key": k, "label": l, "mean": float(m), "std": float(s), "coef": float(c)}
                 for (k, l), m, s, c in zip(FEATURES, scaler.mean_, scaler.scale_, clf.coef_[0])],
    "intercept": float(clf.intercept_[0]),
    "metrics": metrics,
}
path = pathlib.Path(__file__).resolve().parent.parent / "model.json"
path.write_text(json.dumps(out, indent=2))
print("Saved", path)
for f in out["features"]:
    print(f"{f['key']:24s} coef={f['coef']:+.3f}")
