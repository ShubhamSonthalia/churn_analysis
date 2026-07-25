"""
Churn prediction + retention-recommendation logic.

Scoring now happens in-process (the original app shelled out to
`python run.py` via subprocess on every button click, which was slow
and fragile). Recommendations are driven by the same feature-importance
ranking the model actually relies on, computed once offline with
sklearn's permutation_importance and shipped in feature_importance.json,
so the "prevention" side of the product is grounded in the model
rather than guesswork.
"""

import json
import os
import pickle

import pandas as pd
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "neural_net.pkl")
INSIGHTS_PATH = os.path.join(BASE_DIR, "feature_importance.json")

CATEGORICAL_COLUMNS = ["gender", "subscription_type", "contract_length"]
NUMERIC_COLUMNS = [
    "age", "tenure", "usage_frequency", "support_calls",
    "payment_delay", "last_interaction", "total_spend",
]
NON_FEATURE_COLUMNS = ["customerid", "customer_name", "churn", "predicted_churn",
                        "churn_probability", "churn_classification"]

with open(MODEL_PATH, "rb") as f:
    MODEL = pickle.load(f)

with open(INSIGHTS_PATH) as f:
    _INSIGHTS = json.load(f)

THRESHOLDS = _INSIGHTS["thresholds"]
FEATURE_IMPORTANCE = _INSIGHTS["feature_importance"]

REQUIRED_COLUMNS = {
    "customerid", "customer_name", "age", "gender", "tenure", "usage_frequency",
    "support_calls", "payment_delay", "subscription_type", "contract_length",
    "total_spend", "last_interaction",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def validate_columns(df: pd.DataFrame) -> list[str]:
    """Return a list of missing required columns, if any."""
    return sorted(REQUIRED_COLUMNS - set(df.columns))


def _encode_for_model(df: pd.DataFrame) -> pd.DataFrame:
    encoded = df.copy()
    for col in CATEGORICAL_COLUMNS:
        if col in encoded.columns and encoded[col].dtype == object:
            encoded[col] = LabelEncoder().fit_transform(encoded[col].astype(str))
    for col in NUMERIC_COLUMNS:
        if col in encoded.columns:
            encoded[col] = pd.to_numeric(encoded[col], errors="coerce")
    return encoded


def score_customers(raw_df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Run the model over a dataframe of customers and return an enriched copy.

    Adds: churn_probability, predicted_churn, churn_classification.
    Categorical columns are kept in their original human-readable form
    in the returned frame (only an internal copy is label-encoded for
    the model itself), so downstream display/recommendation code can
    read e.g. contract_length == "Monthly" directly.
    """
    df = normalize_columns(raw_df)
    encoded = _encode_for_model(df)

    feature_cols = [c for c in encoded.columns if c not in NON_FEATURE_COLUMNS]
    X = encoded[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True)).fillna(0)

    probabilities = MODEL.predict_proba(X)[:, 1]
    df = df.reset_index(drop=True)
    df["churn_probability"] = probabilities.round(4)
    df["predicted_churn"] = (probabilities >= threshold).astype(int)

    df["churn_classification"] = "Low Churn"
    df.loc[(probabilities >= 0.5) & (probabilities < 0.7), "churn_classification"] = "Medium Churn"
    df.loc[probabilities >= 0.7, "churn_classification"] = "High Churn"

    return df


def recommend_actions(row: pd.Series) -> list[str]:
    """Rule-based retention playbook for one scored customer row.

    Thresholds come from the historical dataset's own percentiles, and
    the priority order mirrors the model's real feature-importance
    ranking (support_calls first, since it's by far the strongest
    churn driver in this model).
    """
    t = THRESHOLDS
    actions = []

    def get(col, default=0):
        val = row.get(col, default)
        return default if pd.isna(val) else val

    if get("support_calls") >= t["support_calls"]["p75"]:
        actions.append(
            "Escalate to a senior support agent — repeated support contact is the single "
            "strongest churn signal for this customer."
        )
    if get("payment_delay") >= t["payment_delay"]["p75"]:
        actions.append(
            "Offer a flexible billing date or short-term payment plan to reduce payment friction."
        )
    if get("last_interaction") >= t["last_interaction"]["p75"]:
        actions.append(
            "Send a personalized re-engagement message — this customer hasn't interacted recently."
        )
    if str(get("contract_length", "")).strip().lower() == "monthly":
        actions.append(
            "Pitch a discounted annual plan; longer contracts show meaningfully lower churn."
        )
    if get("total_spend") <= t["total_spend"]["p50"]:
        actions.append(
            "Highlight underused features or offer a loyalty discount to raise perceived value."
        )
    if get("usage_frequency") <= t["usage_frequency"]["p50"]:
        actions.append(
            "Send a product-usage tutorial or onboarding check-in to lift engagement."
        )

    if not actions:
        actions.append("No major risk flags detected : keep up standard engagement touchpoints.")
    return actions


def feature_importance_dataframe() -> pd.DataFrame:
    return pd.DataFrame(FEATURE_IMPORTANCE)
