"""
Overview-tab analytics.

Replaces the old workflow of pre-baking static Plotly HTML files with
a `graphs/generate_*.py` script per chart. Charts are now built live,
in the same process, from the historical dataset — so they can't go
stale and there's no separate build step to remember to re-run.
"""

import os

import pandas as pd
import plotly.graph_objects as go

from theme import BLUE, NAVY, PANEL_GRAY, WHITE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "original_dataset.csv")

_df = pd.read_csv(DATASET_PATH)
_df.columns = [c.strip().lower().replace(" ", "_") for c in _df.columns]
_df = _df.dropna(subset=["churn"])
HISTORY_DF = _df


def _base_layout(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=title,
        plot_bgcolor=PANEL_GRAY,
        paper_bgcolor=WHITE,
        font={"family": "Arial, sans-serif", "color": NAVY},
        margin=dict(l=40, r=20, t=50, b=40),
        height=340,
    )
    return fig


def kpi_summary() -> dict:
    df = HISTORY_DF
    churned = df[df["churn"] == 1]
    return {
        "total_customers": int(len(df)),
        "churn_rate": round(100 * churned.shape[0] / len(df), 1) if len(df) else 0.0,
        "avg_tenure": round(df["tenure"].mean(), 1) if "tenure" in df else 0.0,
        "revenue_at_risk": round(churned["total_spend"].sum(), 2) if "total_spend" in df else 0.0,
    }


def fig_churn_by_age_group() -> go.Figure:
    df = HISTORY_DF[HISTORY_DF["churn"] == 1].copy()
    bins = [10, 20, 30, 40, 50, 60, 70, 80]
    labels = ["10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)
    counts = df["age_group"].value_counts().sort_index()
    fig = go.Figure(go.Bar(x=counts.values, y=counts.index, orientation="h", marker_color=BLUE))
    return _base_layout(fig, "Churned Customers by Age Group")


def fig_churn_rate_by_contract() -> go.Figure:
    df = HISTORY_DF
    rate = (df.groupby("contract_length")["churn"].mean() * 100).round(1).sort_values()
    fig = go.Figure(go.Bar(x=rate.index, y=rate.values, marker_color=BLUE,
                           text=[f"{v}%" for v in rate.values], textposition="outside"))
    fig.update_yaxes(title="Churn rate (%)")
    return _base_layout(fig, "Churn Rate by Contract Length")


def fig_churn_rate_by_subscription() -> go.Figure:
    df = HISTORY_DF
    rate = (df.groupby("subscription_type")["churn"].mean() * 100).round(1).sort_values()
    fig = go.Figure(go.Bar(x=rate.index, y=rate.values, marker_color=BLUE,
                           text=[f"{v}%" for v in rate.values], textposition="outside"))
    fig.update_yaxes(title="Churn rate (%)")
    return _base_layout(fig, "Churn Rate by Subscription Type")


def fig_support_calls_vs_churn() -> go.Figure:
    df = HISTORY_DF.copy()
    bins = list(range(0, int(df["support_calls"].max()) + 3, 2))
    labels = [f"{b}-{b+2}" for b in bins[:-1]]
    df["support_calls_group"] = pd.cut(df["support_calls"], bins=bins, labels=labels, right=False)
    rate = (df.groupby("support_calls_group", observed=True)["churn"].mean() * 100).round(1)
    fig = go.Figure(go.Bar(x=rate.index.astype(str), y=rate.values, marker_color=BLUE))
    fig.update_yaxes(title="Churn rate (%)")
    fig.update_xaxes(title="Support calls")
    return _base_layout(fig, "Churn Rate by Support Calls")


def fig_payment_delay_vs_churn() -> go.Figure:
    df = HISTORY_DF.copy()
    bins = list(range(0, 56, 5))
    labels = [f"{b}-{b+5}" for b in bins[:-1]]
    df["payment_delay_group"] = pd.cut(df["payment_delay"], bins=bins, labels=labels, right=False)
    rate = (df.groupby("payment_delay_group", observed=True)["churn"].mean() * 100).round(1)
    fig = go.Figure(go.Bar(x=rate.index.astype(str), y=rate.values, marker_color=BLUE))
    fig.update_yaxes(title="Churn rate (%)")
    fig.update_xaxes(title="Payment delay (days)")
    return _base_layout(fig, "Churn Rate by Payment Delay")


def fig_gender_split() -> go.Figure:
    df = HISTORY_DF[HISTORY_DF["churn"] == 1]
    counts = df["gender"].value_counts()
    fig = go.Figure(go.Pie(labels=counts.index, values=counts.values,
                           marker_colors=[BLUE, NAVY], hole=0.45))
    return _base_layout(fig, "Churned Customers by Gender")
