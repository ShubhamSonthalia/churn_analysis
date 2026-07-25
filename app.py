"""
Customer Churn Analysis & Prevention
-------------------------------------
A Dash app that scores customers for churn risk with a trained neural
network, visualizes historical churn drivers, and turns the model's
output into a concrete, per-customer retention playbook.

Run with:
    python app.py
then open http://127.0.0.1:8050
"""

import base64
import io
import os
from datetime import datetime

import pandas as pd
from dash import Dash, dcc, html, Input, Output, State, dash_table, ctx
from dash.exceptions import PreventUpdate

import analytics
import churn_model as cm
import data_store
from theme import (
    BLUE, NAVY, PANEL_GRAY, CARD_GRAY, CARD_BLUE_TINT, CARD_WHITE_TINT, WHITE,
    CARD_STYLE, PANEL_STYLE, SECTION_HEADER_STYLE, PRIMARY_BUTTON_STYLE,
    SECONDARY_BUTTON_STYLE, TAB_STYLE, TAB_SELECTED_STYLE,
    risk_color, risk_bg,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CUSTOMERS_PATH = os.path.join(BASE_DIR, "customers.csv")

# ---------------------------------------------------------------------------
# App init
# ---------------------------------------------------------------------------

app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Customer Churn Analysis & Prevention"
server = app.server

_default_raw = pd.read_csv(DEFAULT_CUSTOMERS_PATH)
_default_scored = cm.score_customers(_default_raw)


# ---------------------------------------------------------------------------
# Small reusable building blocks
# ---------------------------------------------------------------------------

def kpi_card(label: str, value: str, accent: str = BLUE) -> html.Div:
    return html.Div(
        [
            html.Div(label, style={"fontSize": "0.85em", "color": "#5b6b7a", "fontWeight": "bold",
                                     "textTransform": "uppercase", "letterSpacing": "0.03em"}),
            html.Div(value, style={"fontSize": "1.8em", "fontWeight": "bold", "color": accent,
                                     "marginTop": "6px"}),
        ],
        style={**CARD_STYLE, "flex": "1", "minWidth": "180px", "borderTop": f"4px solid {accent}"},
    )


def risk_badge(classification: str) -> html.Span:
    return html.Span(
        classification,
        style={
            "backgroundColor": risk_bg(classification),
            "color": risk_color(classification),
            "padding": "4px 12px",
            "borderRadius": "999px",
            "fontWeight": "bold",
            "fontSize": "0.85em",
        },
    )


def section_title(text: str) -> html.H3:
    return html.H3(text, style=SECTION_HEADER_STYLE)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

app.layout = html.Div(
    style={"backgroundColor": WHITE, "fontFamily": "Arial, sans-serif", "padding": "24px 32px"},
    children=[
        dcc.Store(id="raw-store", data=_default_raw.to_dict("records")),
        dcc.Store(id="scored-store", data=_default_scored.to_dict("records")),
        dcc.Store(id="threshold-store", data=0.5),
        dcc.Download(id="download-results"),

        html.Div(
            [
                html.H1("Customer Churn Analysis & Prevention",
                        style={"textAlign": "center", "color": NAVY, "marginBottom": "4px",
                               "fontSize": "2.3em", "fontWeight": "bold"}),
                html.H2(
                    
                ),
            ]
        ),

        dcc.Tabs(
            id="tabs",
            value="tab-overview",
            style={"display": "flex", "justifyContent": "center", "flexWrap": "wrap", "border": "none"},
            children=[
                dcc.Tab(label="Overview", value="tab-overview", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                dcc.Tab(label="Batch Analysis", value="tab-batch", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                dcc.Tab(label="Customer Lookup", value="tab-lookup", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                dcc.Tab(label="Prevention Playbook", value="tab-prevention", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
            ],
        ),

        html.Div(id="tab-content", style={"marginTop": "24px"}),
    ],
)


# ---------------------------------------------------------------------------
# Tab renderers
# ---------------------------------------------------------------------------

def render_overview() -> html.Div:
    kpi = analytics.kpi_summary()
    return html.Div(
        [
            html.Div(
                [
                    kpi_card("Total Customers", f"{kpi['total_customers']:,}"),
                    kpi_card("Historical Churn Rate", f"{kpi['churn_rate']}%", accent="#FF6B6B"),
                    kpi_card("Avg. Tenure", f"{kpi['avg_tenure']} mo"),
                    kpi_card("Revenue at Risk", f"${kpi['revenue_at_risk']:,.0f}", accent="#FF6B6B"),
                ],
                style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginBottom": "24px"},
            ),
            html.P(
                "These figures are computed live from the historical dataset "
                "(original_dataset.csv), not pre-baked images, so they always reflect the data on disk.",
                style={"color": "#6b7a88", "fontSize": "0.9em"},
            ),
            html.Div(
                [
                    html.Div(dcc.Graph(figure=analytics.fig_churn_rate_by_contract(), config={"displayModeBar": False}),
                              style={**CARD_STYLE, "flex": "1", "minWidth": "360px"}),
                    html.Div(dcc.Graph(figure=analytics.fig_churn_rate_by_subscription(), config={"displayModeBar": False}),
                              style={**CARD_STYLE, "flex": "1", "minWidth": "360px"}),
                ],
                style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginBottom": "16px"},
            ),
            html.Div(
                [
                    html.Div(dcc.Graph(figure=analytics.fig_support_calls_vs_churn(), config={"displayModeBar": False}),
                              style={**CARD_STYLE, "flex": "1", "minWidth": "360px"}),
                    html.Div(dcc.Graph(figure=analytics.fig_payment_delay_vs_churn(), config={"displayModeBar": False}),
                              style={**CARD_STYLE, "flex": "1", "minWidth": "360px"}),
                ],
                style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginBottom": "16px"},
            ),
            html.Div(
                [
                    html.Div(dcc.Graph(figure=analytics.fig_churn_by_age_group(), config={"displayModeBar": False}),
                              style={**CARD_STYLE, "flex": "1", "minWidth": "360px"}),
                    html.Div(dcc.Graph(figure=analytics.fig_gender_split(), config={"displayModeBar": False}),
                              style={**CARD_STYLE, "flex": "1", "minWidth": "360px"}),
                ],
                style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
            ),
        ]
    )


def render_batch() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    section_title("Score a Customer Roster"),
                    html.P(
                        "Uses the bundled customers.csv by default, or upload your own CSV with the "
                        "same columns (CustomerID, Customer_Name, Age, Gender, Tenure, Usage Frequency, "
                        "Support Calls, Payment Delay, Subscription Type, Contract Length, Total Spend, "
                        "Last Interaction).",
                        style={"color": "#6b7a88", "fontSize": "0.9em"},
                    ),
                    dcc.Upload(
                        id="upload-data",
                        children=html.Div(["Drag and drop or ", html.A("select a CSV file")]),
                        style={
                            "width": "100%", "height": "56px", "lineHeight": "56px",
                            "borderWidth": "2px", "borderStyle": "dashed", "borderRadius": "10px",
                            "borderColor": BLUE, "textAlign": "center", "color": BLUE,
                            "marginBottom": "12px", "backgroundColor": CARD_BLUE_TINT,
                        },
                        multiple=False,
                    ),
                    html.Div(id="upload-status", style={"marginBottom": "10px", "fontSize": "0.9em"}),

                    html.Div(
                        [
                            html.Label("Churn probability threshold for flagging as churn:",
                                        style={"fontWeight": "bold", "color": NAVY, "marginRight": "10px"}),
                            html.Div(
                                dcc.Slider(
                                    id="threshold-slider", min=0.1, max=0.9, step=0.05, value=0.5,
                                    marks={i / 10: str(i / 10) for i in range(1, 10, 2)},
                                    tooltip={"placement": "bottom", "always_visible": True},
                                ),
                                style={"width": "320px"},
                            ),
                        ],
                        style={"display": "flex", "alignItems": "center", "gap": "10px", "marginBottom": "16px",
                               "flexWrap": "wrap"},
                    ),

                    html.Div(
                        [
                            html.Button("Run Analysis", id="run-analysis-btn", n_clicks=0, style=PRIMARY_BUTTON_STYLE),
                            html.Button("Download Results (CSV)", id="download-btn", n_clicks=0,
                                        style=SECONDARY_BUTTON_STYLE),
                        ],
                        style={"display": "flex", "gap": "12px"},
                    ),
                    html.Div(id="batch-status", style={"marginTop": "12px", "color": "green", "fontWeight": "bold"}),
                ],
                style=PANEL_STYLE,
            ),

            html.Div(id="batch-summary-cards", style={"display": "flex", "gap": "16px", "flexWrap": "wrap",
                                                          "marginBottom": "16px"}),

            html.Div(id="batch-results-table"),
        ]
    )


def render_lookup(scored_records) -> html.Div:
    df = pd.DataFrame(scored_records)
    options = [
        {"label": f"{row.get('customer_name', '')} (ID: {row['customerid']}) — {row['churn_classification']}",
         "value": row["customerid"]}
        for row in df.sort_values("churn_probability", ascending=False).to_dict("records")
    ] if not df.empty else []

    return html.Div(
        [
            section_title("Customer Lookup"),
            dcc.Dropdown(
                id="customer-id-dropdown",
                options=options,
                placeholder="Select a CustomerID",
                style={"marginBottom": "16px", "maxWidth": "500px"},
            ),
            html.Div(id="customer-details"),
        ]
    )


def render_prevention(scored_records) -> html.Div:
    fi_df = cm.feature_importance_dataframe().sort_values("importance")
    import plotly.graph_objects as go
    fig = go.Figure(go.Bar(x=fi_df["importance"], y=fi_df["feature"], orientation="h", marker_color=BLUE))
    fig.update_layout(
        title="What Actually Drives Churn (model feature importance)",
        plot_bgcolor=PANEL_GRAY, paper_bgcolor=WHITE,
        font={"family": "Arial, sans-serif", "color": NAVY},
        margin=dict(l=20, r=20, t=50, b=40), height=380,
    )

    df = pd.DataFrame(scored_records)
    action_counts = {}
    if not df.empty:
        for _, row in df.iterrows():
            for action in cm.recommend_actions(row):
                action_counts[action] = action_counts.get(action, 0) + 1
    ranked_actions = sorted(action_counts.items(), key=lambda t: -t[1])

    return html.Div(
        [
            html.Div(dcc.Graph(figure=fig, config={"displayModeBar": False}), style=CARD_STYLE),
            html.Div(
                [
                    section_title("Recommended Actions Across Your Current Roster"),
                    html.P(
                        "Counts of how many currently-scored customers trigger each retention rule "
                        ": use this to prioritize which playbook action to run first.",
                        style={"color": "#6b7a88", "fontSize": "0.9em"},
                    ),
                    html.Ul(
                        [html.Li(f"{action}  -  {count} customer(s)") for action, count in ranked_actions]
                        or [html.Li("Run a batch analysis to populate this list.")],
                        style={"lineHeight": "1.9"},
                    ),
                ],
                style={**PANEL_STYLE, "marginTop": "16px"},
            ),
        ]
    )


@app.callback(Output("tab-content", "children"), Input("tabs", "value"), Input("scored-store", "data"))
def render_tab(tab, scored_records):
    if tab == "tab-overview":
        return render_overview()
    if tab == "tab-batch":
        return render_batch()
    if tab == "tab-lookup":
        return render_lookup(scored_records)
    if tab == "tab-prevention":
        return render_prevention(scored_records)
    return html.Div("Select a tab.")


# ---------------------------------------------------------------------------
# Upload handling
# ---------------------------------------------------------------------------

@app.callback(
    Output("raw-store", "data"),
    Output("upload-status", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True,
)
def handle_upload(contents, filename):
    if contents is None:
        raise PreventUpdate
    try:
        _, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        normalized = cm.normalize_columns(df)
        missing = cm.validate_columns(normalized)
        if missing:
            return _default_raw.to_dict("records"), html.Span(
                f"'{filename}' is missing required column(s): {', '.join(missing)}. Using default roster instead.",
                style={"color": "#B00020"},
            )
        return df.to_dict("records"), html.Span(f"Loaded '{filename}' ({len(df)} rows).", style={"color": "green"})
    except Exception as exc:
        return _default_raw.to_dict("records"), html.Span(
            f"Could not read '{filename}': {exc}. Using default roster instead.", style={"color": "#B00020"}
        )


# ---------------------------------------------------------------------------
# Run analysis
# ---------------------------------------------------------------------------

@app.callback(
    Output("scored-store", "data"),
    Output("batch-status", "children"),
    Input("run-analysis-btn", "n_clicks"),
    State("raw-store", "data"),
    State("threshold-slider", "value"),
    prevent_initial_call=True,
)
def run_analysis(n_clicks, raw_records, threshold):
    if not n_clicks:
        raise PreventUpdate
    try:
        raw_df = pd.DataFrame(raw_records)
        scored = cm.score_customers(raw_df, threshold=threshold or 0.5)
        data_store.sync_customers(scored)
        timestamp = datetime.now().strftime("%H:%M:%S")
        return scored.to_dict("records"), f"Model run complete at {timestamp} — {len(scored)} customers scored."
    except Exception as exc:
        raise PreventUpdate from exc


@app.callback(
    Output("batch-summary-cards", "children"),
    Output("batch-results-table", "children"),
    Input("scored-store", "data"),
)
def update_batch_view(scored_records):
    df = pd.DataFrame(scored_records)
    if df.empty:
        return [], html.Div("No data scored yet.")

    counts = df["churn_classification"].value_counts()
    cards = [
        kpi_card("Total Scored", f"{len(df)}"),
        kpi_card("High Churn", f"{counts.get('High Churn', 0)}", accent="#FF0000"),
        kpi_card("Medium Churn", f"{counts.get('Medium Churn', 0)}", accent="#B8860B"),
        kpi_card("Low Churn", f"{counts.get('Low Churn', 0)}", accent="#1E7B34"),
    ]

    display_cols = [c for c in [
        "customerid", "customer_name", "age", "gender", "tenure", "usage_frequency",
        "support_calls", "payment_delay", "subscription_type", "contract_length",
        "total_spend", "last_interaction", "churn_probability", "churn_classification",
    ] if c in df.columns]

    table = dash_table.DataTable(
        id="results-table",
        columns=[{"name": c.replace("_", " ").title(), "id": c} for c in display_cols],
        data=df[display_cols].to_dict("records"),
        sort_action="native",
        filter_action="native",
        page_size=10,
        style_table={"overflowX": "auto", "border": "1px solid #ddd", "borderRadius": "8px"},
        style_cell={"textAlign": "center", "padding": "10px", "fontFamily": "Arial, sans-serif", "fontSize": "13px"},
        style_header={"backgroundColor": BLUE, "color": "white", "fontWeight": "bold"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#f9f9f9"},
            {"if": {"filter_query": "{churn_classification} = 'High Churn'"},
             "backgroundColor": "#FFCCCC", "color": "#FF0000", "fontWeight": "bold"},
            {"if": {"filter_query": "{churn_classification} = 'Medium Churn'"},
             "backgroundColor": "#FFF3CD", "color": "#B8860B", "fontWeight": "bold"},
        ],
    )
    return cards, table


@app.callback(
    Output("download-results", "data"),
    Input("download-btn", "n_clicks"),
    State("scored-store", "data"),
    prevent_initial_call=True,
)
def download_results(n_clicks, scored_records):
    if not n_clicks:
        raise PreventUpdate
    df = pd.DataFrame(scored_records)
    return dcc.send_data_frame(df.to_csv, "churn_predictions.csv", index=False)


# ---------------------------------------------------------------------------
# Customer lookup
# ---------------------------------------------------------------------------

@app.callback(
    Output("customer-details", "children"),
    Input("customer-id-dropdown", "value"),
    State("scored-store", "data"),
)
def display_customer_details(customer_id, scored_records):
    if customer_id is None:
        return html.Div("Select a CustomerID to view details.", style={"color": "#6b7a88"})

    df = pd.DataFrame(scored_records)
    matches = df[df["customerid"] == customer_id]
    if matches.empty:
        return html.Div("No data found for this CustomerID.", style={"color": "#B00020"})
    row = matches.iloc[0]

    mongo_doc = data_store.fetch_customer(customer_id)

    prob_pct = round(row["churn_probability"] * 100, 1)

    return html.Div(
        [
            html.Div(
                [
                    html.H3("Personal Information", style={"color": NAVY, "marginBottom": "10px"}),
                    html.P(f"Customer ID: {row['customerid']}"),
                    html.P(f"Name: {row.get('customer_name', 'N/A')}"),
                    html.P(f"Age: {row.get('age', 'N/A')}"),
                    html.P(f"Gender: {row.get('gender', 'N/A')}"),
                ],
                style={**CARD_STYLE, "backgroundColor": CARD_BLUE_TINT, "marginBottom": "15px"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Activity Information", style={"color": NAVY, "marginBottom": "10px"}),
                            html.P(f"Tenure: {row.get('tenure', 'N/A')} months"),
                            html.P(f"Usage Frequency: {row.get('usage_frequency', 'N/A')}"),
                            html.P(f"Support Calls: {row.get('support_calls', 'N/A')}"),
                        ],
                        style={**CARD_STYLE, "backgroundColor": CARD_GRAY, "flex": "1"},
                    ),
                    html.Div(
                        [
                            html.H3("Payment Details", style={"color": NAVY, "marginBottom": "10px"}),
                            html.P(f"Payment Delay: {row.get('payment_delay', 'N/A')} days"),
                            html.P(f"Subscription Type: {row.get('subscription_type', 'N/A')}"),
                            html.P(f"Contract Length: {row.get('contract_length', 'N/A')}"),
                            html.P(f"Total Spend: ${row.get('total_spend', 0):,.2f}"),
                            html.P(f"Last Interaction: {row.get('last_interaction', 'N/A')} days ago"),
                        ],
                        style={**CARD_STYLE, "backgroundColor": CARD_WHITE_TINT, "flex": "1"},
                    ),
                ],
                style={"display": "flex", "gap": "16px", "marginBottom": "15px"},
            ),
            html.Div(
                [
                    html.H3("Churn Risk", style={"color": NAVY, "marginBottom": "10px"}),
                    html.Div(
                        [
                            html.Span(f"{prob_pct}% predicted probability", style={"fontSize": "1.3em", "fontWeight": "bold",
                                                                                     "marginRight": "12px", "color": NAVY}),
                            risk_badge(row["churn_classification"]),
                        ]
                    ),
                    html.Div(
                        style={"backgroundColor": "#eee", "borderRadius": "999px", "height": "10px", "marginTop": "10px"},
                        children=html.Div(style={
                            "width": f"{prob_pct}%", "backgroundColor": risk_color(row["churn_classification"]),
                            "height": "10px", "borderRadius": "999px",
                        }),
                    ),
                ],
                style={**CARD_STYLE, "marginBottom": "15px"},
            ),
            html.Div(
                [
                    html.H3("Recommended Retention Actions", style={"color": NAVY, "marginBottom": "10px"}),
                    html.Ul([html.Li(a) for a in cm.recommend_actions(row)], style={"lineHeight": "1.8"}),
                ],
                style={**CARD_STYLE, "backgroundColor": CARD_BLUE_TINT},
            ),
            html.P(
                f"Live database record found (MongoDB)." if mongo_doc else "",
                style={"color": "#6b7a88", "fontSize": "0.8em", "marginTop": "10px"},
            ) if data_store.mongo_enabled() else None,
        ]
    )


if __name__ == "__main__":
    app.run(debug=True)
