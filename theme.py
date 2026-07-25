# Core palette (carried over from the original app + graph scripts)
NAVY = "#011D35"
BLUE = "#35A4FE"
LIGHT_BLUE = "#D1EAFF"
PANEL_GRAY = "#f9f9f9"
CARD_GRAY = "#f3f4f7"
CARD_BLUE_TINT = "#e8f0fe"
CARD_WHITE_TINT = "#f9fafb"
WHITE = "#ffffff"

RISK_HIGH = "#FF0000"
RISK_HIGH_BG = "#FFCCCC"
RISK_MEDIUM = "#B8860B"
RISK_MEDIUM_BG = "#FFF3CD"
RISK_LOW = "#1E7B34"
RISK_LOW_BG = "#DFF5E1"

FONT_FAMILY = "Arial, sans-serif"

CARD_STYLE = {
    "backgroundColor": WHITE,
    "borderRadius": "10px",
    "padding": "18px",
    "boxShadow": "0 2px 5px rgba(0, 0, 0, 0.08)",
}

PANEL_STYLE = {
    "backgroundColor": PANEL_GRAY,
    "border": "none",
    "borderRadius": "10px",
    "padding": "15px",
    "marginBottom": "20px",
}

SECTION_HEADER_STYLE = {
    "color": NAVY,
    "borderBottom": f"3px solid {BLUE}",
    "paddingBottom": "6px",
    "display": "inline-block",
}

PRIMARY_BUTTON_STYLE = {
    "marginTop": "5px",
    "padding": "8px 20px",
    "height": "38px",
    "fontSize": "1em",
    "fontWeight": "bold",
    "color": WHITE,
    "backgroundColor": BLUE,
    "border": "none",
    "borderRadius": "10px",
    "cursor": "pointer",
}

SECONDARY_BUTTON_STYLE = {
    **PRIMARY_BUTTON_STYLE,
    "backgroundColor": WHITE,
    "color": BLUE,
    "border": f"2px solid {BLUE}",
}

TAB_STYLE = {
    "padding": "10px 18px",
    "fontSize": "1.05em",
    "fontWeight": "bold",
    "color": BLUE,
    "backgroundColor": LIGHT_BLUE,
    "marginRight": "8px",
    "border": "none",
    "borderRadius": "10px",
}

TAB_SELECTED_STYLE = {
    **TAB_STYLE,
    "color": WHITE,
    "backgroundColor": BLUE,
}


def risk_color(classification: str) -> str:
    return {
        "High Churn": RISK_HIGH,
        "Medium Churn": RISK_MEDIUM,
        "Low Churn": RISK_LOW,
    }.get(classification, NAVY)


def risk_bg(classification: str) -> str:
    return {
        "High Churn": RISK_HIGH_BG,
        "Medium Churn": RISK_MEDIUM_BG,
        "Low Churn": RISK_LOW_BG,
    }.get(classification, WHITE)
