"""
dashboard.py

Honeywell SentinelAI hackathon project.

A commercial-grade Streamlit SOC (Security Operations Center) dashboard for
visualizing and analyzing the pre-processed, pre-scored login-event dataset.
This dashboard is read-only: it never retrains the anomaly detection model,
it only loads the artifacts already produced by generate_data.py and
train_model.py and renders interactive analytics on top of them.

Inputs (already produced upstream):
    data/processed/processed_logs.csv
    models/anomaly_model.pkl
    models/scaler.pkl
    models/label_encoders.pkl

Run with:
    streamlit run dashboard.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------- #
# Paths and constants
# --------------------------------------------------------------------------- #

PROCESSED_DATA_PATH = Path("data/processed/processed_logs.csv")
MODEL_PATH = Path("models/anomaly_model.pkl")
SCALER_PATH = Path("models/scaler.pkl")
ENCODERS_PATH = Path("models/label_encoders.pkl")

# Columns that were label-encoded to integers by train_model.py. The stored
# LabelEncoders are used to decode them back to human-readable text for
# display, filtering, and charting purposes.
ENCODED_COLUMNS: List[str] = [
    "entity_type",
    "geo_location",
    "resource_accessed",
    "auth_method",
    "command_sequence",
    "browser",
    "operating_system",
    "login_result",
]

RISK_LEVEL_ORDER = ["Low", "Medium", "High", "Critical"]

RISK_LEVEL_COLORS = {
    "Low": "#22C55E",
    "Medium": "#FACC15",
    "High": "#FB923C",
    "Critical": "#EF4444",
}

# --------------------------------------------------------------------------- #
# Theme: professional dark-blue SOC look with white rounded cards
# --------------------------------------------------------------------------- #

CUSTOM_CSS = """
<style>
:root {
    --sentinel-navy: #0B1E3F;
    --sentinel-navy-deep: #071431;
    --sentinel-blue: #1E56A0;
    --sentinel-accent: #3AA0FF;
    --sentinel-card: #FFFFFF;
}

.stApp {
    background: linear-gradient(180deg, var(--sentinel-navy-deep) 0%, var(--sentinel-navy) 100%);
}

section[data-testid="stSidebar"] {
    background-color: var(--sentinel-navy-deep);
    border-right: 1px solid rgba(58, 160, 255, 0.25);
}

section[data-testid="stSidebar"] * {
    color: #E6EEFA !important;
}

h1, h2, h3, h4, h5, h6 {
    color: #F0F6FF !important;
    font-family: "Segoe UI", "Helvetica Neue", sans-serif;
}

p, span, label, .stMarkdown {
    color: #DCE6F5;
}

div[data-testid="stMetric"] {
    background-color: var(--sentinel-card);
    border-radius: 14px;
    padding: 16px 18px;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
    border: 1px solid rgba(30, 86, 160, 0.15);
}

div[data-testid="stMetric"] label,
div[data-testid="stMetric"] div {
    color: var(--sentinel-navy-deep) !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: rgba(255, 255, 255, 0.03);
    border-radius: 16px;
    border: 1px solid rgba(58, 160, 255, 0.18);
}

.sentinel-logo {
    font-size: 30px;
    font-weight: 800;
    color: #F0F6FF;
    letter-spacing: 0.5px;
}

.sentinel-subtitle {
    color: #9FBFEA;
    font-size: 13px;
    margin-top: -6px;
}

hr {
    border-color: rgba(58, 160, 255, 0.25);
}
</style>
"""


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner="Loading SentinelAI dataset...")
def load_data(path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Load the processed dataset and decode label-encoded columns.

    Args:
        path: Path to the processed CSV produced by train_model.py.

    Returns:
        Dataframe with human-readable categorical columns and a parsed
        timestamp column.

    Raises:
        FileNotFoundError: If the processed dataset does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(str(path))

    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    encoders = _load_encoders()
    if encoders:
        for column in ENCODED_COLUMNS:
            if column in df.columns and column in encoders:
                encoder = encoders[column]
                # Encoded values are integer codes into encoder.classes_.
                codes = df[column].astype(int).clip(0, len(encoder.classes_) - 1)
                df[column] = encoder.inverse_transform(codes)

    return df


@st.cache_resource(show_spinner=False)
def _load_encoders() -> Optional[Dict[str, object]]:
    """Load the fitted LabelEncoders used to decode categorical columns.

    Returns:
        Dict of encoders keyed by column name, or None if not found.
    """
    if not ENCODERS_PATH.exists():
        return None
    return joblib.load(ENCODERS_PATH)


# --------------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------------- #

def apply_filters(df: pd.DataFrame, filters: Dict[str, object]) -> pd.DataFrame:
    """Apply the sidebar filter selections to the dataframe.

    Args:
        df: Full processed dataframe.
        filters: Dict of filter selections keyed by column name, plus a
            "date_range" key holding a (start_date, end_date) tuple.

    Returns:
        Filtered dataframe.
    """
    filtered = df.copy()

    multiselect_columns = [
        "entity_type", "geo_location", "attack_type",
        "risk_level", "browser", "operating_system", "auth_method",
    ]
    for column in multiselect_columns:
        selected = filters.get(column)
        if selected:
            filtered = filtered[filtered[column].isin(selected)]

    date_range = filters.get("date_range")
    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        filtered = filtered[
            (filtered["timestamp"].dt.date >= start_date)
            & (filtered["timestamp"].dt.date <= end_date)
        ]

    return filtered


# --------------------------------------------------------------------------- #
# KPI cards
# --------------------------------------------------------------------------- #

def create_kpi_cards(df: pd.DataFrame) -> None:
    """Render the top-level KPI metric cards.

    Args:
        df: Filtered dataframe to summarize.
    """
    total_events = len(df)
    normal_events = int((df["prediction"] == 0).sum())
    anomalies = int((df["prediction"] == 1).sum())
    avg_risk = df["risk_score"].mean() if total_events else 0.0
    critical_alerts = int((df["risk_level"] == "Critical").sum())

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("🧾 Total Events", f"{total_events:,}")
    col2.metric("✅ Normal Events", f"{normal_events:,}")
    col3.metric("🚨 Anomalies", f"{anomalies:,}")
    col4.metric("📊 Average Risk Score", f"{avg_risk:.1f}")
    col5.metric("🔥 Critical Alerts", f"{critical_alerts:,}")


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #

def create_attack_chart(df: pd.DataFrame) -> go.Figure:
    """Build the attack type distribution pie chart.

    Args:
        df: Filtered dataframe.

    Returns:
        Plotly pie chart figure.
    """
    counts = df["attack_type"].value_counts().reset_index()
    counts.columns = ["attack_type", "count"]
    fig = px.pie(
        counts, names="attack_type", values="count",
        title="Attack Type Distribution", hole=0.0,
        color_discrete_sequence=px.colors.sequential.Blues_r,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#0B1E3F")
    return fig


def create_risk_chart(df: pd.DataFrame) -> go.Figure:
    """Build the risk level distribution donut chart.

    Args:
        df: Filtered dataframe.

    Returns:
        Plotly donut chart figure.
    """
    counts = df["risk_level"].value_counts().reindex(RISK_LEVEL_ORDER).fillna(0)
    fig = px.pie(
        names=counts.index, values=counts.values, hole=0.55,
        title="Risk Level Distribution",
        color=counts.index,
        color_discrete_map=RISK_LEVEL_COLORS,
    )
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#0B1E3F")
    return fig


def create_timeline(df: pd.DataFrame) -> go.Figure:
    """Build the login events over time line chart.

    Args:
        df: Filtered dataframe.

    Returns:
        Plotly line chart figure.
    """
    timeline = (
        df.assign(date=df["timestamp"].dt.date)
        .groupby(["date", "prediction"])
        .size()
        .reset_index(name="count")
    )
    timeline["status"] = timeline["prediction"].map({0: "Normal", 1: "Anomaly"})

    fig = px.line(
        timeline, x="date", y="count", color="status",
        title="Login Timeline — Events Over Time", markers=True,
        color_discrete_map={"Normal": "#3AA0FF", "Anomaly": "#EF4444"},
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#E6EEFA",
                       plot_bgcolor="rgba(255,255,255,0.04)")
    return fig


def create_top_risk_chart(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Build the top-N riskiest entities horizontal bar chart.

    Args:
        df: Filtered dataframe.
        top_n: Number of top entities to display.

    Returns:
        Plotly horizontal bar chart figure.
    """
    top_risk = (
        df.groupby("entity_id")["risk_score"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )
    fig = px.bar(
        top_risk, x="risk_score", y="entity_id", orientation="h",
        title=f"Top {top_n} Risk Users", color="risk_score",
        color_continuous_scale="Reds",
    )
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        paper_bgcolor="rgba(0,0,0,0)", font_color="#E6EEFA",
        plot_bgcolor="rgba(255,255,255,0.04)",
    )
    return fig


def create_geo_chart(df: pd.DataFrame) -> go.Figure:
    """Build the country vs number-of-events bar chart.

    Args:
        df: Filtered dataframe.

    Returns:
        Plotly bar chart figure.
    """
    counts = df["geo_location"].value_counts().reset_index()
    counts.columns = ["country", "count"]
    fig = px.bar(
        counts, x="country", y="count", title="Geo Distribution",
        color="count", color_continuous_scale="Blues",
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#E6EEFA",
                       plot_bgcolor="rgba(255,255,255,0.04)")
    return fig


def create_resource_chart(df: pd.DataFrame) -> go.Figure:
    """Build the resource access distribution treemap.

    Args:
        df: Filtered dataframe.

    Returns:
        Plotly treemap figure.
    """
    counts = df["resource_accessed"].value_counts().reset_index()
    counts.columns = ["resource_accessed", "count"]
    fig = px.treemap(
        counts, path=["resource_accessed"], values="count",
        title="Resource Access Distribution",
        color="count", color_continuous_scale="Blues",
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#0B1E3F")
    return fig


def create_os_chart(df: pd.DataFrame) -> go.Figure:
    """Build the operating system distribution pie chart.

    Args:
        df: Filtered dataframe.

    Returns:
        Plotly pie chart figure.
    """
    counts = df["operating_system"].value_counts().reset_index()
    counts.columns = ["operating_system", "count"]
    fig = px.pie(
        counts, names="operating_system", values="count",
        title="Operating System Distribution",
        color_discrete_sequence=px.colors.sequential.Teal,
    )
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#0B1E3F")
    return fig


def create_browser_chart(df: pd.DataFrame) -> go.Figure:
    """Build the browser distribution pie chart.

    Args:
        df: Filtered dataframe.

    Returns:
        Plotly pie chart figure.
    """
    counts = df["browser"].value_counts().reset_index()
    counts.columns = ["browser", "count"]
    fig = px.pie(
        counts, names="browser", values="count",
        title="Browser Distribution",
        color_discrete_sequence=px.colors.sequential.Purp,
    )
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#0B1E3F")
    return fig


def create_auth_chart(df: pd.DataFrame) -> go.Figure:
    """Build the authentication method distribution bar chart.

    Args:
        df: Filtered dataframe.

    Returns:
        Plotly bar chart figure.
    """
    counts = df["auth_method"].value_counts().reset_index()
    counts.columns = ["auth_method", "count"]
    fig = px.bar(
        counts, x="auth_method", y="count", title="Authentication Method Distribution",
        color="count", color_continuous_scale="Blues",
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#E6EEFA",
                       plot_bgcolor="rgba(255,255,255,0.04)")
    return fig


def create_entity_chart(df: pd.DataFrame) -> go.Figure:
    """Build the entity type distribution bar chart.

    Args:
        df: Filtered dataframe.

    Returns:
        Plotly bar chart figure.
    """
    counts = df["entity_type"].value_counts().reset_index()
    counts.columns = ["entity_type", "count"]
    fig = px.bar(
        counts, x="entity_type", y="count", title="Entity Type Distribution",
        color="entity_type", color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#E6EEFA",
                       plot_bgcolor="rgba(255,255,255,0.04)")
    return fig


def create_heatmap(df: pd.DataFrame) -> go.Figure:
    """Build the hour-vs-weekday login event heatmap.

    Args:
        df: Filtered dataframe.

    Returns:
        Plotly heatmap figure.
    """
    weekday_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    pivot = (
        df.assign(weekday=df["timestamp"].dt.weekday, hour=df["timestamp"].dt.hour)
        .groupby(["weekday", "hour"])
        .size()
        .reset_index(name="count")
        .pivot(index="weekday", columns="hour", values="count")
        .reindex(index=range(7), columns=range(24), fill_value=0)
    )
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=list(range(24)),
            y=weekday_labels,
            colorscale="Blues",
            colorbar=dict(title="Events"),
        )
    )
    fig.update_layout(
        title="Daily Heatmap — Hour vs Weekday",
        xaxis_title="Hour of Day", yaxis_title="Weekday",
        paper_bgcolor="rgba(0,0,0,0)", font_color="#E6EEFA",
        plot_bgcolor="rgba(255,255,255,0.04)",
    )
    return fig


def create_gauge(df: pd.DataFrame) -> go.Figure:
    """Build the average risk score gauge indicator.

    Args:
        df: Filtered dataframe.

    Returns:
        Plotly indicator gauge figure.
    """
    avg_risk = df["risk_score"].mean() if len(df) else 0.0
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=avg_risk,
            title={"text": "Average Risk Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1E56A0"},
                "steps": [
                    {"range": [0, 25], "color": "#22C55E"},
                    {"range": [25, 50], "color": "#FACC15"},
                    {"range": [50, 75], "color": "#FB923C"},
                    {"range": [75, 100], "color": "#EF4444"},
                ],
            },
        )
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#E6EEFA", height=300)
    return fig


# --------------------------------------------------------------------------- #
# Threat explorer table + suspicious event detail
# --------------------------------------------------------------------------- #

def create_risk_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build the Threat Explorer table of anomalies sorted by risk score.

    Args:
        df: Filtered dataframe.

    Returns:
        Dataframe of anomalies only, sorted by risk_score descending, with
        display-friendly columns.
    """
    anomalies = df[df["prediction"] == 1].copy()
    anomalies = anomalies.sort_values("risk_score", ascending=False)

    display_columns = [
        "timestamp", "entity_id", "geo_location", "attack_type",
        "risk_score", "risk_level", "explanation",
    ]
    return anomalies[display_columns].rename(columns={"geo_location": "country"})


def render_event_detail(row: pd.Series) -> None:
    """Render the full detail view for a single selected suspicious event.

    Args:
        row: The selected row (full original columns) from the dataframe.
    """
    st.markdown("#### 🔍 Suspicious Event Detail")
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        col1.markdown(f"**Entity:** {row['entity_id']}")
        col1.markdown(f"**Country:** {row['geo_location']}")
        col1.markdown(f"**IP Address:** {row['source_ip']}")

        col2.markdown(f"**Resource:** {row['resource_accessed']}")
        col2.markdown(f"**Browser:** {row['browser']}")
        col2.markdown(f"**Operating System:** {row['operating_system']}")

        col3.markdown(f"**Command Sequence:** {row['command_sequence']}")
        col3.markdown(f"**Risk Score:** {row['risk_score']} ({row['risk_level']})")
        prediction_label = "Anomaly" if row["prediction"] == 1 else "Normal"
        col3.markdown(f"**Prediction:** {prediction_label}")

        st.markdown(f"**Explanation:** {row['explanation']}")


# --------------------------------------------------------------------------- #
# Executive summary
# --------------------------------------------------------------------------- #

def create_summary(df: pd.DataFrame) -> Dict[str, object]:
    """Compute the automatically generated executive summary.

    Args:
        df: Filtered dataframe.

    Returns:
        Dict of summary statistics ready for display.
    """
    if df.empty:
        return {
            "total": 0, "normal": 0, "anomalies": 0,
            "highest_risk_country": "N/A", "most_common_attack": "N/A",
            "highest_risk_user": "N/A", "avg_risk": 0.0, "critical_alerts": 0,
        }

    anomalies_df = df[df["prediction"] == 1]
    attack_counts = anomalies_df["attack_type"][anomalies_df["attack_type"] != "Normal"]

    return {
        "total": len(df),
        "normal": int((df["prediction"] == 0).sum()),
        "anomalies": int((df["prediction"] == 1).sum()),
        "highest_risk_country": (
            df.groupby("geo_location")["risk_score"].mean().idxmax()
            if not df.empty else "N/A"
        ),
        "most_common_attack": (
            attack_counts.mode().iloc[0] if not attack_counts.empty else "N/A"
        ),
        "highest_risk_user": (
            df.groupby("entity_id")["risk_score"].mean().idxmax()
            if not df.empty else "N/A"
        ),
        "avg_risk": df["risk_score"].mean(),
        "critical_alerts": int((df["risk_level"] == "Critical").sum()),
    }


def render_summary(summary: Dict[str, object]) -> None:
    """Render the executive summary as a readable text block.

    Args:
        summary: Dict produced by create_summary().
    """
    st.markdown("#### 📋 Executive Summary")
    with st.container(border=True):
        st.markdown(
            f"""
- **Total Events:** {summary['total']:,}
- **Normal:** {summary['normal']:,}
- **Anomalies:** {summary['anomalies']:,}
- **Highest Risk Country:** {summary['highest_risk_country']}
- **Most Common Attack:** {summary['most_common_attack']}
- **Highest Risk User:** {summary['highest_risk_user']}
- **Average Risk Score:** {summary['avg_risk']:.1f}
- **Critical Alerts:** {summary['critical_alerts']:,}
            """
        )


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

def render_sidebar(df: pd.DataFrame) -> Dict[str, object]:
    """Render the sidebar (logo, navigation, filters) and collect selections.

    Args:
        df: Full unfiltered dataframe, used to populate filter options.

    Returns:
        Dict containing the selected page and all filter selections.
    """
    with st.sidebar:
        st.markdown('<div class="sentinel-logo">🛡️ SentinelAI</div>', unsafe_allow_html=True)
        st.markdown('<div class="sentinel-subtitle">Honeywell Cyber Defense Unit</div>', unsafe_allow_html=True)
        st.markdown("---")

        page = st.radio(
            "Navigation",
            ["Dashboard", "Threat Analytics", "Risk Explorer", "About"],
            index=0,
        )

        st.markdown("---")
        st.markdown("**Filters**")

        min_date = df["timestamp"].min().date()
        max_date = df["timestamp"].max().date()
        date_range = st.date_input(
            "Date Range", value=(min_date, max_date),
            min_value=min_date, max_value=max_date,
        )

        filters = {
            "entity_type": st.multiselect("Entity Type", sorted(df["entity_type"].unique())),
            "geo_location": st.multiselect("Country", sorted(df["geo_location"].unique())),
            "attack_type": st.multiselect("Attack Type", sorted(df["attack_type"].unique())),
            "risk_level": st.multiselect("Risk Level", RISK_LEVEL_ORDER),
            "browser": st.multiselect("Browser", sorted(df["browser"].unique())),
            "operating_system": st.multiselect("Operating System", sorted(df["operating_system"].unique())),
            "auth_method": st.multiselect("Authentication Method", sorted(df["auth_method"].unique())),
            "date_range": date_range,
        }
        filters["page"] = page

    return filters


# --------------------------------------------------------------------------- #
# Page renderers
# --------------------------------------------------------------------------- #

def render_dashboard_page(df: pd.DataFrame) -> None:
    """Render the main Dashboard page: KPIs, gauge, summary, core charts.

    Args:
        df: Filtered dataframe.
    """
    st.subheader("📊 Overview")
    create_kpi_cards(df)
    st.markdown("")

    col1, col2 = st.columns([1, 2])
    with col1:
        with st.container(border=True):
            st.plotly_chart(create_gauge(df), use_container_width=True)
        with st.container(border=True):
            render_summary(create_summary(df))
    with col2:
        with st.container(border=True):
            st.plotly_chart(create_timeline(df), use_container_width=True)
        with st.container(border=True):
            st.plotly_chart(create_heatmap(df), use_container_width=True)

    st.markdown("")
    col3, col4, col5 = st.columns(3)
    with col3:
        with st.container(border=True):
            st.plotly_chart(create_attack_chart(df), use_container_width=True)
    with col4:
        with st.container(border=True):
            st.plotly_chart(create_risk_chart(df), use_container_width=True)
    with col5:
        with st.container(border=True):
            st.plotly_chart(create_entity_chart(df), use_container_width=True)


def render_threat_analytics_page(df: pd.DataFrame) -> None:
    """Render the Threat Analytics page: geo, resources, devices, top users.

    Args:
        df: Filtered dataframe.
    """
    st.subheader("🛰️ Threat Analytics")

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.plotly_chart(create_geo_chart(df), use_container_width=True)
    with col2:
        with st.container(border=True):
            st.plotly_chart(create_top_risk_chart(df), use_container_width=True)

    with st.container(border=True):
        st.plotly_chart(create_resource_chart(df), use_container_width=True)

    col3, col4, col5 = st.columns(3)
    with col3:
        with st.container(border=True):
            st.plotly_chart(create_os_chart(df), use_container_width=True)
    with col4:
        with st.container(border=True):
            st.plotly_chart(create_browser_chart(df), use_container_width=True)
    with col5:
        with st.container(border=True):
            st.plotly_chart(create_auth_chart(df), use_container_width=True)


def render_risk_explorer_page(df: pd.DataFrame) -> None:
    """Render the Risk Explorer page: threat table, detail view, downloads.

    Args:
        df: Filtered dataframe.
    """
    st.subheader("🕵️ Threat Explorer")
    st.caption("Anomalous events only, sorted by highest risk score.")

    table = create_risk_table(df)

    with st.container(border=True):
        event = st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="threat_table",
        )

    selected_rows = event.selection.rows if event and event.selection else []
    if selected_rows:
        selected_index = table.index[selected_rows[0]]
        full_row = df.loc[selected_index]
        render_event_detail(full_row)
    else:
        st.info("Select a row above to view full suspicious event detail.")

    st.markdown("")
    st.markdown("#### ⬇️ Downloads")
    col1, col2 = st.columns(2)
    col1.download_button(
        "Download Processed Dataset (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="processed_logs_filtered.csv",
        mime="text/csv",
    )
    col2.download_button(
        "Download Anomalies Only (CSV)",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name="anomalies_only.csv",
        mime="text/csv",
    )


def render_about_page() -> None:
    """Render the static About page describing the project."""
    st.subheader("ℹ️ About SentinelAI")
    with st.container(border=True):
        st.markdown(
            """
**Honeywell SentinelAI** is an AI-powered behavioural anomaly detection
platform for enterprise authentication logs.

The pipeline behind this dashboard:

1. **`generate_data.py`** — synthesizes realistic enterprise login events
   across users, service accounts, and edge devices, injecting seven
   distinct attack patterns (brute force, impossible travel, credential
   stuffing, device spoofing, lateral movement, low-and-slow exfiltration,
   and insider drift).
2. **`train_model.py`** — engineers behavioural features, encodes
   categorical fields, trains an Isolation Forest anomaly detector, and
   generates an explainable, weighted risk score for every event.
3. **`dashboard.py`** (this app) — visualizes the processed results as a
   Security Operations Center (SOC) style analytics dashboard, without
   ever retraining the model.

Use the sidebar filters to narrow the dataset by entity type, country,
attack type, risk level, browser, operating system, authentication method,
and date range. Every chart on every page updates automatically.
            """
        )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    """Application entry point."""
    st.set_page_config(
        page_title="Honeywell SentinelAI",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.title("🛡️ Honeywell SentinelAI")
    st.caption("AI Powered Behavioural Anomaly Detection")

    try:
        df = load_data()
    except FileNotFoundError:
        st.error(
            f"Processed dataset not found at `{PROCESSED_DATA_PATH}`. "
            "Run generate_data.py and train_model.py first."
        )
        return

    filters = render_sidebar(df)
    filtered_df = apply_filters(df, filters)

    if filtered_df.empty:
        st.warning("No events match the current filter selection.")
        return

    page = filters["page"]
    if page == "Dashboard":
        render_dashboard_page(filtered_df)
    elif page == "Threat Analytics":
        render_threat_analytics_page(filtered_df)
    elif page == "Risk Explorer":
        render_risk_explorer_page(filtered_df)
    else:
        render_about_page()


if __name__ == "__main__":
    main()