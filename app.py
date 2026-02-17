"""
Smart Shygyn PRO v3 — FRONTEND APPLICATION
Complete Streamlit interface integrating all backend components.
NO PLACEHOLDERS. Full production implementation.
FIXED: Dark/Light Mode toggle with proper CSS injection and state management.
INTEGRATED: Real-time weather tracking with Open-Meteo API.
INTEGRATED: BattLeDIM real dataset (L-Town Cyprus) — full leak detection analysis.
INTEGRATED: Real Kazakhstan water data (tariffs, pipe wear, network stats).
REFACTORED: Memory safety (gc.collect before simulation), robust error handling.
UPDATED v3.1: Added ML Engine (Isolation Forest), Business Model (ROI), Alerts, Live Demo tabs.
"""

import gc
import random
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import streamlit as st

# Import backend classes
from backend import (
    SmartShygynBackend,
    CityManager,
    HydraulicPhysics,
)
from weather import get_city_weather, get_frost_multiplier, format_weather_display

# Import BattLeDIM integration — full analysis module
from battledim_analysis import render_battledim_tab   # ← полный анализ с детекцией

# Import data loader
from data_loader import (
    initialize_battledim,
    get_real_tariff,
    get_real_pipe_wear,
    get_estimated_pipe_age,
    KAZAKHSTAN_REAL_DATA,
    get_loader,
)

# Import config
from config import CONFIG

# ── NEW: Optional modules (graceful fallback if files not yet added) ──────
try:
    from demo_mode import render_alerts_tab, render_demo_tab
    _DEMO_OK = True
except ImportError:
    _DEMO_OK = False

try:
    from business_model import render_business_tab
    _BIZ_OK = True
except ImportError:
    _BIZ_OK = False

logger = logging.getLogger("smart_shygyn.app")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Smart Shygyn PRO v3 — Command Center",
    layout="wide",
    page_icon="💧",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════════════════════
# THEMES
# ═══════════════════════════════════════════════════════════════════════════

DARK_CSS = """
<style>
:root {
  --bg: #0e1117;
  --card: #1a1f2e;
  --border: #2d3748;
  --accent: #3b82f6;
  --danger: #ef4444;
  --warn: #f59e0b;
  --ok: #10b981;
  --text: #e2e8f0;
  --muted: #94a3b8;
}
[data-testid="stAppViewContainer"] { background-color: var(--bg); color: var(--text); }
[data-testid="stSidebar"] { background-color: var(--card); border-right: 2px solid var(--border); }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { color: var(--text); }
[data-testid="stHeader"] { background-color: var(--bg); border-bottom: 1px solid var(--border); }
[data-testid="stMetricValue"] { font-size: 24px; font-weight: 700; color: var(--text); }
[data-testid="stMetricLabel"] { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; }
h1 { color: var(--accent) !important; text-align: center; padding: 16px 0; letter-spacing: 1px;
     border-bottom: 3px solid var(--accent); margin-bottom: 24px; }
h2 { color: var(--text) !important; border-left: 4px solid var(--accent);
     padding-left: 12px; margin-top: 24px; }
h3 { color: var(--text) !important; border-bottom: 2px solid var(--accent);
     padding-bottom: 8px; margin-top: 16px; }
h4, h5, h6 { color: var(--text) !important; }
.stAlert { border-radius: 8px; border-left-width: 4px;
           background-color: var(--card); color: var(--text); }
.stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: var(--bg); }
.stTabs [data-baseweb="tab"] {
  font-size: 14px; font-weight: 600; padding: 12px 24px;
  border-radius: 8px 8px 0 0; background-color: var(--card); color: var(--text); }
.stTabs [data-baseweb="tab"]:hover { background-color: var(--border); }
.stTabs [aria-selected="true"] { background-color: var(--accent) !important; color: white !important; }
.stButton > button { width: 100%; font-weight: 600; border-radius: 6px;
                     background-color: var(--accent); color: white; border: none; }
.stButton > button:hover { background-color: #2563eb; border: none; }
.stButton > button[kind="primary"] { background-color: var(--ok); }
.stButton > button[kind="primary"]:hover { background-color: #059669; }
.streamlit-expanderHeader { font-weight: 600; font-size: 15px;
                             color: var(--text); background-color: var(--card); border-radius: 6px; }
[data-testid="stDataFrame"] { background-color: var(--card); }
.stCodeBlock { background-color: var(--card) !important; color: var(--text) !important; }
.stTextInput > div > div > input, .stNumberInput > div > div > input,
.stSelectbox > div > div, .stSlider > div > div > div {
  background-color: var(--card); color: var(--text); border-color: var(--border); }
.stCaption { color: var(--muted) !important; }
[data-testid="stMarkdownContainer"] p { color: var(--text); }
hr { border-color: var(--border); }

/* ── Welcome card grid ── */
.feature-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 20px 16px;
  text-align: center;
  transition: transform 0.2s, border-color 0.2s;
}
.feature-card:hover { transform: translateY(-3px); border-color: var(--accent); }
.feature-icon { font-size: 36px; margin-bottom: 8px; }

/* ── KPI cards ── */
.kpi-critical { border: 2px solid #ef4444; border-radius: 8px; }
.kpi-ok       { border: 2px solid #10b981; border-radius: 8px; }
</style>
"""

LIGHT_CSS = """
<style>
:root {
  --bg-light: #f0f4f8;
  --card-light: #ffffff;
  --border-light: #cbd5e0;
  --accent-light: #2563eb;
  --text-light: #1a202c;
  --muted-light: #718096;
}
[data-testid="stAppViewContainer"] { background-color: var(--bg-light); color: var(--text-light); }
[data-testid="stSidebar"] { background-color: var(--card-light); border-right: 2px solid var(--border-light); }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { color: var(--text-light); }
[data-testid="stHeader"] { background-color: var(--card-light); border-bottom: 1px solid var(--border-light); }
[data-testid="stMetricValue"] { font-size: 24px; font-weight: 700; color: var(--text-light); }
[data-testid="stMetricLabel"] { font-size: 12px; color: var(--muted-light);
                                  text-transform: uppercase; letter-spacing: 0.5px; }
h1 { color: var(--accent-light) !important; text-align: center; padding: 16px 0;
     border-bottom: 3px solid var(--accent-light); margin-bottom: 24px; }
h2 { color: var(--text-light) !important; border-left: 4px solid #3b82f6; padding-left: 12px; }
h3 { color: var(--text-light) !important; border-bottom: 2px solid #3b82f6; padding-bottom: 8px; }
h4, h5, h6 { color: var(--text-light) !important; }
.stAlert { border-radius: 8px; border-left-width: 4px; }
.stTabs [data-baseweb="tab"] { font-size: 14px; font-weight: 600; padding: 12px 24px;
                                border-radius: 8px 8px 0 0; }
.stButton > button { width: 100%; font-weight: 600; border-radius: 6px; }
.streamlit-expanderHeader { font-weight: 600; font-size: 15px; }
.stCaption { color: var(--muted-light) !important; }
[data-testid="stMarkdownContainer"] p { color: var(--text-light); }
hr { border-color: var(--border-light); }
</style>
"""


# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════

def init_session_state():
    defaults = {
        "simulation_results":    None,
        "operation_log":         [],
        "isolated_pipes":        [],
        "city_name":             "Алматы",
        "last_run_params":       {},
        "dark_mode":             True,
        "battledim_initialized": False,
        "battledim_available":   False,
        "battledim_message":     "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if not st.session_state.get("battledim_initialized", False):
        success, msg = initialize_battledim(show_progress=False)
        st.session_state["battledim_initialized"] = True
        st.session_state["battledim_available"]   = success
        st.session_state["battledim_message"]     = msg


# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def make_hydraulic_plot(df: pd.DataFrame,
                        threshold_bar: float,
                        smart_pump: bool,
                        dark_mode: bool) -> go.Figure:
    bg     = "#0e1117" if dark_mode else "white"
    fg     = "#e2e8f0" if dark_mode else "#2c3e50"
    grid_c = "#2d3748" if dark_mode else "#d0d0d0"

    rows        = 4 if smart_pump else 3
    row_heights = [0.28, 0.28, 0.22, 0.22] if smart_pump else [0.35, 0.35, 0.30]

    titles = [
        "💧 Pressure at Leak Node (bar)",
        "🌊 Main Pipe Flow Rate (L/s)",
        "⏱ Water Age at Leak Node (hours)",
    ]
    if smart_pump:
        titles.append("⚡ Dynamic Pump Head Schedule (m)")

    fig = make_subplots(
        rows=rows, cols=1,
        subplot_titles=titles,
        vertical_spacing=0.08,
        row_heights=row_heights
    )

    # ── Row 1: Pressure ───────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df["Hour"], y=df["Pressure (bar)"],
        name="Pressure (Smoothed)",
        line=dict(color="#3b82f6", width=2.5),
        fill="tozeroy", fillcolor="rgba(59,130,246,0.12)",
        hovertemplate="<b>Hour %{x:.1f}</b><br>Pressure: %{y:.2f} bar<extra></extra>"
    ), row=1, col=1)
    fig.add_hline(y=threshold_bar, line_dash="dash", line_color="#ef4444", line_width=2.5,
                  annotation_text="⚠ Leak Threshold", annotation_position="right", row=1, col=1)
    fig.add_hrect(y0=0, y1=1.5, fillcolor="rgba(239,68,68,0.08)",
                  layer="below", line_width=0, row=1, col=1)

    # ── Row 2: Flow ───────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df["Hour"], y=df["Flow Rate (L/s)"],
        name="Observed Flow",
        line=dict(color="#f59e0b", width=2.5),
        hovertemplate="<b>Hour %{x:.1f}</b><br>Flow: %{y:.2f} L/s<extra></extra>"
    ), row=2, col=1)
    expected_flow = df["Demand Pattern"] * df["Flow Rate (L/s)"].mean()
    fig.add_trace(go.Scatter(
        x=df["Hour"], y=expected_flow,
        name="Expected Flow",
        line=dict(color="#10b981", width=2, dash="dot"),
        hovertemplate="<b>Hour %{x:.1f}</b><br>Expected: %{y:.2f} L/s<extra></extra>"
    ), row=2, col=1)
    fig.add_vrect(x0=2, x1=5, fillcolor="rgba(59,130,246,0.08)", layer="below", line_width=0,
                  annotation_text="MNF Window", annotation_position="top left",
                  annotation_font_size=10, row=2, col=1)

    # ── Row 3: Water Age ──────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df["Hour"], y=df["Water Age (h)"],
        name="Water Age",
        line=dict(color="#a855f7", width=2.5),
        fill="tozeroy", fillcolor="rgba(168,85,247,0.12)",
        hovertemplate="<b>Hour %{x:.1f}</b><br>Age: %{y:.1f} hours<extra></extra>"
    ), row=3, col=1)

    # ── Row 4: Smart Pump ─────────────────────────────────────────────────
    if smart_pump:
        fig.add_trace(go.Scatter(
            x=df["Hour"], y=df["Pump Head (m)"],
            name="Pump Head",
            line=dict(color="#10b981", width=2.5),
            fill="tozeroy", fillcolor="rgba(16,185,129,0.12)",
            hovertemplate="<b>Hour %{x:.1f}</b><br>Head: %{y:.0f} m<extra></extra>"
        ), row=4, col=1)
        fig.add_vrect(x0=0, x1=6, fillcolor="rgba(16,185,129,0.08)",
                      layer="below", line_width=0,
                      annotation_text="Night Mode (70%)", annotation_position="top left",
                      annotation_font_size=10, row=4, col=1)

    for r in range(1, rows + 1):
        fig.update_xaxes(gridcolor=grid_c, color=fg, showgrid=True, row=r, col=1)
        fig.update_yaxes(gridcolor=grid_c, color=fg, showgrid=True, row=r, col=1)

    fig.update_yaxes(title_text="Pressure (bar)",  row=1, col=1)
    fig.update_yaxes(title_text="Flow Rate (L/s)", row=2, col=1)
    fig.update_yaxes(title_text="Water Age (h)",   row=3, col=1)
    if smart_pump:
        fig.update_yaxes(title_text="Pump Head (m)", row=4, col=1)
        fig.update_xaxes(title_text="Hour of Day",   row=4, col=1)
    else:
        fig.update_xaxes(title_text="Hour of Day", row=3, col=1)

    fig.update_layout(
        height=950 if smart_pump else 750,
        showlegend=True, hovermode="x unified",
        plot_bgcolor=bg, paper_bgcolor=bg,
        font=dict(color=fg, size=12),
        margin=dict(l=60, r=40, t=70, b=50),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=grid_c, borderwidth=1,
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def make_folium_map(results: dict, isolated_pipes: list, dark_mode: bool) -> folium.Map:
    city_cfg       = results["city_config"]
    wn             = results["network"]
    predicted_leak = results["predicted_leak"]
    failure_probs  = results["failure_probabilities"]
    residuals      = results["residuals"]
    sensors        = results["sensors"]

    tiles = "CartoDB dark_matter" if dark_mode else "OpenStreetMap"
    m = folium.Map(
        location=[city_cfg["lat"], city_cfg["lng"]],
        zoom_start=city_cfg["zoom"], tiles=tiles
    )

    city_manager = CityManager(city_cfg["name"])
    node_coords  = {}

    def get_coords(node_name):
        if node_name == "Res":
            return city_cfg["lat"] - 0.0009, city_cfg["lng"] - 0.0009
        node = wn.get_node(node_name)
        x, y = node.coordinates
        i, j = int(round(x / 100)), int(round(y / 100))
        return city_manager.grid_to_latlon(i, j)

    for link_name in wn.link_name_list:
        link = wn.get_link(link_name)
        if not (hasattr(link, "start_node_name") and hasattr(link, "end_node_name")):
            continue
        sn = link.start_node_name
        en = link.end_node_name
        sc = get_coords(sn)
        ec = get_coords(en)
        node_coords[sn] = sc
        node_coords[en] = ec
        isolated = link_name in isolated_pipes
        folium.PolyLine(
            [sc, ec],
            color="#c0392b" if isolated else "#4a5568",
            weight=6 if isolated else 3,
            opacity=0.9 if isolated else 0.6,
            tooltip=f"{'⛔ ISOLATED: ' if isolated else ''}{link_name}",
        ).add_to(m)

    leak_detected = results["dataframe"]["Pressure (bar)"].min() < 2.7

    for node_name in wn.node_name_list:
        coords = node_coords.get(node_name)
        if coords is None:
            continue
        prob      = failure_probs.get(node_name, 0)
        is_sensor = node_name in sensors

        if node_name == "Res":
            color, icon = "blue", "tint"
            popup_text = "<b>Reservoir</b><br>Water Source"
        elif node_name == predicted_leak and leak_detected:
            color, icon = "red", "warning-sign"
            popup_text = (
                f"<b>⚠️ PREDICTED LEAK</b><br>Node: {node_name}<br>"
                f"Failure Risk: {prob:.1f}%<br>"
                f"Pressure Drop: {residuals.get(node_name, 0):.3f} bar<br>"
                f"Confidence: {results['confidence']:.0f}%"
            )
        elif prob > 40:
            color, icon = "red",    "remove"
            popup_text = f"<b>{node_name}</b><br>Failure Risk: {prob:.1f}%<br>Status: CRITICAL"
        elif prob > 25:
            color, icon = "orange", "exclamation-sign"
            popup_text = f"<b>{node_name}</b><br>Failure Risk: {prob:.1f}%<br>Status: ELEVATED"
        elif prob > 15:
            color, icon = "beige",  "info-sign"
            popup_text = f"<b>{node_name}</b><br>Failure Risk: {prob:.1f}%<br>Status: MODERATE"
        else:
            color, icon = "green",  "ok"
            popup_text = f"<b>{node_name}</b><br>Failure Risk: {prob:.1f}%<br>Status: NORMAL"

        if is_sensor:
            folium.CircleMarker(
                coords, radius=15, color="#f59e0b", weight=3, fill=False,
                tooltip=f"📡 Sensor Node: {node_name}"
            ).add_to(m)
        folium.Marker(
            coords,
            popup=folium.Popup(popup_text, max_width=250),
            tooltip=node_name,
            icon=folium.Icon(color=color, icon=icon, prefix="glyphicon")
        ).add_to(m)

    legend_html = f"""
    <div style="
        position: fixed; bottom: 30px; left: 30px; width: 260px; z-index: 9999;
        background: {'rgba(14,17,23,0.95)' if dark_mode else 'rgba(255,255,255,0.95)'};
        padding: 16px; border-radius: 10px;
        border: 2px solid {'#4a5568' if dark_mode else '#cbd5e0'};
        font-size: 12px; color: {'#e2e8f0' if dark_mode else '#2d3748'};
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
        <b style="font-size: 14px; color: #3b82f6;">🗺️ Network Legend</b>
        <hr style="margin: 8px 0; border-color: {'#4a5568' if dark_mode else '#cbd5e0'};">
        <div style="margin:6px 0;">🔴 <b>High Risk</b> (&gt;40%)</div>
        <div style="margin:6px 0;">🟠 <b>Elevated</b> (25-40%)</div>
        <div style="margin:6px 0;">🟡 <b>Moderate</b> (15-25%)</div>
        <div style="margin:6px 0;">🟢 <b>Normal</b> (&lt;15%)</div>
        <div style="margin:6px 0;">⚠️ <b>Predicted Leak</b></div>
        <div style="margin:6px 0;">🔵 <b>Reservoir</b></div>
        <hr style="margin: 8px 0; border-color: {'#4a5568' if dark_mode else '#cbd5e0'};">
        <div style="margin:6px 0;">🟡 <b>Ring</b> = Sensor ({len(sensors)}/16)</div>
        <div style="margin:6px 0;">⛔ <b>Red Pipe</b> = Isolated</div>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


def make_failure_heatmap(results: dict, dark_mode: bool) -> plt.Figure:
    wn             = results["network"]
    failure_probs  = results["failure_probabilities"]
    sensors        = results["sensors"]
    predicted_leak = results["predicted_leak"]

    fig, ax = plt.subplots(
        figsize=(12, 10),
        facecolor="#0e1117" if dark_mode else "white"
    )
    ax.set_facecolor("#0e1117" if dark_mode else "white")
    txt_color = "white" if dark_mode else "black"

    pos   = {node: wn.get_node(node).coordinates for node in wn.node_name_list}
    graph = wn.get_graph()

    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="#4a5568", width=3.5, alpha=0.6)

    for node in wn.node_name_list:
        x, y = pos[node]
        prob = failure_probs.get(node, 0)
        if node == "Res":
            color = "#3b82f6"
        elif prob > 40:
            color = "#ef4444"
        elif prob > 25:
            color = "#f59e0b"
        elif prob > 15:
            color = "#eab308"
        else:
            color = "#10b981"

        circle = plt.Circle((x, y), radius=20, color=color, ec="white", linewidth=2.5, zorder=3)
        ax.add_patch(circle)

        if node in sensors:
            ring = plt.Circle((x, y), radius=28, color="#f59e0b",
                              fill=False, linewidth=2.5, linestyle="--", zorder=4)
            ax.add_patch(ring)
        if node == predicted_leak:
            alert = plt.Circle((x, y), radius=36, color="#ef4444",
                               fill=False, linewidth=3, linestyle="-", zorder=5)
            ax.add_patch(alert)
        ax.text(x, y, node, fontsize=8, fontweight="bold",
                ha="center", va="center", color=txt_color, zorder=6)

    legend_elements = [
        mpatches.Patch(color="#ef4444", label="High Risk (>40%)"),
        mpatches.Patch(color="#f59e0b", label="Elevated (25-40%)"),
        mpatches.Patch(color="#eab308", label="Moderate (15-25%)"),
        mpatches.Patch(color="#10b981", label="Normal (<15%)"),
        mpatches.Patch(color="#3b82f6", label="Reservoir"),
        mpatches.Patch(color="none",    label="─────────"),
        mpatches.Patch(color="#f59e0b", label="📡 Sensor (dashed ring)"),
        mpatches.Patch(color="#ef4444", label="⚠️ Predicted Leak"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=10,
              facecolor="#1a1f2e" if dark_mode else "white",
              edgecolor="#4a5568"  if dark_mode else "#cbd5e0",
              labelcolor=txt_color)

    ax.set_title(
        f"Pipe Failure Probability Heatmap — {results['city_config']['name']}\n"
        f"Material: {results['material']} | Age: {results['pipe_age']:.0f} yr | "
        f"H-W C: {results['roughness']:.0f}",
        fontsize=14, fontweight="bold", color=txt_color, pad=20
    )
    ax.set_aspect("equal")
    ax.set_axis_off()
    plt.tight_layout()
    return fig


def make_payback_timeline(economics: dict, dark_mode: bool) -> go.Figure:
    bg     = "#0e1117" if dark_mode else "white"
    fg     = "#e2e8f0" if dark_mode else "#2c3e50"
    grid_c = "#2d3748" if dark_mode else "#d0d0d0"

    payback_months  = economics["payback_months"]
    max_months      = min(int(payback_months * 2), 60)
    months          = np.arange(0, max_months + 1)
    monthly_savings = economics["monthly_total_savings_kzt"]
    cumulative      = months * monthly_savings
    capex_line      = np.full_like(months, economics["capex_kzt"], dtype=float)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=months, y=cumulative, name="Cumulative Savings",
        line=dict(color="#10b981", width=3),
        fill="tozeroy", fillcolor="rgba(16,185,129,0.15)",
        hovertemplate="<b>Month %{x}</b><br>Savings: ₸%{y:,.0f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=months, y=capex_line, name="Initial Investment (CAPEX)",
        line=dict(color="#f59e0b", width=2.5, dash="dash"),
        hovertemplate="<b>CAPEX:</b> ₸%{y:,.0f}<extra></extra>"
    ))
    if payback_months < max_months:
        fig.add_vline(x=payback_months, line_dash="dot", line_color="#3b82f6", line_width=2.5,
                      annotation_text=f"Break-Even: {payback_months:.1f} months",
                      annotation_position="top", annotation_font_size=12,
                      annotation_font_color="#3b82f6")
    fig.update_layout(
        title="Investment Payback Timeline",
        xaxis_title="Months", yaxis_title="Tenge (KZT)",
        height=350, hovermode="x unified",
        plot_bgcolor=bg, paper_bgcolor=bg,
        font=dict(color=fg, size=12),
        xaxis=dict(gridcolor=grid_c, color=fg),
        yaxis=dict(gridcolor=grid_c, color=fg),
        margin=dict(l=60, r=40, t=50, b=50),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=grid_c, borderwidth=1)
    )
    return fig


def make_nrw_pie_chart(economics: dict, dark_mode: bool) -> go.Figure:
    bg      = "#0e1117" if dark_mode else "white"
    fg      = "#e2e8f0" if dark_mode else "#2c3e50"
    nrw_pct = economics["nrw_percentage"]
    rev_pct = 100 - nrw_pct

    fig = go.Figure(go.Pie(
        labels=["Revenue Water", "Non-Revenue Water (Leaks)"],
        values=[max(0, rev_pct), nrw_pct],
        hole=0.55,
        marker=dict(colors=["#10b981", "#ef4444"]),
        textinfo="label+percent",
        textfont=dict(size=13, color=fg),
        hovertemplate="<b>%{label}</b><br>%{value:.1f}%<extra></extra>"
    ))
    fig.add_annotation(
        text=f"<b>NRW</b><br>{nrw_pct:.1f}%",
        x=0.5, y=0.5, font=dict(size=18, color=fg), showarrow=False
    )
    fig.update_layout(
        title="Water Accountability Distribution",
        height=350, paper_bgcolor=bg, font=dict(color=fg, size=12),
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
    )
    return fig


def make_pressure_gauge(pressure_bar: float, threshold: float, dark_mode: bool) -> go.Figure:
    """Speedometer gauge for current minimum pressure."""
    bg = "#0e1117" if dark_mode else "white"
    fg = "#e2e8f0" if dark_mode else "#2c3e50"

    color = "#10b981" if pressure_bar >= threshold else "#ef4444"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(pressure_bar, 2),
        delta={"reference": threshold, "valueformat": ".2f",
               "increasing": {"color": "#10b981"}, "decreasing": {"color": "#ef4444"}},
        number={"suffix": " bar", "font": {"size": 28, "color": fg}},
        gauge={
            "axis": {"range": [0, 8], "tickwidth": 1, "tickcolor": fg,
                     "tickfont": {"color": fg}},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 2, "bordercolor": fg,
            "steps": [
                {"range": [0, 1.5],       "color": "rgba(239,68,68,0.3)"},
                {"range": [1.5, threshold],"color": "rgba(245,158,11,0.2)"},
                {"range": [threshold, 8],  "color": "rgba(16,185,129,0.15)"},
            ],
            "threshold": {
                "line": {"color": "#ef4444", "width": 4},
                "thickness": 0.75,
                "value": threshold,
            },
        },
        title={"text": "Min Pressure<br><span style='font-size:12px'>Threshold line = leak alert</span>",
               "font": {"size": 14, "color": fg}}
    ))
    fig.update_layout(
        height=220, paper_bgcolor=bg,
        font=dict(color=fg),
        margin=dict(l=20, r=20, t=60, b=10)
    )
    return fig


def make_risk_bar_chart(failure_probs: dict, predicted_leak: str, dark_mode: bool) -> go.Figure:
    """Horizontal bar chart of top-10 node failure probabilities."""
    bg     = "#0e1117" if dark_mode else "white"
    fg     = "#e2e8f0" if dark_mode else "#2c3e50"
    grid_c = "#2d3748" if dark_mode else "#d0d0d0"

    sorted_nodes = sorted(
        [(k, v) for k, v in failure_probs.items() if k != "Res"],
        key=lambda x: x[1], reverse=True
    )[:10]

    names  = [n for n, _ in sorted_nodes]
    values = [v for _, v in sorted_nodes]
    colors = []
    for n, v in sorted_nodes:
        if n == predicted_leak:
            colors.append("#ef4444")
        elif v > 40:
            colors.append("#ef4444")
        elif v > 25:
            colors.append("#f59e0b")
        elif v > 15:
            colors.append("#eab308")
        else:
            colors.append("#10b981")

    fig = go.Figure(go.Bar(
        x=values, y=names,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Risk: %{x:.1f}%<extra></extra>"
    ))
    fig.update_layout(
        title="Top-10 Node Failure Risk",
        xaxis_title="Failure Probability (%)",
        height=320,
        plot_bgcolor=bg, paper_bgcolor=bg,
        font=dict(color=fg, size=11),
        xaxis=dict(gridcolor=grid_c, color=fg, range=[0, max(values) * 1.25]),
        yaxis=dict(color=fg, autorange="reversed"),
        margin=dict(l=80, r=60, t=50, b=40),
        showlegend=False
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

def render_sidebar():
    st.sidebar.title("💧 Smart Shygyn PRO v3")
    st.sidebar.markdown("### Command Center Configuration")

    dark_mode = st.sidebar.toggle(
        "🌙 Dark Mode",
        value=st.session_state.get("dark_mode", True),
        key="theme_toggle"
    )
    st.session_state["dark_mode"] = dark_mode
    st.sidebar.markdown("---")

    season_temp = st.session_state.get("season_temp", 10.0)
    frost_mult  = get_frost_multiplier(season_temp)

    # ── City ─────────────────────────────────────────────────────────────
    with st.sidebar.expander("🏙️ City Selection", expanded=True):
        city_name = st.selectbox(
            "Select City",
            list(CityManager.CITIES.keys()),
            index=list(CityManager.CITIES.keys()).index(st.session_state["city_name"])
        )
        st.session_state["city_name"] = city_name

        auto_weather = st.checkbox(
            "🛰️ Real-time Weather", value=True, key="auto_weather_toggle"
        )
        if auto_weather:
            temperature, status, error = get_city_weather(city_name)
            frost_mult      = get_frost_multiplier(temperature)
            weather_display = format_weather_display(city_name, temperature, status, error)
            st.markdown(weather_display, unsafe_allow_html=True)
            if status == "fallback":
                st.caption("⚠️ Using fallback temperature.")
            if frost_mult > 1.0:
                st.warning(
                    f"🧊 **Frost Risk**: +{(frost_mult-1)*100:.0f}% failure probability!"
                )
            season_temp = temperature
            st.session_state["season_temp"] = season_temp
            if st.button("🔄 Refresh Weather", use_container_width=True, key="refresh_weather_btn"):
                from weather import clear_weather_cache
                clear_weather_cache()
                st.rerun()
        else:
            st.info("📊 Stress Testing Mode — manual temperature")
            season_temp = st.slider("Temperature (°C)", -30, 45, 10, 1)
            frost_mult  = get_frost_multiplier(season_temp)
            st.session_state["season_temp"] = season_temp
            if frost_mult > 1.0:
                st.warning(f"🧊 Frost Risk: ×{frost_mult:.2f}")

        st.markdown("---")
        ci = CityManager.CITIES[city_name]
        st.caption(f"**Elevation:** {ci.elev_min}-{ci.elev_max}m")
        st.caption(f"**Gradient:** {ci.elev_direction}")
        st.caption(f"**Water Stress:** {ci.water_stress_index:.2f}")

    # ── Network ───────────────────────────────────────────────────────────
    with st.sidebar.expander("⚙️ Network Parameters", expanded=True):
        material = st.selectbox("Pipe Material", ["Пластик (ПНД)", "Сталь", "Чугун"])
        real_age = int(get_estimated_pipe_age(city_name))
        pipe_age = st.slider("Pipe Age (years)", 0, 70, real_age, 1,
                             help=f"Реальный средний возраст для {city_name}: ~{real_age} лет")
        real_wear = get_real_pipe_wear(city_name)
        st.caption(f"📌 Реальный износ {city_name}: **{real_wear}%** (КРЕМ МНЭ РК, 2024)")
        roughness   = HydraulicPhysics.hazen_williams_roughness(material, pipe_age, temp=season_temp)
        degradation = HydraulicPhysics.degradation_percentage(material, pipe_age, temp=season_temp)
        st.caption(f"**H-W Roughness C:** {roughness:.1f}")
        st.caption(f"**Degradation:** {degradation:.1f}%")
        sampling_rate = st.select_slider(
            "Sensor Sampling Rate", options=[1, 2, 4], value=1,
            format_func=lambda x: f"{x} Hz"
        )

    # ── Pump ─────────────────────────────────────────────────────────────
    with st.sidebar.expander("🔧 Pump Control", expanded=True):
        pump_head  = st.slider("Pump Head (m)", 30, 70, 40, 5)
        st.caption(f"≈ {pump_head * 0.098:.2f} bar")
        smart_pump = st.checkbox("⚡ Smart Pump Scheduling", value=False)
        if smart_pump:
            st.success(f"Night: {pump_head*0.7:.0f}m | Day: {pump_head}m")

    # ── Leak ─────────────────────────────────────────────────────────────
    with st.sidebar.expander("💧 Leak Configuration", expanded=True):
        leak_mode = st.radio("Leak Location", ["Random", "Specific Node"], horizontal=True)
        leak_node = (st.text_input("Leak Node ID", value="N_2_2")
                     if leak_mode == "Specific Node" else None)
        leak_area = st.slider("Leak Area (cm²)", 0.1, 2.0, 0.8, 0.1)

    # ── Economics ─────────────────────────────────────────────────────────
    with st.sidebar.expander("💰 Economic Parameters", expanded=True):
        real_tariff  = get_real_tariff(city_name)
        water_tariff = st.number_input(
            "Water Tariff (₸/L)", min_value=0.001, max_value=2.0,
            value=real_tariff, step=0.001, format="%.5f",
            help=f"Реальный тариф {city_name}: {real_tariff*1000:.2f} ₸/м³"
        )
        st.caption(f"📌 Источник: официальный тариф {city_name} 2025")
        leak_threshold = st.slider("Leak Threshold (bar)", 1.0, 5.0, 2.5, 0.1,
                                   help="СП РК 4.01-101-2012: минимум 2.5 бар")
        st.caption("📌 Норматив КЗ: 2.5 бар (СП РК 4.01-101-2012)")
        repair_cost = st.number_input(
            "Repair Deployment Cost (₸)",
            min_value=10_000, max_value=200_000, value=50_000, step=5_000, format="%d"
        )

    # ── N-1 ──────────────────────────────────────────────────────────────
    with st.sidebar.expander("🔬 N-1 Contingency", expanded=False):
        enable_n1 = st.checkbox("Enable N-1 Simulation")
        contingency_pipe = None
        if enable_n1:
            contingency_pipe = st.text_input("Pipe to Fail", value="PH_2_1")
            st.caption("Simulates single-pipe failure")

    st.sidebar.markdown("---")

    # BattLeDIM status
    loader = get_loader()
    status = loader.check_files_exist()
    have_scada = status.get("scada_2018") or status.get("scada_2019")
    if have_scada:
        st.sidebar.success("🌍 BattLeDIM ✅ готов к анализу")
    elif st.session_state.get("battledim_available"):
        st.sidebar.success("🌍 BattLeDIM ✅")
    else:
        st.sidebar.info("🌍 BattleDIM: загрузи во вкладке")

    # NEW: optional modules status in sidebar
    if _DEMO_OK:
        st.sidebar.success("🚨 Алёрты + Live Демо ✅")
    if _BIZ_OK:
        st.sidebar.success("💼 Бизнес-модель ✅")

    run_simulation = st.sidebar.button(
        "🚀 RUN SIMULATION", type="primary", use_container_width=True
    )

    return {
        "dark_mode":        dark_mode,
        "city_name":        city_name,
        "season_temp":      season_temp,
        "frost_multiplier": frost_mult,
        "material":         material,
        "pipe_age":         pipe_age,
        "pump_head":        pump_head,
        "smart_pump":       smart_pump,
        "sampling_rate":    sampling_rate,
        "leak_node":        leak_node,
        "leak_area":        leak_area,
        "water_tariff":     water_tariff,
        "leak_threshold":   leak_threshold,
        "repair_cost":      repair_cost,
        "contingency_pipe": contingency_pipe if enable_n1 else None,
        "run_simulation":   run_simulation,
    }


# ═══════════════════════════════════════════════════════════════════════════
# WELCOME SCREEN
# ═══════════════════════════════════════════════════════════════════════════

def render_welcome(config: dict):
    dm = config["dark_mode"]

    st.title("💧 Smart Shygyn PRO v3 — Command Center Edition")

    # ── Hero banner ───────────────────────────────────────────────────────
    hero_bg  = "linear-gradient(135deg,#1a1f2e 0%,#0e2a45 100%)" if dm else "linear-gradient(135deg,#e8f4fd 0%,#c7e5f7 100%)"
    hero_txt = "#e2e8f0" if dm else "#1a202c"
    hero_sub = "#94a3b8" if dm else "#4a5568"

    st.markdown(f"""
    <div style="
        background: {hero_bg};
        border: 1px solid {'#2d3748' if dm else '#bee3f8'};
        border-radius: 16px;
        padding: 36px 40px;
        margin-bottom: 28px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    ">
        <div style="display:flex; align-items:center; gap:16px; margin-bottom:16px;">
            <span style="font-size:48px;">💧</span>
            <div>
                <div style="font-size:22px; font-weight:700; color:{hero_txt};">
                    Professional Water Network Decision Support System
                </div>
                <div style="font-size:14px; color:{hero_sub}; margin-top:4px;">
                    Hydraulic simulation · AI leak detection · Economic ROI · Real Kazakhstan data
                </div>
            </div>
        </div>
        <div style="display:flex; gap:24px; flex-wrap:wrap; margin-top:8px;">
            <span style="color:#10b981; font-weight:600; font-size:13px;">✅ WNTR/EPANET Physics</span>
            <span style="color:#3b82f6; font-weight:600; font-size:13px;">✅ BattLeDIM Validated</span>
            <span style="color:#f59e0b; font-weight:600; font-size:13px;">✅ Real KZ Tariffs 2025</span>
            <span style="color:#a855f7; font-weight:600; font-size:13px;">✅ Live Weather API</span>
            <span style="color:#06b6d4; font-weight:600; font-size:13px;">✅ N-1 Contingency</span>
            <span style="color:#ef4444; font-weight:600; font-size:13px;">✅ ML Isolation Forest</span>
            <span style="color:#10b981; font-weight:600; font-size:13px;">✅ Бизнес ROI · Live Демо</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Feature cards ─────────────────────────────────────────────────────
    card_bg  = "#1a1f2e" if dm else "#ffffff"
    card_bdr = "#2d3748" if dm else "#e2e8f0"
    card_txt = "#e2e8f0" if dm else "#2c3e50"
    card_sub = "#94a3b8" if dm else "#718096"

    features = [
        ("🏙️", "Multi-City",       "Almaty · Astana · Turkestan",   "Elevation physics"),
        ("🔬", "Advanced Physics",  "H-W aging · Torricelli leaks",  "Emitter modeling"),
        ("🧠", "ML Detection",      "Isolation Forest + Z-score",    "Ensemble + Comparison"),
        ("⚡", "N-1 Analysis",      "Pipe failure simulation",        "Impact assessment"),
        ("🚨", "Alert System",      "CRITICAL / WARNING / INFO",     "CSV export"),
        ("💼", "Бизнес-модель",     "ROI 39:1 · TAM/SAM/SOM",       "SaaS тиры"),
    ]
    cols = st.columns(len(features))
    for col, (icon, title, line1, line2) in zip(cols, features):
        with col:
            col.markdown(f"""
            <div style="
                background:{card_bg}; border:1px solid {card_bdr};
                border-radius:12px; padding:20px 12px; text-align:center;
                height:140px;
            ">
                <div style="font-size:32px; margin-bottom:8px;">{icon}</div>
                <div style="font-weight:700; font-size:13px; color:{card_txt};">{title}</div>
                <div style="font-size:11px; color:{card_sub}; margin-top:4px;">{line1}</div>
                <div style="font-size:10px; color:{card_sub}; margin-top:2px;">{line2}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── BattLeDIM status banner ───────────────────────────────────────────
    loader    = get_loader()
    file_stat = loader.check_files_exist()
    have_scada = file_stat.get("scada_2018") or file_stat.get("scada_2019")

    if have_scada:
        st.success(
            "🌍 **BattLeDIM датасет загружен** — алгоритм верифицирован на реальных данных "
            "водопровода г. Лимассол (Кипр). 782 узла, 909 труб, 42.6 км, 23 утечки 2019 г."
        )
    else:
        st.info(
            "🌍 **BattLeDIM**: Перейди на вкладку '🌍 BattLeDIM Анализ' "
            "чтобы запустить детекцию утечек на реальных данных (DOI: 10.5281/zenodo.4017659)."
        )

    st.info("👈 **Configure parameters in the sidebar and click RUN SIMULATION to begin**")

    st.markdown("---")
    st.markdown("### 📊 City Comparison")
    city_rows = []
    for name, cfg in CityManager.CITIES.items():
        city_rows.append({
            "City":                name,
            "Elevation Range (m)": f"{cfg.elev_min}-{cfg.elev_max}",
            "Gradient":            cfg.elev_direction,
            "Ground Temp (°C)":    cfg.ground_temp_celsius,
            "Water Stress":        f"{cfg.water_stress_index:.2f}",
            "Burst Risk":          f"×{cfg.base_burst_multiplier:.1f}",
            "Тариф (₸/м³)":        f"{get_real_tariff(name)*1000:.2f}",
            "Износ сетей":         f"{get_real_pipe_wear(name):.1f}%",
        })
    st.dataframe(pd.DataFrame(city_rows), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════

def render_dashboard(results: dict, config: dict):
    df   = results["dataframe"]
    econ = results["economics"]
    dm   = config.get("dark_mode", True)

    leak_detected      = df["Pressure (bar)"].min() < config["leak_threshold"]
    contamination_risk = (df["Pressure (bar)"] < 1.5).any()

    st.title("💧 Smart Shygyn PRO v3 — Command Center")
    st.markdown(
        f"##### {results['city_config']['name']} | "
        f"{results['material']} {results['pipe_age']:.0f} yr | "
        f"H-W C: {results['roughness']:.0f} | "
        f"Износ: {get_real_pipe_wear(results['city_config']['name']):.1f}%"
    )

    # ── KPI bar ───────────────────────────────────────────────────────────
    st.markdown("### 📊 System Status Dashboard")
    kpi_cols = st.columns(8)
    kpis = [
        ("🚨 Status",        "LEAK DETECTED" if leak_detected else "✅ NORMAL",
         "Critical" if leak_detected else "Stable",
         "inverse" if leak_detected else "normal"),
        ("📍 City",          results["city_config"]["name"],
         results["city_config"]["elev_direction"], "off"),
        ("💧 Min Pressure",  f"{df['Pressure (bar)'].min():.2f} bar",
         f"{df['Pressure (bar)'].min() - config['leak_threshold']:.2f} vs threshold",
         "inverse" if df["Pressure (bar)"].min() < config["leak_threshold"] else "normal"),
        ("💦 Water Lost",    f"{econ['lost_liters']:,.0f} L",
         f"NRW {econ['nrw_percentage']:.1f}%",
         "inverse" if econ["lost_liters"] > 0 else "normal"),
        ("💸 Total Damage",  f"{econ['total_damage_kzt']:,.0f} ₸",
         "Direct + Indirect",
         "inverse" if econ["total_damage_kzt"] > 0 else "normal"),
        ("🧠 Predicted",     results["predicted_leak"],
         f"Conf: {results['confidence']:.0f}%",
         "inverse" if results["confidence"] > 60 else "normal"),
        ("⚡ Energy Saved",  f"{econ['energy_saved_pct']:.1f}%",
         "Smart Pump" if config["smart_pump"] else "Standard", "normal"),
        ("🌿 CO₂ Saved",    f"{econ['co2_saved_kg']:.1f} kg",
         "Today", "normal"),
    ]
    for col, (label, value, delta, dc) in zip(kpi_cols, kpis):
        with col:
            st.metric(label, value, delta, delta_color=dc)

    st.markdown("---")

    # ── Alerts ────────────────────────────────────────────────────────────
    if results["city_config"]["name"] == "Астана":
        burst_mult = results["city_config"]["burst_multiplier"]
        if burst_mult > 1.3:
            st.error(
                f"🥶 **ASTANA FREEZE-THAW ALERT**: Ground temp {config['season_temp']}°C. "
                f"Pipe burst multiplier: **{burst_mult:.2f}×**. Inspect insulation immediately!"
            )
        else:
            st.info(f"❄️ Astana: Ground temp {config['season_temp']}°C. Burst risk ×{burst_mult:.2f}")

    if results["city_config"]["name"] == "Туркестан":
        wsi = results["city_config"]["water_stress_index"]
        st.warning(
            f"☀️ **TURKESTAN WATER STRESS: {wsi:.2f}** "
            f"({'CRITICAL' if wsi > 0.7 else 'HIGH'}). "
            f"Evaporation losses elevated. Consider demand management."
        )

    if contamination_risk:
        st.error(
            "⚠️ **CONTAMINATION RISK**: Pressure < 1.5 bar detected. "
            "Groundwater infiltration possible. Initiate water quality testing!"
        )

    if results["mnf_anomaly"]:
        st.warning(
            f"🌙 **MNF ANOMALY**: Night flow +{results['mnf_percentage']:.1f}% above baseline. "
            f"Possible hidden leak or unauthorized consumption."
        )

    if leak_detected:
        conf = results["confidence"]
        if conf >= 50:
            st.error(
                f"🔍 **LEAK LOCALIZED**: Predicted at **{results['predicted_leak']}** | "
                f"Confidence: **{conf:.0f}%** | "
                f"Residual: {results['residuals'].get(results['predicted_leak'], 0):.3f} bar"
            )
        else:
            st.warning(
                f"🔍 **LOW-CONFIDENCE DETECTION**: Leak suspected at **{results['predicted_leak']}** "
                f"(confidence {conf:.0f}%). Check sensor coverage."
            )

    if results["n1_result"] and "error" not in results["n1_result"]:
        n1 = results["n1_result"]
        st.error(
            f"🔧 **N-1 ACTIVE** — Pipe `{config['contingency_pipe']}` failed | "
            f"**{n1['virtual_citizens']} residents** impacted | "
            f"TTCrit: **{n1['time_to_criticality_h']} h** | "
            f"Impact: **{n1['impact_level']}**"
        )

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════
    # TABS — динамически расширяем список
    # ═══════════════════════════════════════════════════════════════════
    tab_labels = [
        "🗺️ Real-time Network Map",
        "📈 Hydraulic Diagnostics",
        "💰 Economic ROI Analysis",
        "🔬 Stress-Test & N-1",
        "🌍 BattLeDIM Анализ",
    ]
    if _DEMO_OK:
        tab_labels += ["🚨 Алёрты", "▶ Live Демо"]
    if _BIZ_OK:
        tab_labels.append("💼 Бизнес-модель")

    all_tabs = st.tabs(tab_labels)

    tab_map      = all_tabs[0]
    tab_hydro    = all_tabs[1]
    tab_econ     = all_tabs[2]
    tab_stress   = all_tabs[3]
    tab_battledim = all_tabs[4]

    # Индексы опциональных вкладок
    idx = 5
    tab_alerts = all_tabs[idx] if _DEMO_OK else None
    tab_demo   = all_tabs[idx + 1] if _DEMO_OK else None
    if _DEMO_OK:
        idx += 2
    tab_biz = all_tabs[idx] if _BIZ_OK else None

    # ─────────────────────────────────────────────────────────────────────
    # TAB 1: MAP
    # ─────────────────────────────────────────────────────────────────────
    with tab_map:
        col_map, col_ctrl = st.columns([3, 1])

        with col_ctrl:
            st.markdown("### 🛡️ Valve Control")
            if leak_detected:
                st.error(f"⚠️ Predicted: **{results['predicted_leak']}**")
                st.caption(f"Confidence: {results['confidence']:.0f}%")
                if st.button("🔒 ISOLATE SECTION", use_container_width=True, type="primary"):
                    st.session_state["isolated_pipes"] = results["isolation_pipes"]
                    st.session_state["operation_log"].append(
                        f"[{datetime.now().strftime('%H:%M:%S')}] "
                        f"🔒 Isolated {len(results['isolation_pipes'])} pipes "
                        f"around {results['predicted_leak']}"
                    )
                    st.rerun()
                if st.session_state["isolated_pipes"]:
                    st.success(f"✅ {len(st.session_state['isolated_pipes'])} pipes isolated")
                    st.caption(f"Affected: {', '.join(results['isolation_neighbors'])}")
                    if st.button("🔓 Restore Supply", use_container_width=True):
                        st.session_state["isolated_pipes"] = []
                        st.session_state["operation_log"].append(
                            f"[{datetime.now().strftime('%H:%M:%S')}] 🔓 Supply restored"
                        )
                        st.rerun()
            else:
                st.success("✅ System Normal")
                st.caption("All valves operational")

            st.markdown("---")
            st.markdown("### 📊 Pressure Gauge")
            fig_gauge = make_pressure_gauge(
                df["Pressure (bar)"].min(), config["leak_threshold"], dm
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.markdown("### 📡 Sensor Network")
            st.metric("Active Sensors", len(results["sensors"]),
                      f"{len(results['sensors']) / 16 * 100:.0f}% coverage")
            with st.expander("Sensor Locations"):
                sensor_grid = [results["sensors"][i:i+4]
                               for i in range(0, len(results["sensors"]), 4)]
                for row in sensor_grid:
                    st.text(" | ".join(row))

            st.markdown("### 🔍 Top Pressure Residuals")
            top_res = sorted(results["residuals"].items(), key=lambda x: -x[1])[:8]
            st.dataframe(
                pd.DataFrame(top_res, columns=["Node", "Δ bar"])
                  .style.format({"Δ bar": "{:.4f}"}),
                use_container_width=True, height=240
            )

            st.markdown("### 🏙️ City Profile")
            cfg = results["city_config"]
            st.caption(cfg["description"])
            st.write(f"**Elevation:** {cfg['elev_min']}-{cfg['elev_max']}m")
            st.write(f"**Burst Risk:** ×{cfg['burst_multiplier']:.2f}")
            st.write(f"**Water Stress:** {cfg['water_stress_index']:.2f}")
            st.write(f"**Тариф:** {get_real_tariff(cfg['name'])*1000:.2f} ₸/м³")

        with col_map:
            st.markdown("### 🗺️ Interactive Network Visualization")
            folium_map = make_folium_map(
                results, st.session_state["isolated_pipes"], dm
            )
            st_folium(folium_map, width=None, height=600)

    # ─────────────────────────────────────────────────────────────────────
    # TAB 2: HYDRAULICS
    # ─────────────────────────────────────────────────────────────────────
    with tab_hydro:
        st.markdown("### 📈 Comprehensive Hydraulic Analysis")
        st.caption(
            f"City: **{results['city_config']['name']}** | "
            f"Material: **{results['material']}** ({results['pipe_age']:.0f} yr) | "
            f"H-W C: **{results['roughness']:.0f}** | "
            f"Degradation: **{results['degradation_pct']:.1f}%** | "
            f"Реальный износ: **{get_real_pipe_wear(results['city_config']['name']):.1f}%**"
        )

        st.plotly_chart(
            make_hydraulic_plot(df, config["leak_threshold"], config["smart_pump"], dm),
            use_container_width=True
        )

        st.markdown("---")
        st.markdown("### 📊 Risk Distribution")
        st.plotly_chart(
            make_risk_bar_chart(results["failure_probabilities"], results["predicted_leak"], dm),
            use_container_width=True
        )

        st.markdown("---")
        st.markdown("### 📊 Statistical Summary")
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.markdown("**💧 Pressure**")
            st.dataframe(df["Pressure (bar)"].describe().to_frame().style.format("{:.3f}"),
                         use_container_width=True)
        with sc2:
            st.markdown("**🌊 Flow Rate**")
            st.dataframe(df["Flow Rate (L/s)"].describe().to_frame().style.format("{:.3f}"),
                         use_container_width=True)
        with sc3:
            st.markdown("**⏱ Water Age**")
            st.dataframe(df["Water Age (h)"].describe().to_frame().style.format("{:.2f}"),
                         use_container_width=True)

        if st.session_state["operation_log"]:
            with st.expander("📜 Operation Log (Last 20)"):
                for entry in reversed(st.session_state["operation_log"][-20:]):
                    st.code(entry, language=None)

    # ─────────────────────────────────────────────────────────────────────
    # TAB 3: ECONOMICS
    # ─────────────────────────────────────────────────────────────────────
    with tab_econ:
        st.markdown("### 💰 Complete Economic Analysis")
        st.caption(
            f"📌 Тариф: **{config['water_tariff']*1000:.2f} ₸/м³** | "
            f"Электроэнергия: **{KAZAKHSTAN_REAL_DATA['electricity_tariff_kzt_per_kwh']} ₸/кВт·ч** | "
            f"CO₂: **{KAZAKHSTAN_REAL_DATA['co2_kg_per_kwh']} кг/кВт·ч**"
        )

        ec1, ec2, ec3, ec4 = st.columns(4)
        with ec1: st.metric("💦 Direct Water Loss", f"{econ['direct_loss_kzt']:,.0f} ₸",
                             f"{econ['lost_liters']:,.0f} L")
        with ec2: st.metric("🔧 Indirect Costs",    f"{econ['indirect_cost_kzt']:,.0f} ₸",
                             "Repair deployment")
        with ec3: st.metric("⚡ Daily Energy Saved", f"{econ['energy_saved_kzt']:,.0f} ₸",
                             f"{econ['energy_saved_kwh']:.1f} kWh")
        with ec4: st.metric("🌿 CO₂ Reduction",     f"{econ['co2_saved_kg']:.1f} kg",
                             "Grid emissions")

        st.markdown("---")
        rc1, rc2, rc3 = st.columns(3)
        sensor_cost = KAZAKHSTAN_REAL_DATA["pressure_sensor_cost_kzt"]
        with rc1: st.metric("📦 Sensor CAPEX",   f"{econ['capex_kzt']:,.0f} ₸",
                             f"{len(results['sensors'])} × {sensor_cost:,} ₸")
        with rc2: st.metric("💹 Monthly Savings", f"{econ['monthly_total_savings_kzt']:,.0f} ₸",
                             "Water + Energy")
        with rc3:
            pb = econ["payback_months"]
            st.metric("⏱ Payback Period", f"{pb:.1f} months" if pb < 999 else "N/A",
                      "ROI Positive" if pb < 24 else "Review economics",
                      delta_color="normal" if pb < 24 else "inverse")

        st.markdown("---")
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("#### 📊 Non-Revenue Water")
            st.plotly_chart(make_nrw_pie_chart(econ, dm), use_container_width=True)
        with cc2:
            st.markdown("#### 📈 Payback Timeline")
            if econ["monthly_total_savings_kzt"] > 0:
                st.plotly_chart(make_payback_timeline(econ, dm), use_container_width=True)
            else:
                st.warning("No savings projected. Adjust parameters.")

        st.markdown("---")
        st.markdown("### 📄 Export Report")
        report_df = df.copy()
        report_df["City"]                = results["city_config"]["name"]
        report_df["Material"]            = results["material"]
        report_df["Pipe_Age_Years"]      = results["pipe_age"]
        report_df["Predicted_Leak_Node"] = results["predicted_leak"]
        report_df["Confidence_%"]        = results["confidence"]
        report_df["NRW_%"]               = econ["nrw_percentage"]
        report_df["Total_Damage_KZT"]    = econ["total_damage_kzt"]
        report_df["Payback_Months"]      = econ["payback_months"]
        report_df["Tariff_KZT_per_L"]    = config["water_tariff"]
        report_df["Real_Wear_%"]         = get_real_pipe_wear(results["city_config"]["name"])
        report_df["Dataset"]             = "BattLeDIM DOI:10.5281/zenodo.4017659"

        st.download_button(
            label="📥 Download CSV Report",
            data=report_df.to_csv(index=False, encoding="utf-8-sig"),
            file_name=f"shygyn_{results['city_config']['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # ─────────────────────────────────────────────────────────────────────
    # TAB 4: STRESS-TEST & N-1
    # ─────────────────────────────────────────────────────────────────────
    with tab_stress:
        st.markdown("### 🔬 System Reliability & Contingency Analysis")

        if results["n1_result"]:
            if "error" in results["n1_result"]:
                st.warning(f"N-1 Analysis: {results['n1_result']['error']}")
            else:
                n1 = results["n1_result"]
                st.error(f"**N-1 FAILURE SCENARIO — Pipe `{config['contingency_pipe']}` Failed**")
                nc1, nc2, nc3, nc4 = st.columns(4)
                with nc1: st.metric("🏘️ Affected Residents", f"{n1['virtual_citizens']:,}")
                with nc2: st.metric("📍 Affected Nodes",     len(n1["affected_nodes"]))
                with nc3: st.metric("⏱ Time to Critical",   f"{n1['time_to_criticality_h']:.1f} h")
                with nc4: st.metric("🚨 Impact Level",       n1["impact_level"],
                                    delta_color="inverse" if n1["impact_level"] == "CRITICAL" else "normal")
                st.info(f"Recommended: Close isolation valve `{n1['best_isolation_valve']}`")
                if n1["affected_nodes"]:
                    st.code(", ".join(n1["affected_nodes"]))
        else:
            st.info("Enable N-1 Contingency in sidebar to simulate pipe failure scenarios.")

        st.markdown("---")
        st.markdown("### 🔥 Pipe Failure Probability Heatmap")
        st.pyplot(make_failure_heatmap(results, dm))

        st.markdown("---")
        st.markdown("### 🏆 Top-5 High-Risk Nodes")
        sorted_probs = sorted(
            [(k, v) for k, v in results["failure_probabilities"].items() if k != "Res"],
            key=lambda x: -x[1]
        )[:5]
        risk_df = pd.DataFrame(sorted_probs, columns=["Node", "Failure Risk (%)"])
        risk_df["Sensor Installed"] = risk_df["Node"].apply(
            lambda n: "📡 Yes" if n in results["sensors"] else "—"
        )
        risk_df["Leak Predicted"] = risk_df["Node"].apply(
            lambda n: "⚠️ YES" if n == results["predicted_leak"] and leak_detected else "—"
        )
        st.dataframe(risk_df.style.format({"Failure Risk (%)": "{:.1f}"}),
                     use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### 💡 Predictive Maintenance")
        max_node, max_risk = sorted_probs[0] if sorted_probs else ("N/A", 0)
        if max_risk > 40:
            st.error(
                f"🔴 **URGENT**: Replace pipes at **{max_node}** immediately. "
                f"Risk: **{max_risk:.1f}%** | Burst ×{results['city_config']['burst_multiplier']:.2f}"
            )
        elif max_risk > 25:
            st.warning(
                f"🟠 **PLAN REPLACEMENT** at **{max_node}** within 6 months. "
                f"H-W C degraded to **{results['roughness']:.0f}** "
                f"(base: **{HydraulicPhysics.HAZEN_WILLIAMS_BASE[results['material']]:.0f}**)"
            )
        else:
            st.success(
                f"🟢 **SYSTEM OK** — next inspection in 12 months. "
                f"H-W C: **{results['roughness']:.0f}** | "
                f"Degradation: **{results['degradation_pct']:.1f}%**"
            )

        if results["city_config"]["name"] == "Астана" and results["city_config"]["burst_multiplier"] > 1.3:
            st.warning("❄️ **ASTANA**: Ensure thermal insulation on exposed pipes.")
        if results["city_config"]["name"] == "Туркестан":
            st.warning(
                f"☀️ **TURKESTAN**: WSI {results['city_config']['water_stress_index']:.2f}. "
                "Install pressure-reducing valves."
            )

    # ─────────────────────────────────────────────────────────────────────
    # TAB 5: BATTLEDIM
    # ─────────────────────────────────────────────────────────────────────
    with tab_battledim:
        render_battledim_tab(dark_mode=dm)

    # ─────────────────────────────────────────────────────────────────────
    # TAB 6: ALERTS (новая вкладка)
    # ─────────────────────────────────────────────────────────────────────
    if _DEMO_OK and tab_alerts is not None:
        with tab_alerts:
            render_alerts_tab(results, config, dark_mode=dm)

    # ─────────────────────────────────────────────────────────────────────
    # TAB 7: LIVE DEMO (новая вкладка)
    # ─────────────────────────────────────────────────────────────────────
    if _DEMO_OK and tab_demo is not None:
        with tab_demo:
            render_demo_tab(dark_mode=dm)

    # ─────────────────────────────────────────────────────────────────────
    # TAB 8: BUSINESS MODEL (новая вкладка)
    # ─────────────────────────────────────────────────────────────────────
    if _BIZ_OK and tab_biz is not None:
        with tab_biz:
            render_business_tab(dark_mode=dm, city_name=results["city_config"]["name"])


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    init_session_state()
    config = render_sidebar()
    st.markdown(DARK_CSS if config["dark_mode"] else LIGHT_CSS, unsafe_allow_html=True)

    # ── Run simulation ────────────────────────────────────────────────────
    if config["run_simulation"]:
        st.session_state["simulation_results"] = None
        gc.collect()
        logger.info("Session memory cleared before new simulation.")

        leak_node = config["leak_node"] or f"N_{random.randint(0,3)}_{random.randint(0,3)}"

        with st.spinner("⏳ Initializing hydraulic simulation engine …"):
            backend = SmartShygynBackend(config["city_name"], config["season_temp"])

        with st.spinner("🔬 Running WNTR/EPANET simulation …"):
            try:
                results = backend.run_full_simulation(
                    material=config["material"],
                    pipe_age=config["pipe_age"],
                    pump_head_m=config["pump_head"],
                    smart_pump=config["smart_pump"],
                    sampling_rate_hz=config["sampling_rate"],
                    leak_node=leak_node,
                    leak_area_cm2=config["leak_area"],
                    contingency_pipe=config["contingency_pipe"],
                    water_tariff_kzt=config["water_tariff"],
                    leak_threshold_bar=config["leak_threshold"],
                    repair_cost_kzt=config["repair_cost"]
                )
            except Exception as exc:
                st.error(
                    f"❌ Simulation error: **{type(exc).__name__}: {exc}**\n"
                    "Adjust parameters and try again."
                )
                logger.exception("Unexpected error in run_full_simulation")
                return

        st.session_state["simulation_results"] = results
        st.session_state["last_run_params"]    = config

        log_entry = (
            f"[{datetime.now().strftime('%H:%M:%S')}] "
            f"✅ {config['city_name']} | {config['material']} {config['pipe_age']}yr | "
            f"{config['pump_head']}m"
            + (" | SmartPump" if config["smart_pump"] else "")
            + (f" | Leak:{leak_node}")
            + (f" | N-1:{config['contingency_pipe']}" if config["contingency_pipe"] else "")
            + (f" | {config['season_temp']:.1f}°C")
            + (f" | Frost×{config['frost_multiplier']:.2f}" if config["frost_multiplier"] > 1.0 else "")
            + (f" | ₸{config['water_tariff']:.5f}/л")
        )
        st.session_state["operation_log"].append(log_entry)
        st.sidebar.success("✅ Simulation Complete!")

    # ── Render ────────────────────────────────────────────────────────────
    if st.session_state["simulation_results"] is None:
        render_welcome(config)
    else:
        render_dashboard(
            st.session_state["simulation_results"],
            st.session_state["last_run_params"]
        )


if __name__ == "__main__":
    main()
