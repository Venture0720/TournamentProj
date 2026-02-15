"""
Smart Shygyn PRO v3 — FRONTEND APPLICATION
Complete Streamlit interface integrating all backend components.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from datetime import datetime
import random
import gc

from backend import SmartShygynBackend, CityManager, HydraulicPhysics
from weather import get_city_weather, get_frost_multiplier, format_weather_display


st.set_page_config(
    page_title="Smart Shygyn PRO v3 — Command Center",
    layout="wide",
    page_icon="💧",
    initial_sidebar_state="expanded",
)


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
[data-testid="stAppViewContainer"] {
  background-color: var(--bg);
  color: var(--text);
}
[data-testid="stSidebar"] {
  background-color: var(--card);
  border-right: 2px solid var(--border);
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
  color: var(--text);
}
[data-testid="stHeader"] {
  background-color: var(--bg);
  border-bottom: 1px solid var(--border);
}
[data-testid="stMetricValue"] {
  font-size: 24px;
  font-weight: 700;
  color: var(--text);
}
[data-testid="stMetricLabel"] {
  font-size: 12px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
h1 {
  color: var(--accent) !important;
  text-align: center;
  padding: 16px 0;
  letter-spacing: 1px;
  border-bottom: 3px solid var(--accent);
  margin-bottom: 24px;
}
h2 {
  color: var(--text) !important;
  border-left: 4px solid var(--accent);
  padding-left: 12px;
  margin-top: 24px;
}
h3 {
  color: var(--text) !important;
  border-bottom: 2px solid var(--accent);
  padding-bottom: 8px;
  margin-top: 16px;
}
h4, h5, h6 {
  color: var(--text) !important;
}
.stAlert {
  border-radius: 8px;
  border-left-width: 4px;
  background-color: var(--card);
  color: var(--text);
}
.stTabs [data-baseweb="tab-list"] {
  gap: 8px;
  background-color: var(--bg);
}
.stTabs [data-baseweb="tab"] {
  font-size: 14px;
  font-weight: 600;
  padding: 12px 24px;
  border-radius: 8px 8px 0 0;
  background-color: var(--card);
  color: var(--text);
}
.stTabs [data-baseweb="tab"]:hover {
  background-color: var(--border);
}
.stTabs [aria-selected="true"] {
  background-color: var(--accent) !important;
  color: white !important;
}
.stButton > button {
  width: 100%;
  font-weight: 600;
  border-radius: 6px;
  background-color: var(--accent);
  color: white;
  border: none;
}
.stButton > button:hover {
  background-color: #2563eb;
  border: none;
}
.stButton > button[kind="primary"] {
  background-color: var(--ok);
}
.stButton > button[kind="primary"]:hover {
  background-color: #059669;
}
.streamlit-expanderHeader {
  font-weight: 600;
  font-size: 15px;
  color: var(--text);
  background-color: var(--card);
  border-radius: 6px;
}
[data-testid="stDataFrame"] {
  background-color: var(--card);
}
.stCodeBlock {
  background-color: var(--card) !important;
  color: var(--text) !important;
}
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div,
.stSlider > div > div > div {
  background-color: var(--card);
  color: var(--text);
  border-color: var(--border);
}
.stCaption {
  color: var(--muted) !important;
}
[data-testid="stMarkdownContainer"] p {
  color: var(--text);
}
hr {
  border-color: var(--border);
}
</style>
"""

LIGHT_CSS = """
<style>
:root {
  --bg-light: #ffffff;
  --card-light: #f8f9fa;
  --border-light: #e2e8f0;
  --accent-light: #1f77b4;
  --text-light: #2c3e50;
  --muted-light: #6c757d;
}
[data-testid="stAppViewContainer"] {
  background-color: var(--bg-light);
  color: var(--text-light);
}
[data-testid="stSidebar"] {
  background-color: var(--card-light);
  border-right: 2px solid var(--border-light);
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
  color: var(--text-light);
}
[data-testid="stHeader"] {
  background-color: var(--bg-light);
  border-bottom: 1px solid var(--border-light);
}
[data-testid="stMetricValue"] {
  font-size: 24px;
  font-weight: 700;
  color: var(--text-light);
}
[data-testid="stMetricLabel"] {
  font-size: 12px;
  color: var(--muted-light);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
h1 {
  color: var(--accent-light) !important;
  text-align: center;
  padding: 16px 0;
  border-bottom: 3px solid var(--accent-light);
  margin-bottom: 24px;
}
h2 {
  color: var(--text-light) !important;
  border-left: 4px solid #3498db;
  padding-left: 12px;
  margin-top: 24px;
}
h3 {
  color: var(--text-light) !important;
  border-bottom: 2px solid #3498db;
  padding-bottom: 8px;
  margin-top: 16px;
}
h4, h5, h6 {
  color: var(--text-light) !important;
}
.stAlert {
  border-radius: 8px;
  border-left-width: 4px;
}
.stTabs [data-baseweb="tab-list"] {
  gap: 8px;
}
.stTabs [data-baseweb="tab"] {
  font-size: 14px;
  font-weight: 600;
  padding: 12px 24px;
  border-radius: 8px 8px 0 0;
}
.stButton > button {
  width: 100%;
  font-weight: 600;
  border-radius: 6px;
}
.streamlit-expanderHeader {
  font-weight: 600;
  font-size: 15px;
}
.stCaption {
  color: var(--muted-light) !important;
}
[data-testid="stMarkdownContainer"] p {
  color: var(--text-light);
}
hr {
  border-color: var(--border-light);
}
</style>
"""


def init_session_state():
    defaults = {
        "simulation_results": None,
        "operation_log": [],
        "isolated_pipes": [],
        "city_name": "Алматы",
        "last_run_params": {},
        "dark_mode": True,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def make_hydraulic_plot(df, threshold_bar, smart_pump, dark_mode):
    bg = "#0e1117" if dark_mode else "white"
    fg = "#e2e8f0" if dark_mode else "#2c3e50"
    grid_c = "#2d3748" if dark_mode else "#d0d0d0"
    
    rows = 4 if smart_pump else 3
    row_heights = [0.28, 0.28, 0.22, 0.22] if smart_pump else [0.35, 0.35, 0.30]
    
    titles = ["💧 Pressure at Leak Node (bar)", "🌊 Main Pipe Flow Rate (L/s)", "⏱ Water Age at Leak Node (hours)"]
    if smart_pump:
        titles.append("⚡ Dynamic Pump Head Schedule (m)")
    
    fig = make_subplots(rows=rows, cols=1, subplot_titles=titles, vertical_spacing=0.08, row_heights=row_heights)
    
    fig.add_trace(go.Scatter(x=df["Hour"], y=df["Pressure (bar)"], name="Pressure (Smoothed)", line=dict(color="#3b82f6", width=2.5), fill="tozeroy", fillcolor="rgba(59, 130, 246, 0.12)", hovertemplate="<b>Hour %{x:.1f}</b><br>Pressure: %{y:.2f} bar<extra></extra>"), row=1, col=1)
    fig.add_hline(y=threshold_bar, line_dash="dash", line_color="#ef4444", line_width=2.5, annotation_text="⚠ Leak Threshold", annotation_position="right", row=1, col=1)
    fig.add_hrect(y0=0, y1=1.5, fillcolor="rgba(239, 68, 68, 0.08)", layer="below", line_width=0, row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df["Hour"], y=df["Flow Rate (L/s)"], name="Observed Flow", line=dict(color="#f59e0b", width=2.5), hovertemplate="<b>Hour %{x:.1f}</b><br>Flow: %{y:.2f} L/s<extra></extra>"), row=2, col=1)
    expected_flow = df["Demand Pattern"] * df["Flow Rate (L/s)"].mean()
    fig.add_trace(go.Scatter(x=df["Hour"], y=expected_flow, name="Expected Flow", line=dict(color="#10b981", width=2, dash="dot"), hovertemplate="<b>Hour %{x:.1f}</b><br>Expected: %{y:.2f} L/s<extra></extra>"), row=2, col=1)
    fig.add_vrect(x0=2, x1=5, fillcolor="rgba(59, 130, 246, 0.08)", layer="below", line_width=0, annotation_text="MNF Window", annotation_position="top left", annotation_font_size=10, row=2, col=1)
    
    fig.add_trace(go.Scatter(x=df["Hour"], y=df["Water Age (h)"], name="Water Age", line=dict(color="#a855f7", width=2.5), fill="tozeroy", fillcolor="rgba(168, 85, 247, 0.12)", hovertemplate="<b>Hour %{x:.1f}</b><br>Age: %{y:.1f} hours<extra></extra>"), row=3, col=1)
    
    if smart_pump:
        fig.add_trace(go.Scatter(x=df["Hour"], y=df["Pump Head (m)"], name="Pump Head", line=dict(color="#10b981", width=2.5), fill="tozeroy", fillcolor="rgba(16, 185, 129, 0.12)", hovertemplate="<b>Hour %{x:.1f}</b><br>Head: %{y:.0f} m<extra></extra>"), row=4, col=1)
        fig.add_vrect(x0=0, x1=6, fillcolor="rgba(16, 185, 129, 0.08)", layer="below", line_width=0, annotation_text="Night Mode (70%)", annotation_position="top left", annotation_font_size=10, row=4, col=1)
        fig.add_vrect(x0=23, x1=24, fillcolor="rgba(16, 185, 129, 0.08)", layer="below", line_width=0, row=4, col=1)
    
    for r in range(1, rows + 1):
        fig.update_xaxes(gridcolor=grid_c, color=fg, showgrid=True, row=r, col=1)
        fig.update_yaxes(gridcolor=grid_c, color=fg, showgrid=True, row=r, col=1)
    
    fig.update_yaxes(title_text="Pressure (bar)", row=1, col=1)
    fig.update_yaxes(title_text="Flow Rate (L/s)", row=2, col=1)
    fig.update_yaxes(title_text="Water Age (h)", row=3, col=1)
    
    if smart_pump:
        fig.update_yaxes(title_text="Pump Head (m)", row=4, col=1)
        fig.update_xaxes(title_text="Hour of Day", row=4, col=1)
    else:
        fig.update_xaxes(title_text="Hour of Day", row=3, col=1)
    
    fig.update_layout(height=950 if smart_pump else 750, showlegend=True, hovermode="x unified", plot_bgcolor=bg, paper_bgcolor=bg, font=dict(color=fg, size=12), margin=dict(l=60, r=40, t=70, b=50), legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=grid_c, borderwidth=1, orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    
    return fig


def make_folium_map(results, isolated_pipes, dark_mode):
    city_cfg = results["city_config"]
    wn = results["network"]
    predicted_leak = results["predicted_leak"]
    failure_probs = results["failure_probabilities"]
    residuals = results["residuals"]
    sensors = results["sensors"]
    
    tiles = "CartoDB dark_matter" if dark_mode else "OpenStreetMap"
    m = folium.Map(location=[city_cfg["lat"], city_cfg["lng"]], zoom_start=city_cfg["zoom"], tiles=tiles)
    
    city_manager = CityManager(city_cfg["name"])
    node_coords = {}
    
    for link_name in wn.link_name_list:
        link = wn.get_link(link_name)
        
        if not (hasattr(link, "start_node_name") and hasattr(link, "end_node_name")):
            continue
        
        start_node = link.start_node_name
        end_node = link.end_node_name
        
        def get_coords(node_name):
            if node_name == "Res":
                return city_cfg["lat"] - 0.0009, city_cfg["lng"] - 0.0009
            node = wn.get_node(node_name)
            x, y = node.coordinates
            i, j = int(round(x / 100)), int(round(y / 100))
            return city_manager.grid_to_latlon(i, j)
        
        start_coords = get_coords(start_node)
        end_coords = get_coords(end_node)
        
        node_coords[start_node] = start_coords
        node_coords[end_node] = end_coords
        
        is_isolated = link_name in isolated_pipes
        
        folium.PolyLine([start_coords, end_coords], color="#c0392b" if is_isolated else "#4a5568", weight=6 if is_isolated else 3, opacity=0.9 if is_isolated else 0.6, tooltip=f"{'⛔ ISOLATED: ' if is_isolated else ''}{link_name}").add_to(m)
    
    leak_detected = results["dataframe"]["Pressure (bar)"].min() < 2.7
    
    for node_name in wn.node_name_list:
        coords = node_coords.get(node_name)
        if coords is None:
            continue
        
        prob = failure_probs.get(node_name, 0)
        residual = residuals.get(node_name, 0)
        is_sensor = node_name in sensors
        
        if node_name == "Res":
            color, icon = "blue", "tint"
            popup_text = "<b>Reservoir</b><br>Water Source"
        elif node_name == predicted_leak and leak_detected:
            color, icon = "red", "warning-sign"
            popup_text = f"<b>⚠️ PREDICTED LEAK</b><br>Node: {node_name}<br>Failure Risk: {prob:.1f}%<br>Pressure Drop: {residual:.3f} bar<br>Confidence: {results['confidence']:.0f}%"
        elif prob > 40:
            color, icon = "red", "remove"
            popup_text = f"<b>{node_name}</b><br>Failure Risk: {prob:.1f}%<br>Status: CRITICAL"
        elif prob > 25:
            color, icon = "orange", "exclamation-sign"
            popup_text = f"<b>{node_name}</b><br>Failure Risk: {prob:.1f}%<br>Status: ELEVATED"
        elif prob > 15:
            color, icon = "beige", "info-sign"
            popup_text = f"<b>{node_name}</b><br>Failure Risk: {prob:.1f}%<br>Status: MODERATE"
        else:
            color, icon = "green", "ok"
            popup_text = f"<b>{node_name}</b><br>Failure Risk: {prob:.1f}%<br>Status: NORMAL"
        
        if is_sensor:
            folium.CircleMarker(coords, radius=15, color="#f59e0b", weight=3, fill=False, tooltip=f"📡 Sensor Node: {node_name}").add_to(m)
        
        folium.Marker(coords, popup=folium.Popup(popup_text, max_width=250), tooltip=node_name, icon=folium.Icon(color=color, icon=icon, prefix="glyphicon")).add_to(m)
    
    legend_html = f"""
    <div style="position: fixed; bottom: 30px; left: 30px; width: 260px; z-index: 9999; background: {'rgba(14,17,23,0.95)' if dark_mode else 'rgba(255,255,255,0.95)'}; padding: 16px; border-radius: 10px; border: 2px solid {'#4a5568' if dark_mode else '#cbd5e0'}; font-size: 12px; color: {'#e2e8f0' if dark_mode else '#2d3748'}; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
        <b style="font-size: 14px; color: #3b82f6;">🗺️ Network Legend</b>
        <hr style="margin: 8px 0; border-color: {'#4a5568' if dark_mode else '#cbd5e0'};">
        <div style="margin: 6px 0;">🔴 <b>High Risk</b> (&gt;40%)</div>
        <div style="margin: 6px 0;">🟠 <b>Elevated</b> (25-40%)</div>
        <div style="margin: 6px 0;">🟡 <b>Moderate</b> (15-25%)</div>
        <div style="margin: 6px 0;">🟢 <b>Normal</b> (&lt;15%)</div>
        <div style="margin: 6px 0;">⚠️ <b>Predicted Leak</b></div>
        <div style="margin: 6px 0;">🔵 <b>Reservoir</b></div>
        <hr style="margin: 8px 0; border-color: {'#4a5568' if dark_mode else '#cbd5e0'};">
        <div style="margin: 6px 0;">🟡 <b>Ring</b> = Sensor Node ({len(sensors)}/16)</div>
        <div style="margin: 6px 0;">⛔ <b>Red Pipe</b> = Isolated</div>
    </div>
    """
    
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


def make_failure_heatmap(results, dark_mode):
    wn = results["network"]
    failure_probs = results["failure_probabilities"]
    sensors = results["sensors"]
    predicted_leak = results["predicted_leak"]
    
    fig, ax = plt.subplots(figsize=(12, 10), facecolor="#0e1117" if dark_mode else "white")
    ax.set_facecolor("#0e1117" if dark_mode else "white")
    txt_color = "white" if dark_mode else "black"
    
    pos = {node: wn.get_node(node).coordinates for node in wn.node_name_list}
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
            ring = plt.Circle((x, y), radius=28, color="#f59e0b", fill=False, linewidth=2.5, linestyle="--", zorder=4)
            ax.add_patch(ring)
        
        if node == predicted_leak:
            alert_ring = plt.Circle((x, y), radius=36, color="#ef4444", fill=False, linewidth=3, linestyle="-", zorder=5)
            ax.add_patch(alert_ring)
        
        ax.text(x, y, node, fontsize=8, fontweight="bold", ha="center", va="center", color=txt_color, zorder=6)
    
    legend_elements = [
        mpatches.Patch(color="#ef4444", label="High Risk (>40%)"),
        mpatches.Patch(color="#f59e0b", label="Elevated (25-40%)"),
        mpatches.Patch(color="#eab308", label="Moderate (15-25%)"),
        mpatches.Patch(color="#10b981", label="Normal (<15%)"),
        mpatches.Patch(color="#3b82f6", label="Reservoir"),
        mpatches.Patch(color="none", label="─────────"),
        mpatches.Patch(color="#f59e0b", label="📡 Sensor (dashed ring)"),
        mpatches.Patch(color="#ef4444", label="⚠️ Predicted Leak (solid ring)"),
    ]
    
    ax.legend(handles=legend_elements, loc="upper left", fontsize=10, facecolor="#1a1f2e" if dark_mode else "white", edgecolor="#4a5568" if dark_mode else "#cbd5e0", labelcolor=txt_color)
    
    city_name = results["city_config"]["name"]
    material = results["material"]
    age = results["pipe_age"]
    roughness = results["roughness"]
    
    ax.set_title(f"Pipe Failure Probability Heatmap — {city_name}\nMaterial: {material} | Age: {age:.0f} years | H-W C: {roughness:.0f}", fontsize=14, fontweight="bold", color=txt_color, pad=20)
    
    ax.set_aspect("equal")
    ax.set_axis_off()
    plt.tight_layout()
    
    return fig


def make_payback_timeline(economics, dark_mode):
    bg = "#0e1117" if dark_mode else "white"
    fg = "#e2e8f0" if dark_mode else "#2c3e50"
    grid_c = "#2d3748" if dark_mode else "#d0d0d0"
    
    payback_months = economics["payback_months"]
    max_months = min(int(payback_months * 2), 60)
    months = np.arange(0, max_months + 1)
    
    monthly_savings = economics["monthly_total_savings_kzt"]
    cumulative_savings = months * monthly_savings
    capex_line = np.full_like(months, economics["capex_kzt"], dtype=float)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=months, y=cumulative_savings, name="Cumulative Savings", line=dict(color="#10b981", width=3), fill="tozeroy", fillcolor="rgba(16, 185, 129, 0.15)", hovertemplate="<b>Month %{x}</b><br>Savings: ₸%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=months, y=capex_line, name="Initial Investment (CAPEX)", line=dict(color="#f59e0b", width=2.5, dash="dash"), hovertemplate="<b>CAPEX:</b> ₸%{y:,.0f}<extra></extra>"))
    
    if payback_months < max_months:
        fig.add_vline(x=payback_months, line_dash="dot", line_color="#3b82f6", line_width=2.5, annotation_text=f"Break-Even: {payback_months:.1f} months", annotation_position="top", annotation_font_size=12, annotation_font_color="#3b82f6")
    
    fig.update_layout(title="Investment Payback Timeline", xaxis_title="Months", yaxis_title="Tenge (KZT)", height=350, hovermode="x unified", plot_bgcolor=bg, paper_bgcolor=bg, font=dict(color=fg, size=12), xaxis=dict(gridcolor=grid_c, color=fg), yaxis=dict(gridcolor=grid_c, color=fg), margin=dict(l=60, r=40, t=50, b=50), legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=grid_c, borderwidth=1))
    
    return fig


def make_nrw_pie_chart(economics, dark_mode):
    bg = "#0e1117" if dark_mode else "white"
    fg = "#e2e8f0" if dark_mode else "#2c3e50"
    
    nrw_pct = economics["nrw_percentage"]
    revenue_pct = 100 - nrw_pct
    
    fig = go.Figure(go.Pie(labels=["Revenue Water", "Non-Revenue Water (Leaks)"], values=[max(0, revenue_pct), nrw_pct], hole=0.55, marker=dict(colors=["#10b981", "#ef4444"]), textinfo="label+percent", textfont=dict(size=13, color=fg), hovertemplate="<b>%{label}</b><br>%{value:.1f}%<extra></extra>"))
    
    fig.add_annotation(text=f"<b>NRW</b><br>{nrw_pct:.1f}%", x=0.5, y=0.5, font=dict(size=18, color=fg), showarrow=False)
    fig.update_layout(title="Water Accountability Distribution", height=350, paper_bgcolor=bg, font=dict(color=fg, size=12), margin=dict(l=20, r=20, t=50, b=20), showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5))
    
    return fig


def render_sidebar():
    st.sidebar.title("💧 Smart Shygyn PRO v3")
    st.sidebar.markdown("### Command Center Configuration")
    
    dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=st.session_state.get("dark_mode", True), key="theme_toggle")
    st.session_state["dark_mode"] = dark_mode
    
    st.sidebar.markdown("---")
    
    with st.sidebar.expander("🏙️ City Selection", expanded=True):
        city_name = st.selectbox("Select City", list(CityManager.CITIES.keys()), index=list(CityManager.CITIES.keys()).index(st.session_state["city_name"]))
        st.session_state["city_name"] = city_name
        
        auto_weather = st.checkbox("🛰️ Real-time Weather", value=True, key="auto_weather_toggle", help="Fetch live temperature from Open-Meteo API")
        
        if auto_weather:
            temperature, status, error = get_city_weather(city_name)
            frost_mult = get_frost_multiplier(temperature)
            
            weather_display = format_weather_display(city_name, temperature, status, error)
            st.markdown(weather_display, unsafe_allow_html=True)
            
            if status == "fallback":
                st.caption("⚠️ Using fallback temperature. Check internet connection.")
            
            if frost_mult > 1.0:
                st.warning(f"🧊 **Frost Risk Alert**: Pipe failure probability increased by **{(frost_mult-1)*100:.0f}%** due to freezing conditions!")
            
            season_temp = temperature
            
            if st.button("🔄 Refresh Weather", use_container_width=True, key="refresh_weather_btn"):
                from weather import clear_weather_cache
                clear_weather_cache()
                st.rerun()
        else:
            st.info("📊 **Stress Testing Mode**: Manual temperature control")
            season_temp = st.slider("Temperature (°C)", min_value=-30, max_value=45, value=10, step=1, help="Manual override for scenario testing")
            
            frost_mult = get_frost_multiplier(season_temp)
            if frost_mult > 1.0:
                st.warning(f"🧊 **Frost Risk**: ×{frost_mult:.2f} multiplier active")
        
        st.markdown("---")
        
        city_info = CityManager.CITIES[city_name]
        st.caption(f"**Elevation:** {city_info.elev_min}-{city_info.elev_max}m")
        st.caption(f"**Gradient:** {city_info.elev_direction}")
        st.caption(f"**Water Stress:** {city_info.water_stress_index:.2f}")
    
    with st.sidebar.expander("⚙️ Network Parameters", expanded=True):
        material = st.selectbox("Pipe Material", ["Пластик (ПНД)", "Сталь", "Чугун"], help="Material affects Hazen-Williams roughness degradation")
        
        pipe_age = st.slider("Pipe Age (years)", min_value=0, max_value=60, value=15, step=1, help="Used for H-W roughness degradation model")
        
        roughness = HydraulicPhysics.hazen_williams_roughness(material, pipe_age, season_temp)
        base_roughness = HydraulicPhysics.HAZEN_WILLIAMS_BASE[material]
        degradation = HydraulicPhysics.degradation_percentage(material, pipe_age)
        temp_factor = HydraulicPhysics.temperature_correction_factor(season_temp)
        
        st.caption(f"**H-W Roughness C:** {roughness:.1f}")
        st.caption(f"**Base C:** {base_roughness:.0f} → Aged: {base_roughness * (1 - degradation/100):.1f}")
        st.caption(f"**Temp Correction:** ×{temp_factor:.3f} ({season_temp:.1f}°C)")
        
        sampling_rate = st.select_slider("Sensor Sampling Rate", options=[1, 2, 4], value=1, format_func=lambda x: f"{x} Hz")
    
    with st.sidebar.expander("🔧 Pump Control", expanded=True):
        pump_head = st.slider("Pump Head (m)", min_value=30, max_value=70, value=40, step=5, help="Reservoir pressure head")
        st.caption(f"≈ {pump_head * 0.098:.2f} bar")
        
        smart_pump = st.checkbox("⚡ Smart Pump Scheduling", value=False, help="Night: 70% head | Day: 100% head")
        
        if smart_pump:
            st.success(f"Night: {pump_head * 0.7:.0f}m | Day: {pump_head}m")
    
    with st.sidebar.expander("💧 Leak Configuration", expanded=True):
        leak_mode = st.radio("Leak Location", ["Random", "Specific Node"], horizontal=True)
        
        if leak_mode == "Specific Node":
            leak_node = st.text_input("Leak Node ID", value="N_2_2", placeholder="e.g., N_2_2")
        else:
            leak_node = None
        
        leak_area = st.slider("Leak Area (cm²)", min_value=0.1, max_value=2.0, value=0.8, step=0.1, help="Physical hole size (Torricelli's Law)")
    
    with st.sidebar.expander("💰 Economic Parameters", expanded=True):
        water_tariff = st.number_input("Water Tariff (₸/L)", min_value=0.1, max_value=2.0, value=0.55, step=0.05, format="%.2f")
        leak_threshold = st.slider("Leak Detection Threshold (bar)", min_value=1.0, max_value=5.0, value=2.7, step=0.1, help="Pressure below which leak is detected")
        repair_cost = st.number_input("Repair Deployment Cost (₸)", min_value=10_000, max_value=200_000, value=50_000, step=5_000, format="%d")
    
    with st.sidebar.expander("🔬 N-1 Contingency", expanded=False):
        enable_n1 = st.checkbox("Enable N-1 Simulation")
        
        contingency_pipe = None
        if enable_n1:
            contingency_pipe = st.text_input("Pipe to Fail", value="PH_2_1", placeholder="e.g., PH_2_1, PV_1_2")
            st.caption("Simulates single-pipe failure")
    
    st.sidebar.markdown("---")
    
    run_simulation = st.sidebar.button("🚀 RUN SIMULATION", type="primary", use_container_width=True)
    
    return {
        "dark_mode": dark_mode,
        "city_name": city_name,
        "season_temp": season_temp,
        "frost_multiplier": frost_mult,
        "material": material,
        "pipe_age": pipe_age,
        "pump_head": pump_head,
        "smart_pump": smart_pump,
        "sampling_rate": sampling_rate,
        "leak_node": leak_node,
        "leak_area": leak_area,
        "water_tariff": water_tariff,
        "leak_threshold": leak_threshold,
        "repair_cost": repair_cost,
        "contingency_pipe": contingency_pipe if enable_n1 else None,
        "run_simulation": run_simulation,
    }


def main():
    init_session_state()
    config = render_sidebar()
    
    css_to_apply = DARK_CSS if config["dark_mode"] else LIGHT_CSS
    st.markdown(css_to_apply, unsafe_allow_html=True)
    
    if config["run_simulation"]:
        if "simulation_results" in st.session_state and st.session_state["simulation_results"]:
            old_results = st.session_state["simulation_results"]
            
            if "network" in old_results:
                del old_results["network"]
            if "dataframe" in old_results:
                del old_results["dataframe"]
            
            del old_results
            st.session_state["simulation_results"] = None
            gc.collect()
        
        if config["leak_node"] is None:
            grid_size = 4
            i = random.randint(0, grid_size - 1)
            j = random.randint(0, grid_size - 1)
            leak_node = f"N_{i}_{j}"
        else:
            leak_node = config["leak_node"]
        
        with st.spinner("⏳ Initializing hydraulic simulation engine..."):
            backend = SmartShygynBackend(config["city_name"], config["season_temp"])
        
        with st.spinner("🔬 Running WNTR/EPANET simulation..."):
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
        
        st.session_state["simulation_results"] = results
        st.session_state["last_run_params"] = config
        
        if "error" in results:
            st.sidebar.error(f"⚠️ **SIMULATION FAILED**\n\nError: {results['error']}\n\nUsing fallback data for visualization. Adjust parameters and try again.")
            st.session_state["simulation_error"] = results["error"]
        else:
            st.session_state["simulation_error"] = None
        
        log_entry = (
            f"[{datetime.now().strftime('%H:%M:%S')}] "
            f"✅ {config['city_name']} | {config['material']} {config['pipe_age']}yr | "
            f"{config['pump_head']}m"
            + (" | SmartPump" if config['smart_pump'] else "")
            + (f" | Leak: {leak_node}" if leak_node else "")
            + (f" | N-1: {config['contingency_pipe']}" if config['contingency_pipe'] else "")
            + (f" | Temp: {config['season_temp']:.1f}°C" if config['season_temp'] else "")
            + (f" | Frost: ×{config['frost_multiplier']:.2f}" if config['frost_multiplier'] > 1.0 else "")
        )
        st.session_state["operation_log"].append(log_entry)
        
        st.sidebar.success("✅ Simulation Complete!")
    
    if st.session_state["simulation_results"] is None:
        st.title("💧 Smart Shygyn PRO v3 — Command Center Edition")
        st.markdown("### Professional Water Network Decision Support System")
        st.markdown("---")
        
        cols = st.columns(6)
        
        features = [
            ("🏙️", "Multi-City", "Almaty · Astana · Turkestan", "Elevation physics"),
            ("🔬", "Advanced Physics", "H-W aging · Torricelli leaks", "Emitter modeling"),
            ("🧠", "Smart Detection", "30% sensor coverage", "Residual Matrix EKF"),
            ("⚡", "N-1 Analysis", "Pipe failure simulation", "Impact assessment"),
            ("💰", "Full ROI", "CAPEX/OPEX/Payback", "Carbon footprint"),
            ("🖥️", "Command Center", "Dark/Light mode", "Real-time weather"),
        ]
        
        for col, (icon, title, line1, line2) in zip(cols, features):
            with col:
                st.markdown(f"### {icon} {title}")
                st.markdown(f"**{line1}**")
                st.caption(line2)
        
        st.markdown("---")
        st.info("👈 **Configure parameters in the sidebar and click RUN SIMULATION to begin**")
        
        st.markdown("### 📊 City Comparison")
        
        city_data = []
        for name, cfg in CityManager.CITIES.items():
            city_data.append({
                "City": name,
                "Elevation Range (m)": f"{cfg.elev_min}-{cfg.elev_max}",
                "Gradient": cfg.elev_direction,
                "Ground Temp (°C)": cfg.ground_temp_celsius,
                "Water Stress": f"{cfg.water_stress_index:.2f}",
                "Burst Risk": f"×{cfg.base_burst_multiplier:.1f}",
            })
        
        st.dataframe(pd.DataFrame(city_data), use_container_width=True, hide_index=True)
        
        return
    
    results = st.session_state["simulation_results"]
    config = st.session_state["last_run_params"]
    df = results["dataframe"]
    econ = results["economics"]
    
    st.title("💧 Smart Shygyn PRO v3 — Command Center")
    st.markdown(f"##### Intelligent Water Network Management | {results['city_config']['name']} | Real-time Monitoring & Analytics")
    
    leak_detected = df["Pressure (bar)"].min() < config["leak_threshold"]
    contamination_risk = (df["Pressure (bar)"] < 1.5).any()
    
    st.markdown("### 📊 System Status Dashboard")
    
    kpi_cols = st.columns(8)
    
    kpis = [
        ("🚨 Status", "LEAK" if leak_detected else "NORMAL", "Critical" if leak_detected else "Stable", "inverse" if leak_detected else "normal"),
        ("📍 City", results["city_config"]["name"], results["city_config"]["elev_direction"], "off"),
        ("💧 Pressure Min", f"{df['Pressure (bar)'].min():.2f} bar", f"{df['Pressure (bar)'].min() - config['leak_threshold']:.2f}", "inverse" if df['Pressure (bar)'].min() < config['leak_threshold'] else "normal"),
        ("💦 Water Lost", f"{econ['lost_liters']:,.0f} L", f"NRW {econ['nrw_percentage']:.1f}%", "inverse" if econ['lost_liters'] > 0 else "normal"),
        ("💸 Total Damage", f"{econ['total_damage_kzt']:,.0f} ₸", "Direct+Indirect", "inverse" if econ['total_damage_kzt'] > 0 else "normal"),
        ("🧠 Predicted Node", results["predicted_leak"], f"Conf: {results['confidence']:.0f}%", "inverse" if results['confidence'] > 60 else "normal"),
        ("⚡ Energy Saved", f"{econ['energy_saved_pct']:.1f}%", "Smart Pump" if config['smart_pump'] else "Standard", "normal"),
        ("🌿 CO₂ Saved", f"{econ['co2_saved_kg']:.1f} kg", "Today", "normal"),
    ]
    
    for col, (label, value, delta, delta_color) in zip(kpi_cols, kpis):
        with col:
            st.metric(label, value, delta, delta_color=delta_color)
    
    st.markdown("---")
    
    if results["city_config"]["name"] == "Астана":
        burst_mult = results["city_config"]["burst_multiplier"]
        if burst_mult > 1.3:
            st.error(f"🥶 **ASTANA FREEZE-THAW ALERT**: Ground temp {config['season_temp']}°C. Pipe burst multiplier: **{burst_mult:.2f}×**. Inspect insulation immediately!")
        else:
            st.info(f"❄️ Astana: Ground temp {config['season_temp']}°C. Burst risk ×{burst_mult:.2f}")
    
    if results["city_config"]["name"] == "Туркестан":
        wsi = results["city_config"]["water_stress_index"]
        st.warning(f"☀️ **TURKESTAN WATER STRESS INDEX: {wsi:.2f}** ({'CRITICAL' if wsi > 0.7 else 'HIGH'}). Evaporation losses are elevated. Consider demand management.")
    
    if contamination_risk:
        st.error("⚠️ **CONTAMINATION RISK DETECTED**: Pressure < 1.5 bar detected. Groundwater infiltration possible. Initiate water quality testing!")
    
    if results["mnf_anomaly"]:
        st.warning(f"🌙 **MNF ANOMALY DETECTED**: Night flow +{results['mnf_percentage']:.1f}% above baseline. Possible hidden leak or unauthorized consumption.")
    
    if leak_detected:
        if results["confidence"] >= 50:
            st.error(f"🔍 **LEAK LOCALIZED**: Predicted at **{results['predicted_leak']}** | Confidence: **{results['confidence']:.0f}%** | Residual: {results['residuals'].get(results['predicted_leak'], 0):.3f} bar drop")
        else:
            st.warning(f"🔍 **LOW-CONFIDENCE DETECTION**: Leak suspected at **{results['predicted_leak']}** (confidence {results['confidence']:.0f}%). Check sensor coverage.")
    
    if results["n1_result"] and "error" not in results["n1_result"]:
        n1 = results["n1_result"]
        st.error(f"🔧 **N-1 CONTINGENCY ACTIVE** — Pipe `{config['contingency_pipe']}` failed | **{n1['virtual_citizens']} residents** impacted | Time to criticality: **{n1['time_to_criticality_h']} hours** | Impact: **{n1['impact_level']}**")
    
    st.markdown("---")
    
    tab_map, tab_hydro, tab_econ, tab_stress = st.tabs(["🗺️ Real-time Network Map", "📈 Hydraulic Diagnostics", "💰 Economic ROI Analysis", "🔬 Stress-Test & N-1"])
    
    with tab_map:
        col_map, col_control = st.columns([3, 1])
        
        with col_control:
            st.markdown("### 🛡️ Valve Control")
            
            if leak_detected:
                st.error(f"⚠️ Predicted: **{results['predicted_leak']}**")
                st.caption(f"Confidence: {results['confidence']:.0f}%")
                
                if st.button("🔒 ISOLATE SECTION", use_container_width=True, type="primary"):
                    st.session_state["isolated_pipes"] = results["isolation_pipes"]
                    st.session_state["operation_log"].append(f"[{datetime.now().strftime('%H:%M:%S')}] 🔒 Isolated {len(results['isolation_pipes'])} pipes around {results['predicted_leak']}")
                    st.rerun()
                
                if st.session_state["isolated_pipes"]:
                    st.success(f"✅ {len(st.session_state['isolated_pipes'])} pipes isolated")
                    st.caption(f"Affected neighbors: {', '.join(results['isolation_neighbors'])}")
                    
                    if st.button("🔓 Restore Supply", use_container_width=True):
                        st.session_state["isolated_pipes"] = []
                        st.session_state["operation_log"].append(f"[{datetime.now().strftime('%H:%M:%S')}] 🔓 Supply restored")
                        st.rerun()
            else:
                st.success("✅ System Normal")
                st.caption("All valves operational")
            
            st.markdown("---")
            st.markdown("### 📡 Sensor Network")
            st.metric("Active Sensors", len(results["sensors"]), f"{len(results['sensors'])/16*100:.0f}% coverage")
            
            with st.expander("Sensor Locations"):
                sensor_grid = [results["sensors"][i:i+4] for i in range(0, len(results["sensors"]), 4)]
                for row in sensor_grid:
                    st.text(" | ".join(row))
            
            st.markdown("---")
            st.markdown("### 🔍 Pressure Residuals")
            
            top_residuals = sorted(results["residuals"].items(), key=lambda x: -x[1])[:8]
            residual_df = pd.DataFrame(top_residuals, columns=["Node", "Δ Pressure (bar)"])
            st.dataframe(residual_df.style.format({"Δ Pressure (bar)": "{:.4f}"}), use_container_width=True, height=250)
            
            st.markdown("---")
            st.markdown("### 🏙️ City Profile")
            cfg = results["city_config"]
            
            st.caption(cfg["description"])
            st.write(f"**Elevation:** {cfg['elev_min']}-{cfg['elev_max']}m")
            st.write(f"**Burst Risk:** ×{cfg['burst_multiplier']:.2f}")
            st.write(f"**Water Stress:** {cfg['water_stress_index']:.2f}")
        
        with col_map:
            st.markdown("### 🗺️ Interactive Network Visualization")
            folium_map = make_folium_map(results, st.session_state["isolated_pipes"], config["dark_mode"])
            st_folium(folium_map, width=None, height=600)
    
    with tab_hydro:
        st.markdown("### 📈 Comprehensive Hydraulic Analysis")
        st.caption(f"City: **{results['city_config']['name']}** | Material: **{results['material']}** ({results['pipe_age']:.0f} years) | H-W C: **{results['roughness']:.0f}** | Degradation: **{results['degradation_pct']:.1f}%**")
        
        fig_hydro = make_hydraulic_plot(df, config["leak_threshold"], config["smart_pump"], config["dark_mode"])
        st.plotly_chart(fig_hydro, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 📊 Statistical Summary")
        
        stat_cols = st.columns(3)
        
        with stat_cols[0]:
            st.markdown("**💧 Pressure Statistics**")
            pressure_stats = df["Pressure (bar)"].describe().to_frame()
            st.dataframe(pressure_stats.style.format("{:.3f}"), use_container_width=True)
        
        with stat_cols[1]:
            st.markdown("**🌊 Flow Rate Statistics**")
            flow_stats = df["Flow Rate (L/s)"].describe().to_frame()
            st.dataframe(flow_stats.style.format("{:.3f}"), use_container_width=True)
        
        with stat_cols[2]:
            st.markdown("**⏱ Water Age Statistics**")
            age_stats = df["Water Age (h)"].describe().to_frame()
            st.dataframe(age_stats.style.format("{:.2f}"), use_container_width=True)
        
        st.markdown("---")
        
        if st.session_state["operation_log"]:
            with st.expander("📜 Operation Log (Last 20 Events)"):
                for entry in reversed(st.session_state["operation_log"][-20:]):
                    st.code(entry, language=None)
    
    with tab_econ:
        st.markdown("### 💰 Complete Economic Analysis")
        st.markdown("#### OPEX | CAPEX | ROI | Carbon Footprint")
        
        econ_cols = st.columns(4)
        
        with econ_cols[0]:
            st.metric("💦 Direct Water Loss", f"{econ['direct_loss_kzt']:,.0f} ₸", f"{econ['lost_liters']:,.0f} L lost")
        
        with econ_cols[1]:
            st.metric("🔧 Indirect Costs", f"{econ['indirect_cost_kzt']:,.0f} ₸", "Repair deployment")
        
        with econ_cols[2]:
            st.metric("⚡ Daily Energy Saved", f"{econ['energy_saved_kzt']:,.0f} ₸", f"{econ['energy_saved_kwh']:.1f} kWh")
        
        with econ_cols[3]:
            st.metric("🌿 CO₂ Reduction", f"{econ['co2_saved_kg']:.1f} kg", "Grid emissions")
        
        st.markdown("---")
        
        roi_cols = st.columns(3)
        
        with roi_cols[0]:
            st.metric("📦 Sensor CAPEX", f"{econ['capex_kzt']:,.0f} ₸", f"{len(results['sensors'])} sensors")
        
        with roi_cols[1]:
            st.metric("💹 Monthly Savings", f"{econ['monthly_total_savings_kzt']:,.0f} ₸", "Water + Energy")
        
        with roi_cols[2]:
            payback = econ["payback_months"]
            payback_label = f"{payback:.1f} months" if payback < 999 else "N/A"
            payback_status = "ROI Positive" if payback < 24 else "Review economics"
            st.metric("⏱ Payback Period", payback_label, payback_status, delta_color="normal" if payback < 24 else "inverse")
        
        st.markdown("---")
        
        chart_cols = st.columns(2)
        
        with chart_cols[0]:
            st.markdown("#### 📊 Non-Revenue Water Distribution")
            fig_nrw = make_nrw_pie_chart(econ, config["dark_mode"])
            st.plotly_chart(fig_nrw, use_container_width=True)
        
        with chart_cols[1]:
            st.markdown("#### 📈 Investment Payback Timeline")
            if econ["monthly_total_savings_kzt"] > 0:
                fig_payback = make_payback_timeline(econ, config["dark_mode"])
                st.plotly_chart(fig_payback, use_container_width=True)
            else:
                st.warning("No savings projected. Adjust parameters to achieve positive ROI.")
        
        st.markdown("---")
        st.markdown("### 📄 Export Full Report")
        
        report_df = df.copy()
        report_df["City"] = results["city_config"]["name"]
        report_df["Material"] = results["material"]
        report_df["Pipe_Age_Years"] = results["pipe_age"]
        report_df["Predicted_Leak_Node"] = results["predicted_leak"]
        report_df["Confidence_%"] = results["confidence"]
        report_df["NRW_%"] = econ["nrw_percentage"]
        report_df["Total_Damage_KZT"] = econ["total_damage_kzt"]
        report_df["Payback_Months"] = econ["payback_months"]
        
        csv_data = report_df.to_csv(index=False, encoding="utf-8-sig")
        
        st.download_button(label="📥 Download CSV Report", data=csv_data, file_name=f"shygyn_{results['city_config']['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv", use_container_width=True)
    
    with tab_stress:
        st.markdown("### 🔬 System Reliability & Contingency Analysis")
        
        if results["n1_result"]:
            if "error" in results["n1_result"]:
                st.warning(f"N-1 Analysis: {results['n1_result']['error']}")
            else:
                n1 = results["n1_result"]
                st.error(f"**N-1 FAILURE SCENARIO — Pipe `{config['contingency_pipe']}` Failed**")
                
                n1_cols = st.columns(4)
                
                with n1_cols[0]:
                    st.metric("🏘️ Affected Residents", f"{n1['virtual_citizens']:,}", "Virtual population")
                
                with n1_cols[1]:
                    st.metric("📍 Affected Nodes", len(n1['affected_nodes']), "Disconnected junctions")
                
                with n1_cols[2]:
                    st.metric("⏱ Time to Critical", f"{n1['time_to_criticality_h']:.1f} h", "Tank depletion")
                
                with n1_cols[3]:
                    st.metric("🚨 Impact Level", n1['impact_level'], delta_color="inverse" if n1['impact_level'] == "CRITICAL" else "normal")
                
                st.markdown("**Recommended Action:**")
                st.info(f"Close isolation valve: `{n1['best_isolation_valve']}`")
                
                if n1['affected_nodes']:
                    st.markdown("**Affected Nodes:**")
                    st.code(", ".join(n1['affected_nodes']))
        else:
            st.info("**N-1 Contingency Not Enabled**  \nEnable in sidebar to simulate pipe failure scenarios.")
        
        st.markdown("---")
        st.markdown("### 🔥 Pipe Failure Probability Heatmap")
        
        fig_heatmap = make_failure_heatmap(results, config["dark_mode"])
        st.pyplot(fig_heatmap)
        
        st.markdown("---")
        st.markdown("### 🏆 Top-5 High-Risk Nodes")
        
        sorted_probs = sorted([(k, v) for k, v in results["failure_probabilities"].items() if k != "Res"], key=lambda x: -x[1])[:5]
        
        risk_df = pd.DataFrame(sorted_probs, columns=["Node", "Failure Risk (%)"])
        risk_df["Sensor Installed"] = risk_df["Node"].apply(lambda n: "📡 Yes" if n in results["sensors"] else "—")
        risk_df["Leak Predicted"] = risk_df["Node"].apply(lambda n: "⚠️ YES" if n == results["predicted_leak"] and leak_detected else "—")
        
        st.dataframe(risk_df.style.format({"Failure Risk (%)": "{:.1f}"}), use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.markdown("### 💡 Predictive Maintenance Recommendations")
        
        max_risk_node, max_risk = sorted_probs[0] if sorted_probs else ("N/A", 0)
        
        if max_risk > 40:
            st.error(f"🔴 **URGENT ACTION REQUIRED**  \nReplace pipes at **{max_risk_node}** immediately.  \nFailure risk: **{max_risk:.1f}%** | Burst multiplier: **×{results['city_config']['burst_multiplier']:.2f}**")
        elif max_risk > 25:
            st.warning(f"🟠 **PLAN REPLACEMENT**  \nSchedule pipe replacement at **{max_risk_node}** within 6 months.  \nH-W Roughness degraded to **{results['roughness']:.0f}** (from base **{HydraulicPhysics.HAZEN_WILLIAMS_BASE[results['material']]:.0f}**)")
        else:
            st.success(f"🟢 **SYSTEM ACCEPTABLE**  \nNext routine inspection in 12 months.  \nCurrent H-W C: **{results['roughness']:.0f}** | Degradation: **{results['degradation_pct']:.1f}%**")
        
        if results["city_config"]["name"] == "Астана":
            if results["city_config"]["burst_multiplier"] > 1.3:
                st.warning("❄️ **ASTANA-SPECIFIC**: Ensure thermal insulation on all exposed pipes. Freeze-thaw cycles significantly increase burst risk.")
        
        if results["city_config"]["name"] == "Туркестан":
            st.warning(f"☀️ **TURKESTAN-SPECIFIC**: Water Stress Index **{results['city_config']['water_stress_index']:.2f}**. Install pressure-reducing valves to limit evaporative losses.")


if __name__ == "__main__":
    main()
