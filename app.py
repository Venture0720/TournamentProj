"""
Smart Shygyn PRO v3 — FRONTEND APPLICATION (FULLY INTEGRATED)
All modules wired into the main simulation flow:
  - hydraulic_intelligence.py  → Part 1: freeze-thaw, corrosion, remaining life
  - leak_analytics.py          → Part 2: MNF, virtual sensors, sensor optimizer
  - risk_engine.py             → Part 3: water quality, criticality index CI
  - ml_engine.py               → ML comparison Z-score / IF / Ensemble
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

# ── Core modules ─────────────────────────────────────────────────────────
from backend import SmartShygynBackend, CityManager, HydraulicPhysics
from weather import get_city_weather, get_frost_multiplier, format_weather_display
from battledim_analysis import render_battledim_tab, build_baseline
from data_loader import (
    initialize_battledim, get_real_tariff, get_real_pipe_wear,
    get_estimated_pipe_age, KAZAKHSTAN_REAL_DATA, get_loader,
)
from config import CONFIG

# ── Advanced modules (graceful fallback if missing) ───────────────────────
try:
    from hydraulic_intelligence import HydraulicIntelligenceEngine
    _HI_OK = True
except ImportError:
    _HI_OK = False

try:
    from leak_analytics import LeakAnalyticsEngine
    _LA_OK = True
except ImportError:
    _LA_OK = False

try:
    from risk_engine import (
        WaterAgeAnalyzer, CriticalityIndexCalculator, SocialImpactFactors,
    )
    _RE_OK = True
except ImportError:
    _RE_OK = False

try:
    from ml_engine import compare_methods
    _ML_OK = True
except ImportError:
    _ML_OK = False

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
    layout="wide", page_icon="💧", initial_sidebar_state="expanded",
)

DARK_CSS = """<style>
:root{--bg:#0e1117;--card:#1a1f2e;--border:#2d3748;--accent:#3b82f6;
      --danger:#ef4444;--warn:#f59e0b;--ok:#10b981;--text:#e2e8f0;--muted:#94a3b8;}
[data-testid="stAppViewContainer"]{background-color:var(--bg);color:var(--text);}
[data-testid="stSidebar"]{background-color:var(--card);border-right:2px solid var(--border);}
[data-testid="stHeader"]{background-color:var(--bg);border-bottom:1px solid var(--border);}
[data-testid="stMetricValue"]{font-size:24px;font-weight:700;color:var(--text);}
[data-testid="stMetricLabel"]{font-size:12px;color:var(--muted);text-transform:uppercase;}
h1{color:var(--accent)!important;text-align:center;padding:16px 0;
   border-bottom:3px solid var(--accent);margin-bottom:24px;}
h2{color:var(--text)!important;border-left:4px solid var(--accent);padding-left:12px;margin-top:24px;}
h3{color:var(--text)!important;border-bottom:2px solid var(--accent);padding-bottom:8px;margin-top:16px;}
h4,h5,h6{color:var(--text)!important;}
.stTabs [data-baseweb="tab-list"]{gap:8px;background-color:var(--bg);}
.stTabs [data-baseweb="tab"]{font-size:14px;font-weight:600;padding:12px 24px;
  border-radius:8px 8px 0 0;background-color:var(--card);color:var(--text);}
.stTabs [aria-selected="true"]{background-color:var(--accent)!important;color:white!important;}
.stButton>button{width:100%;font-weight:600;border-radius:6px;
  background-color:var(--accent);color:white;border:none;}
.stCaption{color:var(--muted)!important;}
hr{border-color:var(--border);}
</style>"""

LIGHT_CSS = """<style>
[data-testid="stAppViewContainer"]{background-color:#f0f4f8;color:#1a202c;}
[data-testid="stSidebar"]{background-color:#fff;border-right:2px solid #cbd5e0;}
h1{color:#2563eb!important;text-align:center;padding:16px 0;
   border-bottom:3px solid #2563eb;margin-bottom:24px;}
h2{color:#1a202c!important;border-left:4px solid #3b82f6;padding-left:12px;}
h3{color:#1a202c!important;border-bottom:2px solid #3b82f6;padding-bottom:8px;}
</style>"""


# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════

def init_session_state():
    defaults = {
        "simulation_results":    None,
        "advanced_results":      None,
        "operation_log":         [],
        "isolated_pipes":        [],
        "city_name":             "Алматы",
        "last_run_params":       {},
        "dark_mode":             True,
        "battledim_initialized": False,
        "battledim_available":   False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if not st.session_state.get("battledim_initialized"):
        ok, msg = initialize_battledim(show_progress=False)
        st.session_state["battledim_initialized"] = True
        st.session_state["battledim_available"]   = ok


# ═══════════════════════════════════════════════════════════════════════════
# ADVANCED ENGINE RUNNER  ← NEW integration point
# ═══════════════════════════════════════════════════════════════════════════

def run_advanced_engines(results: dict, config: dict) -> dict:
    """
    Runs Parts 1, 2, 3 on top of SmartShygynBackend output.
    Each module is isolated in try/except — one failure doesn't kill the rest.
    """
    adv = {}
    city    = config["city_name"]
    mat     = config["material"]
    age     = config["pipe_age"]
    temp    = config["season_temp"]
    df      = results["dataframe"]
    wn      = results["network"]

    # ── Part 1: Advanced degradation (freeze-thaw, corrosion, remaining life)
    if _HI_OK:
        try:
            hi = HydraulicIntelligenceEngine(
                city_name=city, season_temp_celsius=temp,
                material_name=mat, pipe_age_years=age,
            )
            adv["deg"] = hi.get_degradation_metrics()
        except Exception as e:
            adv["deg"] = None
            logger.warning("HydraulicIntelligenceEngine: %s", e)

    # ── Part 2: MNF, virtual sensors, sensor optimization
    if _LA_OK:
        try:
            graph = wn.get_graph()
            la    = LeakAnalyticsEngine(graph)

            n_s   = max(3, len(results["sensors"]))
            opt   = la.optimize_sensor_placement(
                n_sensors=n_s,
                node_criticality=results["failure_probabilities"],
                method="HYBRID",
            )
            cov  = la.evaluate_sensor_coverage(results["failure_probabilities"])
            mnf  = la.analyze_mnf(df)
            lc   = la.classify_leak(df)

            sensor_p = {
                n: float(results["observed_pressures"].get(n, 3.0))
                for n in results["sensors"]
                if n in results["observed_pressures"]
            }
            elevs = {
                n: float(wn.get_node(n).elevation)
                for n in wn.node_name_list if n != "Res"
            }
            vmap = la.estimate_virtual_sensors(sensor_p, elevations=elevs, method="IDW")

            adv["la"] = {
                "optimal_sensors": opt,
                "coverage":        cov,
                "mnf":             mnf,
                "leak_class":      lc,
                "virtual_map":     vmap,
            }
        except Exception as e:
            adv["la"] = None
            logger.warning("LeakAnalyticsEngine: %s", e)

    # ── Part 3: Water quality + criticality index
    if _RE_OK:
        try:
            wqa = WaterAgeAnalyzer()
            avg_age = float(df["Water Age (h)"].mean())
            age_df  = pd.DataFrame([
                {"node": n, "water_age_hours": avg_age}
                for n in wn.node_name_list if n != "Res"
            ])
            wq = wqa.analyze_water_age(age_df)

            calc   = CriticalityIndexCalculator()
            smap   = {
                n: SocialImpactFactors(
                    population_served=250,
                    has_hospital=(n == results["predicted_leak"]),
                )
                for n in wn.node_name_list if n != "Res"
            }
            fp_norm = {n: v / 100.0 for n, v in results["failure_probabilities"].items()}
            crit    = calc.calculate_network_criticality(wn.get_graph(), fp_norm, smap)
            prio    = calc.prioritize_maintenance(crit, budget_constraint=8)

            adv["re"] = {"water_quality": wq, "criticality": crit, "priorities": prio}
        except Exception as e:
            adv["re"] = None
            logger.warning("Risk engine: %s", e)

    return adv


# ═══════════════════════════════════════════════════════════════════════════
# THEME HELPER
# ═══════════════════════════════════════════════════════════════════════════

def _t(dm: bool) -> dict:
    return {"bg":  "#0e1117" if dm else "white",
            "fg":  "#e2e8f0" if dm else "#2c3e50",
            "grd": "#2d3748" if dm else "#d0d0d0"}


# ═══════════════════════════════════════════════════════════════════════════
# ORIGINAL CHART FUNCTIONS (unchanged)
# ═══════════════════════════════════════════════════════════════════════════

def make_hydraulic_plot(df, threshold_bar, smart_pump, dm):
    t    = _t(dm)
    rows = 4 if smart_pump else 3
    rh   = [0.28, 0.28, 0.22, 0.22] if smart_pump else [0.35, 0.35, 0.30]
    titles = ["💧 Pressure (bar)", "🌊 Flow Rate (L/s)", "⏱ Water Age (h)"]
    if smart_pump: titles.append("⚡ Pump Head (m)")
    fig = make_subplots(rows=rows, cols=1, subplot_titles=titles,
                        vertical_spacing=0.08, row_heights=rh)
    fig.add_trace(go.Scatter(x=df["Hour"], y=df["Pressure (bar)"], name="Pressure",
        line=dict(color="#3b82f6", width=2.5), fill="tozeroy",
        fillcolor="rgba(59,130,246,0.12)"), row=1, col=1)
    fig.add_hline(y=threshold_bar, line_dash="dash", line_color="#ef4444",
                  line_width=2.5, annotation_text="⚠ Leak Threshold",
                  annotation_position="right", row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Hour"], y=df["Flow Rate (L/s)"], name="Flow",
        line=dict(color="#f59e0b", width=2.5)), row=2, col=1)
    exp = df["Demand Pattern"] * df["Flow Rate (L/s)"].mean()
    fig.add_trace(go.Scatter(x=df["Hour"], y=exp, name="Expected",
        line=dict(color="#10b981", width=2, dash="dot")), row=2, col=1)
    fig.add_vrect(x0=2, x1=5, fillcolor="rgba(59,130,246,0.08)", layer="below",
                  line_width=0, annotation_text="MNF", row=2, col=1)
    fig.add_trace(go.Scatter(x=df["Hour"], y=df["Water Age (h)"], name="Water Age",
        line=dict(color="#a855f7", width=2.5), fill="tozeroy",
        fillcolor="rgba(168,85,247,0.12)"), row=3, col=1)
    if smart_pump:
        fig.add_trace(go.Scatter(x=df["Hour"], y=df["Pump Head (m)"], name="Pump Head",
            line=dict(color="#10b981", width=2.5), fill="tozeroy",
            fillcolor="rgba(16,185,129,0.12)"), row=4, col=1)
    for r in range(1, rows + 1):
        fig.update_xaxes(gridcolor=t["grd"], color=t["fg"], row=r, col=1)
        fig.update_yaxes(gridcolor=t["grd"], color=t["fg"], row=r, col=1)
    fig.update_yaxes(title_text="Pressure (bar)",  row=1, col=1)
    fig.update_yaxes(title_text="Flow (L/s)",      row=2, col=1)
    fig.update_yaxes(title_text="Water Age (h)",   row=3, col=1)
    if smart_pump: fig.update_yaxes(title_text="Head (m)", row=4, col=1)
    fig.update_xaxes(title_text="Hour", row=rows, col=1)
    fig.update_layout(height=950 if smart_pump else 750, showlegend=True,
        hovermode="x unified", plot_bgcolor=t["bg"], paper_bgcolor=t["bg"],
        font=dict(color=t["fg"], size=12), margin=dict(l=60, r=40, t=70, b=50),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h",
                    yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig


def make_pressure_gauge(pressure_bar, threshold, dm):
    t = _t(dm)
    color = "#10b981" if pressure_bar >= threshold else "#ef4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=round(pressure_bar, 2),
        delta={"reference": threshold, "valueformat": ".2f",
               "increasing": {"color": "#10b981"}, "decreasing": {"color": "#ef4444"}},
        number={"suffix": " bar", "font": {"size": 28, "color": t["fg"]}},
        gauge={"axis": {"range": [0, 8], "tickcolor": t["fg"], "tickfont": {"color": t["fg"]}},
               "bar": {"color": color, "thickness": 0.25},
               "bgcolor": "rgba(0,0,0,0)", "borderwidth": 2, "bordercolor": t["fg"],
               "steps": [{"range": [0, 1.5], "color": "rgba(239,68,68,0.3)"},
                          {"range": [1.5, threshold], "color": "rgba(245,158,11,0.2)"},
                          {"range": [threshold, 8], "color": "rgba(16,185,129,0.15)"}],
               "threshold": {"line": {"color": "#ef4444", "width": 4},
                             "thickness": 0.75, "value": threshold}},
        title={"text": "Min Pressure", "font": {"size": 14, "color": t["fg"]}}))
    fig.update_layout(height=220, paper_bgcolor=t["bg"],
                      font=dict(color=t["fg"]), margin=dict(l=20, r=20, t=60, b=10))
    return fig


def make_risk_bar_chart(failure_probs, predicted_leak, dm):
    t = _t(dm)
    top = sorted([(k, v) for k, v in failure_probs.items() if k != "Res"],
                 key=lambda x: x[1], reverse=True)[:10]
    names  = [n for n, _ in top]
    values = [v for _, v in top]
    colors = ["#ef4444" if (n == predicted_leak or v > 40) else
              "#f59e0b" if v > 25 else "#eab308" if v > 15 else "#10b981"
              for n, v in top]
    fig = go.Figure(go.Bar(x=values, y=names, orientation="h", marker_color=colors,
        text=[f"{v:.1f}%" for v in values], textposition="outside"))
    fig.update_layout(title="Top-10 Node Failure Risk", xaxis_title="Probability (%)",
        height=320, plot_bgcolor=t["bg"], paper_bgcolor=t["bg"],
        font=dict(color=t["fg"], size=11),
        xaxis=dict(gridcolor=t["grd"], color=t["fg"],
                   range=[0, max(values) * 1.25] if values else [0, 10]),
        yaxis=dict(color=t["fg"], autorange="reversed"),
        margin=dict(l=80, r=60, t=50, b=40), showlegend=False)
    return fig


def make_nrw_pie_chart(economics, dm):
    t = _t(dm)
    nrw = economics["nrw_percentage"]
    fig = go.Figure(go.Pie(labels=["Revenue Water", "Non-Revenue Water"],
        values=[max(0, 100 - nrw), nrw], hole=0.55,
        marker=dict(colors=["#10b981", "#ef4444"]),
        textinfo="label+percent", textfont=dict(size=13, color=t["fg"])))
    fig.add_annotation(text=f"<b>NRW</b><br>{nrw:.1f}%",
        x=0.5, y=0.5, font=dict(size=18, color=t["fg"]), showarrow=False)
    fig.update_layout(title="Water Accountability", height=350,
        paper_bgcolor=t["bg"], font=dict(color=t["fg"], size=12),
        margin=dict(l=20, r=20, t=50, b=20), showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5))
    return fig


def make_payback_timeline(economics, dm):
    t = _t(dm)
    pb = economics["payback_months"]
    max_m = min(int(pb * 2), 60)
    months = np.arange(0, max_m + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=months, y=months * economics["monthly_total_savings_kzt"],
        name="Cumulative Savings", line=dict(color="#10b981", width=3),
        fill="tozeroy", fillcolor="rgba(16,185,129,0.15)"))
    fig.add_trace(go.Scatter(x=months,
        y=np.full_like(months, economics["capex_kzt"], dtype=float),
        name="CAPEX", line=dict(color="#f59e0b", width=2.5, dash="dash")))
    if pb < max_m:
        fig.add_vline(x=pb, line_dash="dot", line_color="#3b82f6",
                      annotation_text=f"Break-Even: {pb:.1f} mo",
                      annotation_font_color="#3b82f6")
    fig.update_layout(title="Investment Payback", xaxis_title="Months",
        yaxis_title="₸", height=350, hovermode="x unified",
        plot_bgcolor=t["bg"], paper_bgcolor=t["bg"], font=dict(color=t["fg"], size=12),
        xaxis=dict(gridcolor=t["grd"], color=t["fg"]),
        yaxis=dict(gridcolor=t["grd"], color=t["fg"]),
        margin=dict(l=60, r=40, t=50, b=50))
    return fig


def make_failure_heatmap(results, dm):
    wn   = results["network"]
    fp   = results["failure_probabilities"]
    sens = results["sensors"]
    pred = results["predicted_leak"]
    fig, ax = plt.subplots(figsize=(12, 10), facecolor="#0e1117" if dm else "white")
    ax.set_facecolor("#0e1117" if dm else "white")
    tc = "white" if dm else "black"
    pos   = {n: wn.get_node(n).coordinates for n in wn.node_name_list}
    graph = wn.get_graph()
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="#4a5568", width=3.5, alpha=0.6)
    for node in wn.node_name_list:
        x, y = pos[node]
        p = fp.get(node, 0)
        c = ("#3b82f6" if node == "Res" else "#ef4444" if p > 40 else
             "#f59e0b" if p > 25 else "#eab308" if p > 15 else "#10b981")
        ax.add_patch(plt.Circle((x, y), 20, color=c, ec="white", lw=2.5, zorder=3))
        if node in sens:
            ax.add_patch(plt.Circle((x, y), 28, color="#f59e0b", fill=False, lw=2.5,
                                    linestyle="--", zorder=4))
        if node == pred:
            ax.add_patch(plt.Circle((x, y), 36, color="#ef4444", fill=False, lw=3, zorder=5))
        ax.text(x, y, node, fontsize=8, fontweight="bold",
                ha="center", va="center", color=tc, zorder=6)
    ax.legend(handles=[
        mpatches.Patch(color="#ef4444", label="High Risk (>40%)"),
        mpatches.Patch(color="#f59e0b", label="Elevated (25-40%)"),
        mpatches.Patch(color="#eab308", label="Moderate (15-25%)"),
        mpatches.Patch(color="#10b981", label="Normal (<15%)"),
        mpatches.Patch(color="#3b82f6", label="Reservoir"),
        mpatches.Patch(color="#f59e0b", label="📡 Sensor"),
        mpatches.Patch(color="#ef4444", label="⚠️ Predicted Leak"),
    ], loc="upper left", fontsize=10,
       facecolor="#1a1f2e" if dm else "white",
       edgecolor="#4a5568"  if dm else "#cbd5e0", labelcolor=tc)
    ax.set_title(
        f"Failure Probability Heatmap — {results['city_config']['name']}\n"
        f"Material: {results['material']} | Age: {results['pipe_age']:.0f}yr | "
        f"H-W C: {results['roughness']:.0f}",
        fontsize=14, fontweight="bold", color=tc, pad=20)
    ax.set_aspect("equal"); ax.set_axis_off(); plt.tight_layout()
    return fig


def make_folium_map(results, isolated_pipes, dm):
    import math
    cfg  = results["city_config"]
    wn   = results["network"]
    pred = results["predicted_leak"]
    fp   = results["failure_probabilities"]
    sens = results["sensors"]
    res  = results["residuals"]
    cm   = CityManager(cfg["name"])
    m    = folium.Map(location=[cfg["lat"], cfg["lng"]], zoom_start=cfg["zoom"],
                      tiles="CartoDB dark_matter" if dm else "OpenStreetMap")
    node_coords = {}

    def gc(node_name):
        if node_name == "Res":
            return cfg["lat"] - 0.0009, cfg["lng"] - 0.0009
        node = wn.get_node(node_name)
        x, y = node.coordinates
        return cm.grid_to_latlon(int(round(x / 100)), int(round(y / 100)))

    for lname in wn.link_name_list:
        link = wn.get_link(lname)
        if not (hasattr(link, "start_node_name") and hasattr(link, "end_node_name")):
            continue
        sc = gc(link.start_node_name); ec = gc(link.end_node_name)
        node_coords[link.start_node_name] = sc
        node_coords[link.end_node_name]   = ec
        iso = lname in isolated_pipes
        folium.PolyLine([sc, ec], color="#c0392b" if iso else "#4a5568",
            weight=6 if iso else 3, opacity=0.9 if iso else 0.6,
            tooltip=f"{'⛔ ' if iso else ''}{lname}").add_to(m)

    ld = results["dataframe"]["Pressure (bar)"].min() < 2.7
    for node in wn.node_name_list:
        coords = node_coords.get(node)
        if coords is None: continue
        prob = fp.get(node, 0)
        is_s = node in sens
        if node == "Res":
            cl, ic = "blue", "tint"
            pt = "<b>Reservoir</b>"
        elif node == pred and ld:
            cl, ic = "red", "warning-sign"
            pt = (f"<b>⚠️ PREDICTED LEAK</b><br>{node}<br>"
                  f"Risk: {prob:.1f}%<br>Residual: {res.get(node,0):.3f} bar<br>"
                  f"Conf: {results['confidence']:.0f}%")
        elif prob > 40: cl, ic, pt = "red",    "remove",           f"<b>{node}</b> {prob:.1f}% CRITICAL"
        elif prob > 25: cl, ic, pt = "orange", "exclamation-sign", f"<b>{node}</b> {prob:.1f}% ELEVATED"
        elif prob > 15: cl, ic, pt = "beige",  "info-sign",        f"<b>{node}</b> {prob:.1f}% MODERATE"
        else:           cl, ic, pt = "green",  "ok",               f"<b>{node}</b> {prob:.1f}% NORMAL"
        if is_s:
            folium.CircleMarker(coords, radius=15, color="#f59e0b", weight=3,
                fill=False, tooltip=f"📡 {node}").add_to(m)
        folium.Marker(coords, popup=folium.Popup(pt, max_width=250),
            tooltip=node,
            icon=folium.Icon(color=cl, icon=ic, prefix="glyphicon")).add_to(m)
    legend = f"""<div style="position:fixed;bottom:30px;left:30px;width:230px;z-index:9999;
        background:{'rgba(14,17,23,0.95)' if dm else 'rgba(255,255,255,0.95)'};
        padding:14px;border-radius:10px;border:2px solid {'#4a5568' if dm else '#cbd5e0'};
        font-size:12px;color:{'#e2e8f0' if dm else '#2d3748'};">
        <b style="color:#3b82f6;">🗺️ Legend</b><hr>
        🔴 High &gt;40% | 🟠 25-40% | 🟡 15-25% | 🟢 &lt;15%<hr>
        🔵 Reservoir | ⚠️ Leak | 🟡 Ring=Sensor</div>"""
    m.get_root().html.add_child(folium.Element(legend))
    return m


# ═══════════════════════════════════════════════════════════════════════════
# NEW ADVANCED PANEL RENDERERS
# ═══════════════════════════════════════════════════════════════════════════

def render_degradation_panel(adv: dict, dm: bool):
    """Part 1 — enhanced degradation physics."""
    t   = _t(dm)
    deg = adv.get("deg")
    if deg is None:
        st.caption("ℹ️ hydraulic_intelligence.py не найден — базовая деградация активна.")
        return

    st.markdown("### 🔬 Advanced Degradation Physics (Part 1)")
    d1, d2, d3, d4 = st.columns(4)
    hw_c = deg.get("hazen_williams_current", 0)
    hw_n = deg.get("hazen_williams_new", 150)
    rl   = deg.get("remaining_life_years", 0)
    ft   = deg.get("freeze_thaw_factor", 1.0)
    dp   = deg.get("degradation_percentage", 0)

    with d1: st.metric("H-W C current",    f"{hw_c:.1f}",    f"от {hw_n:.0f} new")
    with d2: st.metric("Деградация",        f"{dp:.1f}%",     delta_color="inverse")
    with d3: st.metric("Осталось лет",      f"{rl:.1f} yr",
                        delta_color="inverse" if rl < 10 else "normal")
    with d4: st.metric("Freeze-Thaw ×",     f"×{ft:.3f}",
                        "No risk" if ft <= 1.0 else "⚠️ Risk",
                        delta_color="inverse" if ft > 1.1 else "normal")

    col_a, col_b = st.columns(2)
    with col_a:
        cats   = ["H-W Degradation", "Freeze-Thaw", "Corrosion", "Thermal"]
        vals   = [min(100, dp),
                  (ft - 1.0) / 1.5 * 100,
                  (deg.get("corrosion_factor", 1.0) - 1.0) / 2.0 * 100,
                  deg.get("thermal_stress_risk", 0) * 100]
        clrs   = ["#ef4444" if v > 50 else "#f59e0b" if v > 25 else "#10b981" for v in vals]
        fig = go.Figure(go.Bar(x=cats, y=vals, marker_color=clrs,
            text=[f"{v:.1f}%" for v in vals], textposition="outside"))
        fig.update_layout(title="Environmental Stress Factors", height=260,
            yaxis=dict(range=[0, 120], gridcolor=t["grd"], color=t["fg"]),
            xaxis=dict(color=t["fg"]),
            plot_bgcolor=t["bg"], paper_bgcolor=t["bg"],
            font=dict(color=t["fg"], size=11), margin=dict(l=40, r=20, t=50, b=40))
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        rows = [(k, str(v)) for k, v in {
            "Материал":        deg.get("material_name", "—"),
            "Возраст":         f"{deg.get('pipe_age_years', 0):.0f} лет",
            "H-W new":         f"{hw_n:.0f}",
            "H-W current":     f"{hw_c:.1f}",
            "H-W min":         f"{deg.get('hazen_williams_min', 0):.0f}",
            "Темп. сезона":    f"{deg.get('season_temp_celsius', 0):.1f}°C",
            "Темп. грунта":    f"{deg.get('ground_temp_celsius', 0):.1f}°C",
            "Env. factor":     f"×{deg.get('environmental_factor', 1.0):.3f}",
        }.items()]
        st.dataframe(pd.DataFrame(rows, columns=["Параметр", "Значение"]),
                     use_container_width=True, hide_index=True)


def render_mnf_panel(adv: dict, dm: bool):
    """Part 2 — MNF classification."""
    la = adv.get("la")
    if la is None: return
    mnf = la.get("mnf", {})
    lc  = la.get("leak_class")
    if not mnf: return

    st.markdown("### 🌙 MNF Analysis — Leak Classification (Part 2)")
    m1, m2, m3, m4 = st.columns(4)
    dev = mnf.get("deviation_pct", 0)
    with m1: st.metric("MNF Actual",   f"{mnf.get('actual_mnf_lps', 0):.3f} L/s")
    with m2: st.metric("MNF Baseline", f"{mnf.get('baseline_mnf_lps', 0):.3f} L/s")
    with m3: st.metric("Deviation",    f"{dev:.1f}%",
                        delta_color="inverse" if dev > 15 else "normal")
    with m4: st.metric("MNF Leak Type", mnf.get("leak_type", "—"))

    if lc:
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Classifier Type",     lc.leak_type)
        with c2: st.metric("Severity",            f"{lc.severity_score:.1f}/100",
                            delta_color="inverse" if lc.severity_score > 30 else "normal")
        with c3: st.metric("Est. Leak Flow",      f"{lc.estimated_flow_lps:.3f} L/s")
        with c4: st.metric("Confidence",          f"{lc.confidence:.2f}")
        if lc.contributing_factors:
            st.caption("**Факторы:** " + " | ".join(lc.contributing_factors))


def render_water_quality_panel(adv: dict, dm: bool):
    """Part 3 — water quality."""
    re = adv.get("re")
    if re is None:
        st.caption("ℹ️ risk_engine.py не найден — качество воды недоступно.")
        return
    wq = re.get("water_quality")
    if wq is None: return

    st.markdown("### 💧 Water Quality & Stagnation (Part 3)")
    std   = wq.quality_standard
    emoji = {"EXCELLENT": "✅", "GOOD": "✅", "ACCEPTABLE": "⚠️",
             "MARGINAL": "🟠", "POOR": "🔴"}.get(std, "—")
    q1, q2, q3, q4 = st.columns(4)
    with q1: st.metric("Avg Water Age", f"{wq.avg_age_hours:.1f} h")
    with q2: st.metric("Max Water Age", f"{wq.max_age_hours:.1f} h",
                        delta_color="inverse" if wq.max_age_hours > 24 else "normal")
    with q3: st.metric("Quality",       f"{emoji} {std}")
    with q4: st.metric("Compliance",    f"{wq.compliance_percentage:.0f}%",
                        "<24h per node",
                        delta_color="inverse" if wq.compliance_percentage < 80 else "normal")
    cl = wq.chlorine_residual_estimate
    st.metric("🧪 Chlorine residual (est.)", f"{cl:.3f} mg/L",
              "✅ Above 0.2 mg/L" if cl >= 0.2 else "⚠️ Below KZ min 0.2 mg/L",
              delta_color="normal" if cl >= 0.2 else "inverse")
    if wq.stagnation_zones:
        with st.expander(f"🚨 {len(wq.stagnation_zones)} stagnation zones"):
            st.dataframe(pd.DataFrame(wq.stagnation_zones), use_container_width=True, hide_index=True)


def render_criticality_panel(adv: dict, dm: bool):
    """Part 3 — criticality index."""
    t  = _t(dm)
    re = adv.get("re")
    if re is None: return
    crit = re.get("criticality", {})
    prio = re.get("priorities", [])
    if not crit: return

    st.markdown("### 🎯 Criticality Index  CI = P_fail × Social Impact (Part 3)")
    st.caption("Приоритет ТО: чем выше CI — тем срочнее замена трубы.")

    top = sorted(crit.values(), key=lambda x: x.get("criticality_index", 0), reverse=True)[:10]
    names  = [n.get("node", "?") for n in top]
    ci_v   = [n.get("criticality_index", 0) for n in top]
    cls    = [n.get("risk_class", "LOW") for n in top]
    cmap   = {"CRITICAL": "#ef4444", "HIGH": "#f59e0b", "MEDIUM": "#eab308", "LOW": "#10b981"}
    fig = go.Figure(go.Bar(x=names, y=ci_v,
        marker_color=[cmap.get(c, "#94a3b8") for c in cls],
        text=[f"{v:.3f}" for v in ci_v], textposition="outside"))
    fig.update_layout(title="Top-10 Nodes by Criticality Index",
        yaxis_title="CI", height=280,
        plot_bgcolor=t["bg"], paper_bgcolor=t["bg"], font=dict(color=t["fg"], size=11),
        xaxis=dict(gridcolor=t["grd"], color=t["fg"]),
        yaxis=dict(gridcolor=t["grd"], color=t["fg"]),
        margin=dict(l=40, r=20, t=50, b=50))
    st.plotly_chart(fig, use_container_width=True)

    if prio:
        st.markdown("**Maintenance Priority List:**")
        rows = [{"Rank": i + 1, "Node": p.get("node"), "CI": f"{p.get('criticality_index',0):.3f}",
                 "P_fail": f"{p.get('failure_probability',0):.3f}",
                 "Social Impact": f"{p.get('social_impact',0):.3f}",
                 "Risk Class": p.get("risk_class"), "Priority": p.get("priority")}
                for i, p in enumerate(prio[:8])]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_virtual_sensors_panel(adv: dict, results: dict, dm: bool):
    """Part 2 — virtual sensors + sensor optimization."""
    t  = _t(dm)
    la = adv.get("la")
    if la is None:
        st.caption("ℹ️ leak_analytics.py не найден — виртуальные датчики недоступны.")
        return

    st.markdown("### 📡 Virtual Sensors & Optimal Placement (Part 2)")
    st.caption("IDW interpolation оценивает давление в узлах БЕЗ физических датчиков.")

    opt  = la.get("optimal_sensors", [])
    cov  = la.get("coverage", {})
    vmap = la.get("virtual_map", {})

    ca, cb = st.columns(2)
    with ca:
        st.markdown("**Sensor placement comparison:**")
        cur = results["sensors"]
        comp = pd.DataFrame([
            {"Метод": "Random (текущий)", "Датчиков": len(cur),
             "Avg dist to sensor": "—", "Coverage %": "—"},
            {"Метод": "HYBRID Optimal (Part 2)", "Датчиков": len(opt),
             "Avg dist to sensor": str(cov.get("avg_distance_to_sensor", "?")),
             "Coverage %": str(cov.get("coverage_percentage", "?")) + "%"},
        ])
        st.dataframe(comp, use_container_width=True, hide_index=True)
        if opt:
            st.caption(f"Рекомендованные узлы: **{', '.join(opt[:8])}**")

    with cb:
        if vmap:
            nodes  = list(vmap.keys())[:16]
            vals   = [vmap[n]["value"] for n in nodes]
            confs  = [vmap[n]["confidence"] for n in nodes]
            is_sen = [vmap[n]["is_sensor"] for n in nodes]
            clrs   = ["#3b82f6" if s else "#a855f7" for s in is_sen]
            fig = go.Figure(go.Bar(x=nodes, y=vals, marker_color=clrs,
                error_y=dict(type="data", array=[1 - c for c in confs], visible=True),
                hovertemplate="<b>%{x}</b><br>P: %{y:.3f} bar<extra></extra>"))
            fig.update_layout(title="Virtual Pressure (blue=sensor, purple=virtual)",
                height=260, plot_bgcolor=t["bg"], paper_bgcolor=t["bg"],
                font=dict(color=t["fg"], size=10),
                xaxis=dict(gridcolor=t["grd"], color=t["fg"]),
                yaxis=dict(gridcolor=t["grd"], color=t["fg"], title="Pressure (bar)"),
                margin=dict(l=50, r=20, t=50, b=40))
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

def render_sidebar():
    st.sidebar.title("💧 Smart Shygyn PRO v3")
    dark_mode = st.sidebar.toggle("🌙 Dark Mode",
        value=st.session_state.get("dark_mode", True), key="theme_toggle")
    st.session_state["dark_mode"] = dark_mode
    st.sidebar.markdown("---")

    season_temp = st.session_state.get("season_temp", 10.0)
    frost_mult  = get_frost_multiplier(season_temp)

    with st.sidebar.expander("🏙️ City Selection", expanded=True):
        city_name = st.selectbox("Select City", list(CityManager.CITIES.keys()),
            index=list(CityManager.CITIES.keys()).index(st.session_state["city_name"]))
        st.session_state["city_name"] = city_name
        auto_w = st.checkbox("🛰️ Real-time Weather", value=True)
        if auto_w:
            temperature, status, error = get_city_weather(city_name)
            frost_mult  = get_frost_multiplier(temperature)
            st.markdown(format_weather_display(city_name, temperature, status, error),
                        unsafe_allow_html=True)
            if frost_mult > 1.0:
                st.warning(f"🧊 Frost Risk: +{(frost_mult-1)*100:.0f}%!")
            season_temp = temperature
            st.session_state["season_temp"] = season_temp
            if st.button("🔄 Refresh Weather", use_container_width=True):
                from weather import clear_weather_cache
                clear_weather_cache(); st.rerun()
        else:
            season_temp = st.slider("Temperature (°C)", -30, 45, 10, 1)
            frost_mult  = get_frost_multiplier(season_temp)
            st.session_state["season_temp"] = season_temp
        ci = CityManager.CITIES[city_name]
        st.caption(f"**Elev:** {ci.elev_min}-{ci.elev_max}m | **Stress:** {ci.water_stress_index:.2f}")

    with st.sidebar.expander("⚙️ Network Parameters", expanded=True):
        material = st.selectbox("Pipe Material", ["Пластик (ПНД)", "Сталь", "Чугун"])
        real_age = int(get_estimated_pipe_age(city_name))
        pipe_age = st.slider("Pipe Age (years)", 0, 70, real_age, 1)
        st.caption(f"📌 Износ {city_name}: **{get_real_pipe_wear(city_name):.1f}%**")
        roughness = HydraulicPhysics.hazen_williams_roughness(material, pipe_age, temp=season_temp)
        st.caption(f"H-W C: **{roughness:.1f}** | "
                   f"Degrad: **{HydraulicPhysics.degradation_percentage(material, pipe_age, temp=season_temp):.1f}%**")
        sampling_rate = st.select_slider("Sensor Hz", [1, 2, 4], 1,
                                         format_func=lambda x: f"{x} Hz")

    with st.sidebar.expander("🔧 Pump Control", expanded=True):
        pump_head  = st.slider("Pump Head (m)", 30, 70, 40, 5)
        smart_pump = st.checkbox("⚡ Smart Pump Scheduling", value=False)

    with st.sidebar.expander("💧 Leak Configuration", expanded=True):
        leak_mode = st.radio("Leak Location", ["Random", "Specific Node"], horizontal=True)
        leak_node = (st.text_input("Leak Node ID", value="N_2_2")
                     if leak_mode == "Specific Node" else None)
        leak_area = st.slider("Leak Area (cm²)", 0.1, 2.0, 0.8, 0.1)

    with st.sidebar.expander("💰 Economic Parameters", expanded=True):
        real_tariff  = get_real_tariff(city_name)
        water_tariff = st.number_input("Water Tariff (₸/L)",
            min_value=0.001, max_value=2.0, value=real_tariff,
            step=0.001, format="%.5f")
        leak_threshold = st.slider("Leak Threshold (bar)", 1.0, 5.0, 2.5, 0.1)
        repair_cost    = st.number_input("Repair Cost (₸)",
            min_value=10_000, max_value=200_000, value=50_000, step=5_000, format="%d")

    with st.sidebar.expander("🔬 N-1 Contingency", expanded=False):
        enable_n1 = st.checkbox("Enable N-1 Simulation")
        contingency_pipe = st.text_input("Pipe to Fail", value="PH_2_1") if enable_n1 else None

    st.sidebar.markdown("---")

    # Status badges
    loader = get_loader()
    fs = loader.check_files_exist()
    if fs.get("scada_2018") or fs.get("scada_2019"):
        st.sidebar.success("🌍 BattLeDIM ✅")
    else:
        st.sidebar.info("🌍 BattLeDIM: загрузи во вкладке")
    badges = [lbl for ok, lbl in [(_HI_OK,"🔬P1"),(_LA_OK,"📡P2"),(_RE_OK,"🎯P3"),(_ML_OK,"🧠ML")] if ok]
    if badges: st.sidebar.success("Active: " + " ".join(badges))
    if _DEMO_OK: st.sidebar.success("🚨 Demo ✅")
    if _BIZ_OK:  st.sidebar.success("💼 Бизнес ✅")

    run = st.sidebar.button("🚀 RUN SIMULATION", type="primary", use_container_width=True)
    return {
        "dark_mode": dark_mode, "city_name": city_name,
        "season_temp": season_temp, "frost_multiplier": frost_mult,
        "material": material, "pipe_age": pipe_age,
        "pump_head": pump_head, "smart_pump": smart_pump,
        "sampling_rate": sampling_rate, "leak_node": leak_node,
        "leak_area": leak_area, "water_tariff": water_tariff,
        "leak_threshold": leak_threshold, "repair_cost": repair_cost,
        "contingency_pipe": contingency_pipe if enable_n1 else None,
        "run_simulation": run,
    }


# ═══════════════════════════════════════════════════════════════════════════
# WELCOME
# ═══════════════════════════════════════════════════════════════════════════

def render_welcome(config):
    dm = config["dark_mode"]
    st.title("💧 Smart Shygyn PRO v3 — Command Center")
    cb = "#1a1f2e" if dm else "#fff"
    cd = "#2d3748" if dm else "#e2e8f0"
    ct = "#e2e8f0" if dm else "#2c3e50"
    cs = "#94a3b8" if dm else "#718096"
    feats = [
        ("🔬","Part 1: Physics","Freeze-thaw·Corrosion","Remaining life"),
        ("📡","Part 2: Analytics","MNF·Virtual Sensors","Sensor optimizer"),
        ("🎯","Part 3: Risk","CI = P×Impact","Maintenance priority"),
        ("🧠","ML Ensemble","IF+Z-score","3-method compare"),
        ("💼","Business ROI","TAM/SAM/SOM","SaaS tiers"),
        ("🌍","BattLeDIM","Real Cyprus data","23 real leaks"),
    ]
    cols = st.columns(len(feats))
    for c, (ic, tt, l1, l2) in zip(cols, feats):
        c.markdown(f"""
        <div style="background:{cb};border:1px solid {cd};border-radius:12px;
            padding:20px 12px;text-align:center;height:140px;">
            <div style="font-size:32px;margin-bottom:8px;">{ic}</div>
            <div style="font-weight:700;font-size:13px;color:{ct};">{tt}</div>
            <div style="font-size:11px;color:{cs};margin-top:4px;">{l1}</div>
            <div style="font-size:10px;color:{cs};margin-top:2px;">{l2}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.info("👈 Configure parameters in the sidebar and click **RUN SIMULATION**")
    st.markdown("### 📊 City Comparison")
    st.dataframe(pd.DataFrame([{
        "City": n, "Elevation": f"{c.elev_min}-{c.elev_max}m",
        "Stress": f"{c.water_stress_index:.2f}",
        "Тариф ₸/м³": f"{get_real_tariff(n)*1000:.2f}",
        "Износ": f"{get_real_pipe_wear(n):.1f}%",
    } for n, c in CityManager.CITIES.items()]),
    use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════

def render_dashboard(results: dict, adv: dict, config: dict):
    df   = results["dataframe"]
    econ = results["economics"]
    dm   = config.get("dark_mode", True)
    ld   = df["Pressure (bar)"].min() < config["leak_threshold"]
    cr   = (df["Pressure (bar)"] < 1.5).any()

    st.title("💧 Smart Shygyn PRO v3 — Command Center")
    st.markdown(
        f"##### {results['city_config']['name']} | "
        f"{results['material']} {results['pipe_age']:.0f}yr | "
        f"H-W C: {results['roughness']:.0f} | "
        f"Износ: {get_real_pipe_wear(results['city_config']['name']):.1f}%")

    st.markdown("### 📊 System Status")
    kc = st.columns(8)
    for col, (lbl, val, delta, dc) in zip(kc, [
        ("🚨 Status",      "LEAK" if ld else "✅ NORMAL",
         "Critical" if ld else "Stable",
         "inverse" if ld else "normal"),
        ("📍 City",         results["city_config"]["name"],
         results["city_config"]["elev_direction"], "off"),
        ("💧 Min Pressure", f"{df['Pressure (bar)'].min():.2f} bar",
         f"{df['Pressure (bar)'].min()-config['leak_threshold']:.2f} vs thr",
         "inverse" if ld else "normal"),
        ("💦 Water Lost",   f"{econ['lost_liters']:,.0f} L",
         f"NRW {econ['nrw_percentage']:.1f}%",
         "inverse" if econ["lost_liters"] > 0 else "normal"),
        ("💸 Damage",       f"{econ['total_damage_kzt']:,.0f} ₸",
         "Direct+Indirect", "inverse" if econ["total_damage_kzt"] > 0 else "normal"),
        ("🧠 Predicted",    results["predicted_leak"],
         f"Conf:{results['confidence']:.0f}%",
         "inverse" if results["confidence"] > 60 else "normal"),
        ("⚡ Energy Saved", f"{econ['energy_saved_pct']:.1f}%",
         "SmartPump" if config["smart_pump"] else "Standard", "normal"),
        ("🌿 CO₂ Saved",   f"{econ['co2_saved_kg']:.1f} kg", "Today", "normal"),
    ]):
        with col: st.metric(lbl, val, delta, delta_color=dc)

    st.markdown("---")

    # Alerts
    if results["city_config"]["name"] == "Астана" and results["city_config"]["burst_multiplier"] > 1.3:
        st.error(f"🥶 ASTANA FREEZE-THAW: ×{results['city_config']['burst_multiplier']:.2f}")
    if results["city_config"]["name"] == "Туркестан":
        st.warning(f"☀️ TURKESTAN Water Stress: {results['city_config']['water_stress_index']:.2f}")
    if cr:
        st.error("⚠️ CONTAMINATION RISK: Pressure < 1.5 bar!")
    if results["mnf_anomaly"]:
        st.warning(f"🌙 MNF ANOMALY: +{results['mnf_percentage']:.1f}% above baseline.")
    if ld:
        st.error(f"🔍 LEAK: **{results['predicted_leak']}** | "
                 f"Conf: **{results['confidence']:.0f}%**")
    if results["n1_result"] and "error" not in (results["n1_result"] or {}):
        n1 = results["n1_result"]
        st.error(f"🔧 N-1: {n1['virtual_citizens']} residents | "
                 f"TTCrit: {n1['time_to_criticality_h']}h | {n1['impact_level']}")

    # Part 2 / Part 3 inline alerts
    if adv:
        la = adv.get("la")
        if la:
            lc = la.get("leak_class")
            if lc and lc.leak_type in ("BURST", "CATASTROPHIC"):
                st.error(f"🧠 ML Classifier: **{lc.leak_type}** — "
                         f"severity {lc.severity_score:.1f}/100, "
                         f"est. {lc.estimated_flow_lps:.3f} L/s")
            elif lc and lc.leak_type == "BACKGROUND":
                st.warning(f"🧠 ML Classifier: **BACKGROUND LEAK** "
                           f"— est. {lc.estimated_flow_lps:.3f} L/s")
        re = adv.get("re")
        if re:
            wq = re.get("water_quality")
            if wq and wq.quality_standard in ("MARGINAL", "POOR"):
                st.warning(f"💧 Water Quality: **{wq.quality_standard}** — "
                           f"avg age {wq.avg_age_hours:.1f}h, Cl {wq.chlorine_residual_estimate:.3f} mg/L")

    st.markdown("---")

    # Build tabs
    tab_labels = ["🗺️ Network Map", "📈 Hydraulics + Degradation",
                  "💰 Economic ROI", "🔬 Stress Test + Criticality",
                  "🌍 BattLeDIM + ML"]
    if _DEMO_OK: tab_labels += ["🚨 Алёрты", "▶ Live Demo"]
    if _BIZ_OK:  tab_labels.append("💼 Бизнес")
    all_tabs = st.tabs(tab_labels)
    tab_map, tab_hydro, tab_econ, tab_stress, tab_bd = all_tabs[:5]
    idx = 5
    tab_alerts = all_tabs[idx]     if _DEMO_OK else None
    tab_demo   = all_tabs[idx + 1] if _DEMO_OK else None
    if _DEMO_OK: idx += 2
    tab_biz = all_tabs[idx] if _BIZ_OK else None

    # ── TAB 1: MAP ────────────────────────────────────────────────────────
    with tab_map:
        col_m, col_c = st.columns([3, 1])
        with col_c:
            st.markdown("### 🛡️ Valve Control")
            if ld:
                st.error(f"⚠️ Predicted: **{results['predicted_leak']}**")
                if st.button("🔒 ISOLATE SECTION", use_container_width=True, type="primary"):
                    st.session_state["isolated_pipes"] = results["isolation_pipes"]
                    st.session_state["operation_log"].append(
                        f"[{datetime.now().strftime('%H:%M:%S')}] 🔒 Isolated "
                        f"{len(results['isolation_pipes'])} pipes")
                    st.rerun()
                if st.session_state["isolated_pipes"]:
                    st.success(f"✅ {len(st.session_state['isolated_pipes'])} pipes isolated")
                    if st.button("🔓 Restore Supply", use_container_width=True):
                        st.session_state["isolated_pipes"] = []; st.rerun()
            else:
                st.success("✅ System Normal")
            st.markdown("---")
            st.markdown("### Pressure Gauge")
            st.plotly_chart(make_pressure_gauge(df["Pressure (bar)"].min(),
                config["leak_threshold"], dm), use_container_width=True)
            st.metric("Active Sensors", len(results["sensors"]),
                      f"{len(results['sensors'])/16*100:.0f}% coverage")
            la = (adv or {}).get("la")
            if la and la.get("optimal_sensors"):
                st.caption(f"🎯 Optimal placement: **{', '.join(la['optimal_sensors'][:6])}**")
        with col_m:
            st.markdown("### 🗺️ Interactive Network Map")
            st_folium(make_folium_map(results, st.session_state["isolated_pipes"], dm),
                      width=None, height=600)

    # ── TAB 2: HYDRAULICS + DEGRADATION (Part 1+2+3) ─────────────────────
    with tab_hydro:
        st.markdown("### 📈 Hydraulic Analysis")
        st.plotly_chart(make_hydraulic_plot(df, config["leak_threshold"],
            config["smart_pump"], dm), use_container_width=True)
        st.markdown("---")
        st.plotly_chart(make_risk_bar_chart(results["failure_probabilities"],
            results["predicted_leak"], dm), use_container_width=True)
        st.markdown("---")

        # Part 1
        if adv: render_degradation_panel(adv, dm); st.markdown("---")
        # Part 2 MNF
        if adv: render_mnf_panel(adv, dm); st.markdown("---")
        # Part 3 Water quality
        if adv: render_water_quality_panel(adv, dm); st.markdown("---")

        st.markdown("### 📊 Statistics")
        s1, s2, s3 = st.columns(3)
        with s1: st.dataframe(df["Pressure (bar)"].describe().to_frame().style.format("{:.3f}"),
                               use_container_width=True)
        with s2: st.dataframe(df["Flow Rate (L/s)"].describe().to_frame().style.format("{:.3f}"),
                               use_container_width=True)
        with s3: st.dataframe(df["Water Age (h)"].describe().to_frame().style.format("{:.2f}"),
                               use_container_width=True)
        if st.session_state["operation_log"]:
            with st.expander("📜 Operation Log"):
                for e in reversed(st.session_state["operation_log"][-20:]):
                    st.code(e, language=None)

    # ── TAB 3: ECONOMICS ──────────────────────────────────────────────────
    with tab_econ:
        st.markdown("### 💰 Economic Analysis")
        e1, e2, e3, e4 = st.columns(4)
        with e1: st.metric("💦 Direct Loss",   f"{econ['direct_loss_kzt']:,.0f} ₸",
                            f"{econ['lost_liters']:,.0f} L")
        with e2: st.metric("🔧 Indirect",      f"{econ['indirect_cost_kzt']:,.0f} ₸")
        with e3: st.metric("⚡ Energy Saved",  f"{econ['energy_saved_kzt']:,.0f} ₸",
                            f"{econ['energy_saved_kwh']:.1f} kWh")
        with e4: st.metric("🌿 CO₂",           f"{econ['co2_saved_kg']:.1f} kg")
        st.markdown("---")
        r1, r2, r3 = st.columns(3)
        with r1: st.metric("📦 CAPEX",          f"{econ['capex_kzt']:,.0f} ₸")
        with r2: st.metric("💹 Monthly Savings", f"{econ['monthly_total_savings_kzt']:,.0f} ₸")
        with r3:
            pb = econ["payback_months"]
            st.metric("⏱ Payback", f"{pb:.1f} mo" if pb < 999 else "N/A",
                      delta_color="normal" if pb < 24 else "inverse")
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(make_nrw_pie_chart(econ, dm), use_container_width=True)
        with c2:
            if econ["monthly_total_savings_kzt"] > 0:
                st.plotly_chart(make_payback_timeline(econ, dm), use_container_width=True)
        st.markdown("---")
        rd = df.copy()
        rd["City"] = results["city_config"]["name"]
        rd["Predicted"] = results["predicted_leak"]
        rd["Confidence"] = results["confidence"]
        st.download_button("📥 Download CSV",
            data=rd.to_csv(index=False, encoding="utf-8-sig"),
            file_name=f"shygyn_{results['city_config']['name']}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv", use_container_width=True)

    # ── TAB 4: STRESS TEST + CRITICALITY (Part 2+3) ───────────────────────
    with tab_stress:
        st.markdown("### 🔬 Reliability & Contingency Analysis")
        if results["n1_result"]:
            if "error" in results["n1_result"]:
                st.warning(f"N-1: {results['n1_result']['error']}")
            else:
                n1 = results["n1_result"]
                st.error(f"**N-1 FAILURE — Pipe `{config['contingency_pipe']}`**")
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("Residents",     f"{n1['virtual_citizens']:,}")
                with c2: st.metric("Nodes",          len(n1["affected_nodes"]))
                with c3: st.metric("TTCrit",         f"{n1['time_to_criticality_h']:.1f} h")
                with c4: st.metric("Impact",          n1["impact_level"],
                                   delta_color="inverse" if n1["impact_level"] == "CRITICAL" else "normal")
        else:
            st.info("Enable N-1 Contingency in sidebar.")
        st.markdown("---")

        # Part 2 virtual sensors
        if adv: render_virtual_sensors_panel(adv, results, dm); st.markdown("---")
        # Part 3 criticality index
        if adv: render_criticality_panel(adv, dm); st.markdown("---")

        st.markdown("### 🔥 Failure Probability Heatmap")
        st.pyplot(make_failure_heatmap(results, dm))
        st.markdown("---")
        top5 = sorted([(k, v) for k, v in results["failure_probabilities"].items() if k != "Res"],
                      key=lambda x: -x[1])[:5]
        rd5 = pd.DataFrame(top5, columns=["Node", "Risk (%)"])
        rd5["Sensor"]    = rd5["Node"].apply(lambda n: "📡" if n in results["sensors"] else "—")
        rd5["Predicted"] = rd5["Node"].apply(lambda n: "⚠️" if n == results["predicted_leak"] else "—")
        st.dataframe(rd5.style.format({"Risk (%)": "{:.1f}"}), use_container_width=True, hide_index=True)
        st.markdown("---")
        mn, mr = top5[0] if top5 else ("N/A", 0)
        if mr > 40:
            st.error(f"🔴 **URGENT**: Replace **{mn}** — {mr:.1f}%")
        elif mr > 25:
            st.warning(f"🟠 **PLAN** replacement at **{mn}** within 6 months.")
        else:
            st.success("🟢 **SYSTEM OK** — next inspection in 12 months.")

    # ── TAB 5: BATTLEDIM + ML COMPARISON ─────────────────────────────────
    with tab_bd:
        render_battledim_tab(dark_mode=dm)

        if _ML_OK:
            st.markdown("---")
            st.markdown("## 🧠 ML Methods Comparison")
            st.caption("Z-score · Isolation Forest · Ensemble — сравнение на BattLeDIM 2019")
            loader = get_loader()
            fs = loader.check_files_exist()
            if fs.get("scada_2018") or fs.get("scada_2019"):
    raw18 = loader.load_scada_2018()
    raw19 = loader.load_scada_2019()
    leaks = loader.load_leaks_2019()

    s19 = raw19["pressures"].dropna(axis=1, how="all") if raw19 else None
    s18 = raw18["pressures"].dropna(axis=1, how="all") if raw18 else None

    # Проверяем что всё есть
    if s19 is None:
        st.warning("Нет данных SCADA 2019 — загрузи BattLeDIM.")
    elif leaks is None:
        st.warning("Нет файла утечек leaks_2019 — ML сравнение невозможно.")
    else:
        # Фолбэк для s18
        if s18 is None:
            st.info("SCADA 2018 не найден — используем первые 60 дней 2019 как baseline.")
            cut = s19.index[0] + pd.Timedelta(days=60)
            s18 = s19[s19.index < cut]
            s19 = s19[s19.index >= cut]

        # Проверяем DatetimeIndex
        if not isinstance(s19.index, pd.DatetimeIndex):
            st.error("Индекс SCADA не является DatetimeIndex — Z-score работать не будет.")
        else:
            ml1, ml2 = st.columns([1, 2])
            with ml1:
                z_thr = st.slider("Z-score порог", 1.5, 5.0, 3.0, 0.1, key="cmp_z")
                m_s   = st.slider("Мин. датчиков", 1, 5, 2, key="cmp_ms")
                cont  = st.slider("IF contamination", 0.01, 0.15, 0.05, 0.01, key="cmp_c")
                run_c = st.button("▶ Сравнить методы", use_container_width=True)
            with ml2:
                if run_c:
                    with st.spinner("Обучаем и сравниваем …"):
                        try:
                            baseline = build_baseline(s18)
                            cmp_df = compare_methods(
                                s19, leaks,
                                scada_2018_df=s18,
                                baseline=baseline,
                                z_threshold=z_thr,
                                min_sensors=m_s,
                                contamination=cont,
                            )
                            st.dataframe(cmp_df, use_container_width=True, hide_index=True)
                            best = cmp_df.dropna(subset=["F1 %"]).sort_values("F1 %", ascending=False)
                            if len(best):
                                st.success(f"🏆 Лучший: **{best.iloc[0]['Метод']}** "
                                           f"— F1 {best.iloc[0]['F1 %']:.1f}%")
                        except Exception as e:
                            st.error(f"Ошибка: {e}")
                            import traceback
                            st.code(traceback.format_exc())
                else:
                    st.info("Нажми **▶ Сравнить методы**")
else:
    st.info("Загрузи BattLeDIM для ML сравнения.")

    # ── OPTIONAL TABS ─────────────────────────────────────────────────────
    if _DEMO_OK and tab_alerts: 
        with tab_alerts: render_alerts_tab(results, config, dark_mode=dm)
    if _DEMO_OK and tab_demo:
        with tab_demo: render_demo_tab(dark_mode=dm)
    if _BIZ_OK and tab_biz:
        with tab_biz: render_business_tab(dark_mode=dm,
                                          city_name=results["city_config"]["name"])


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    init_session_state()
    config = render_sidebar()
    st.markdown(DARK_CSS if config["dark_mode"] else LIGHT_CSS, unsafe_allow_html=True)

    if config["run_simulation"]:
        st.session_state["simulation_results"] = None
        st.session_state["advanced_results"]   = None
        gc.collect()

        leak_node = (config["leak_node"]
                     or f"N_{random.randint(0, 3)}_{random.randint(0, 3)}")

        # ── Step 1: Core WNTR simulation ─────────────────────────────────
        with st.spinner("⏳ Running WNTR/EPANET simulation …"):
            try:
                backend = SmartShygynBackend(config["city_name"], config["season_temp"])
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
                    repair_cost_kzt=config["repair_cost"],
                )
            except Exception as exc:
                st.error(f"❌ Simulation error: **{type(exc).__name__}: {exc}**")
                return

        # ── Step 2: Part 1 + 2 + 3 advanced engines ──────────────────────
        adv = {}
        if any([_HI_OK, _LA_OK, _RE_OK]):
            with st.spinner("🔬 Running advanced analytics (Part 1+2+3) …"):
                try:
                    adv = run_advanced_engines(results, config)
                except Exception as exc:
                    st.warning(f"⚠️ Advanced analytics partial failure: {exc}")

        st.session_state["simulation_results"] = results
        st.session_state["advanced_results"]   = adv
        st.session_state["last_run_params"]    = config

        adv_status = ("✅" if adv else "⚠️ fallback") + (
            f" P1{'✅' if adv.get('deg') else '✗'}"
            f" P2{'✅' if adv.get('la')  else '✗'}"
            f" P3{'✅' if adv.get('re')  else '✗'}"
        ) if adv else " ⚠️ no advanced modules"

        st.session_state["operation_log"].append(
            f"[{datetime.now().strftime('%H:%M:%S')}] ✅ "
            f"{config['city_name']} | {config['material']} {config['pipe_age']}yr | "
            f"{config['pump_head']}m | Leak:{leak_node} | "
            f"{config['season_temp']:.1f}°C | {adv_status}"
        )
        st.sidebar.success("✅ Simulation Complete!")

    # ── Render ────────────────────────────────────────────────────────────
    if st.session_state["simulation_results"] is None:
        render_welcome(config)
    else:
        render_dashboard(
            st.session_state["simulation_results"],
            st.session_state["advanced_results"] or {},
            st.session_state["last_run_params"],
        )


if __name__ == "__main__":
    main()
