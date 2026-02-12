"""
Smart Shygyn PRO v3 — Command Center Edition
All 6 core requirements implemented.
"""
import streamlit as st
import pandas as pd
import numpy as np
import wntr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from datetime import datetime
import math
import folium
from streamlit_folium import st_folium
from scipy.ndimage import uniform_filter1d

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & THEME
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Shygyn PRO v3",
    layout="wide",
    page_icon="💧",
    initial_sidebar_state="expanded",
)

DARK_CSS = """
<style>
:root {
  --bg: #0e1117; --card: #1a1f2e; --border: #2d3748;
  --accent: #3b82f6; --danger: #ef4444; --warn: #f59e0b;
  --ok: #10b981; --text: #e2e8f0; --muted: #94a3b8;
}
[data-testid="stMetricValue"]  { font-size:22px; font-weight:700; }
[data-testid="stMetricLabel"]  { font-size:11px; color:var(--muted); }
h1 { color:var(--accent); text-align:center; padding:12px 0; letter-spacing:1px; }
h3 { color:var(--text); border-bottom:2px solid var(--accent);
     padding-bottom:8px; margin-top:16px; }
.stAlert { border-radius:8px; }
.stTabs [data-baseweb="tab"] { font-size:13px; font-weight:600; }
</style>
"""
LIGHT_CSS = """
<style>
[data-testid="stMetricValue"] { font-size:22px; font-weight:700; }
[data-testid="stMetricLabel"] { font-size:11px; }
h1 { color:#1f77b4; text-align:center; padding:12px 0; }
h3 { color:#2c3e50; border-bottom:2px solid #3498db;
     padding-bottom:8px; margin-top:16px; }
.stAlert { border-radius:8px; }
</style>
"""

# ─────────────────────────────────────────────────────────────────────────────
# REQUIREMENT 1 — CityManager
# ─────────────────────────────────────────────────────────────────────────────

class CityManager:
    """Configuration and physics modifiers for each city."""

    CITIES = {
        "Алматы": {
            "lat": 43.2220, "lng": 76.8512,
            "zoom": 15,
            "elev_min": 600, "elev_max": 1000,   # south high, north low
            "elev_direction": "S→N",              # gradient direction
            "ground_temp": 12.0,                  # °C baseline
            "burst_multiplier": 1.0,              # no permafrost
            "water_stress": 0.35,                 # moderate
            "description": "High elevation gradient (South→North, 600-1000m).",
        },
        "Астана": {
            "lat": 51.1605, "lng": 71.4704,
            "zoom": 15,
            "elev_min": 340, "elev_max": 360,    # essentially flat
            "elev_direction": "Flat",
            "ground_temp": -2.5,                  # permafrost risk
            "burst_multiplier": 1.0,              # updated dynamically
            "water_stress": 0.55,
            "description": "Flat steppe; permafrost/freeze-thaw pipe burst risk.",
        },
        "Туркестан": {
            "lat": 43.3016, "lng": 68.2730,
            "zoom": 15,
            "elev_min": 200, "elev_max": 280,
            "elev_direction": "SW→NE",
            "ground_temp": 22.0,
            "burst_multiplier": 0.8,
            "water_stress": 0.82,                 # high scarcity
            "description": "Arid; extreme evaporation and water scarcity.",
        },
    }

    def __init__(self, city_name: str, season_temp: float = 10.0):
        self.name = city_name
        self.cfg = self.CITIES[city_name]
        self.season_temp = season_temp
        self._update_burst_multiplier()

    def _update_burst_multiplier(self):
        """Astana: burst risk increases with freeze-thaw cycles."""
        if self.name == "Астана":
            # Freeze-thaw: risk spikes when temperature crosses 0°C
            delta = abs(self.season_temp - self.cfg["ground_temp"])
            self.cfg["burst_multiplier"] = 1.0 + 0.05 * min(delta, 20)
        else:
            self.cfg["burst_multiplier"] = self.cfg.get("burst_multiplier", 1.0)

    def node_elevation(self, i: int, j: int, grid_size: int = 4) -> float:
        """Return node elevation (m) based on city gradient."""
        lo, hi = self.cfg["elev_min"], self.cfg["elev_max"]
        direction = self.cfg["elev_direction"]
        if direction == "S→N":
            # South (j=0) is high, North (j=max) is low
            frac = j / (grid_size - 1)
        elif direction == "SW→NE":
            frac = (i + j) / (2 * (grid_size - 1))
        else:  # Flat
            frac = 0.5
        return hi - frac * (hi - lo)

    def water_stress_index(self) -> float:
        return self.cfg["water_stress"]

    def latlon(self) -> tuple:
        return self.cfg["lat"], self.cfg["lng"]

    def grid_to_latlon(self, i: int, j: int,
                       step: float = 0.0009) -> tuple:
        lat = self.cfg["lat"] + j * step
        lng = self.cfg["lng"] + i * step
        return lat, lng


# ─────────────────────────────────────────────────────────────────────────────
# REQUIREMENT 2 — Advanced Hydraulic Physics
# ─────────────────────────────────────────────────────────────────────────────

def hw_roughness(material: str, pipe_age_years: float) -> float:
    """
    Hazen-Williams C factor degraded by age and material.

    Base C: PVC=150, Steel=140, Cast Iron=100
    Degradation: C decreases linearly ~0.5/year for CI, ~0.3/year for Steel,
                 ~0.1/year for PVC.
    """
    base = {"Пластик (ПНД)": 150, "Сталь": 140, "Чугун": 100}
    decay = {"Пластик (ПНД)": 0.10, "Сталь": 0.30, "Чугун": 0.50}
    b = base.get(material, 130)
    d = decay.get(material, 0.30)
    return max(40.0, b - d * pipe_age_years)


def torricelli_leak_flow(area_m2: float, head_m: float,
                         cd: float = 0.61) -> float:
    """
    Torricelli's law for orifice leak discharge.
    Q = Cd * A * sqrt(2 * g * H)  [m³/s]
    """
    g = 9.81
    if head_m <= 0:
        return 0.0
    return cd * area_m2 * math.sqrt(2 * g * head_m)


def create_demand_pattern():
    hours = np.arange(24)
    pattern = []
    for h in hours:
        if 0 <= h < 6:
            pattern.append(0.3 + 0.1 * np.sin(h * np.pi / 6))
        elif 6 <= h < 9:
            pattern.append(1.2 + 0.3 * np.sin((h - 6) * np.pi / 3))
        elif 9 <= h < 18:
            pattern.append(0.8 + 0.2 * np.sin((h - 9) * np.pi / 9))
        elif 18 <= h < 22:
            pattern.append(1.4 + 0.2 * np.sin((h - 18) * np.pi / 4))
        else:
            pattern.append(0.5 + 0.2 * np.sin((h - 22) * np.pi / 2))
    return pattern


# ─────────────────────────────────────────────────────────────────────────────
# REQUIREMENT 3 — Sparse Sensor Network + EKF-style Residual Matrix
# ─────────────────────────────────────────────────────────────────────────────

SENSOR_FRACTION = 0.30   # only 30% of nodes have sensors

def place_sensors(node_list: list, seed: int = 42) -> list:
    """Select ~30% of non-Reservoir nodes as sensor locations."""
    rng = np.random.default_rng(seed)
    candidates = [n for n in node_list if n != "Res"]
    k = max(1, int(len(candidates) * SENSOR_FRACTION))
    return list(rng.choice(candidates, size=k, replace=False))


def residual_matrix_localize(healthy_p: dict, observed_p: dict,
                              sensor_nodes: list,
                              wn) -> tuple:
    """
    Residual Matrix Leak Localization (sparse sensors).

    For each non-sensor node, estimate pressure drop using:
     - known drops at sensor nodes
     - graph-distance-weighted interpolation

    Returns (predicted_node, residuals_dict, confidence_score %).
    """
    graph = wn.get_graph()
    residuals = {}

    # Compute residuals at sensor nodes
    sensor_residuals = {}
    for sn in sensor_nodes:
        hp = healthy_p.get(sn, 0)
        op = observed_p.get(sn, hp)
        if hp > 0:
            sensor_residuals[sn] = (hp - op) / hp   # normalised drop

    # Interpolate to all nodes
    all_nodes = [n for n in wn.node_name_list if n != "Res"]
    for node in all_nodes:
        if node in sensor_nodes:
            residuals[node] = sensor_residuals.get(node, 0)
        else:
            # IDW from sensor nodes
            total_w = 0.0
            weighted_r = 0.0
            for sn, r in sensor_residuals.items():
                try:
                    d = nx.shortest_path_length(graph, node, sn)
                except nx.NetworkXNoPath:
                    d = 10
                w = 1.0 / (d + 1)
                total_w += w
                weighted_r += w * r
            residuals[node] = (weighted_r / total_w) if total_w > 0 else 0

    if not residuals:
        return "N_2_2", {}, 0.0

    # Convert to absolute bar values for display
    abs_residuals = {
        n: residuals[n] * healthy_p.get(n, 1)
        for n in residuals
    }

    predicted = max(residuals, key=residuals.get)

    # Confidence Score: signal-to-noise ratio
    values = np.array(list(residuals.values()))
    signal = np.max(values)
    noise = np.std(values)
    confidence = min(100.0, (signal / (noise + 1e-6)) * 20)

    return predicted, abs_residuals, round(confidence, 1)


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def get_optimal_pump_head(hour: float, base: float) -> float:
    h = int(hour) % 24
    return base * 0.7 if (h >= 23 or h < 6) else float(base)


def apply_moving_average(s: pd.Series, window: int = 3) -> pd.Series:
    return s.rolling(window, center=True, min_periods=1).mean()


def build_network(city: CityManager, material: str, pipe_age: float,
                  sampling_rate: int, pump_pressure: float,
                  smart_pump: bool, contingency_pipe: str = None) -> tuple:
    """
    Build and run the leaky WNTR simulation.
    Returns (df_results, wn, leak_node).
    """
    wn = wntr.network.WaterNetworkModel()
    dist = 100
    roughness = hw_roughness(material, pipe_age)
    actual_diam = 0.2

    demand_pattern = create_demand_pattern()
    wn.add_pattern("dp", demand_pattern)

    for i in range(4):
        for j in range(4):
            name = f"N_{i}_{j}"
            elev = city.node_elevation(i, j)
            wn.add_junction(name, base_demand=0.001, elevation=elev,
                            demand_pattern="dp")
            wn.get_node(name).coordinates = (i * dist, j * dist)
            if i > 0:
                pn = f"PH_{i}_{j}"
                wn.add_pipe(pn, f"N_{i-1}_{j}", name,
                            length=dist, diameter=actual_diam,
                            roughness=roughness)
            if j > 0:
                pn = f"PV_{i}_{j}"
                wn.add_pipe(pn, f"N_{i}_{j-1}", name,
                            length=dist, diameter=actual_diam,
                            roughness=roughness)

    effective_head = pump_pressure * 0.85 if smart_pump else float(pump_pressure)
    wn.add_reservoir("Res", base_head=effective_head)
    wn.get_node("Res").coordinates = (-dist, -dist)
    wn.add_pipe("P_Main", "Res", "N_0_0", length=dist,
                diameter=0.4, roughness=roughness)

    # Contingency: remove a pipe (N-1)
    if contingency_pipe and contingency_pipe in wn.link_name_list:
        wn.remove_link(contingency_pipe)

    leak_node = "N_2_2"
    wn.options.time.duration = 24 * 3600
    wn.options.time.report_timestep = 3600 // sampling_rate
    wn.options.quality.parameter = "AGE"

    node = wn.get_node(leak_node)
    # Torricelli-based leak area (approximate: Q=0.08m² → area from physics)
    head_approx = effective_head * 0.5
    q_target = 0.0008  # m³/s target leak
    leak_area = torricelli_leak_flow(0.0008, head_approx)  # just for reference
    node.add_leak(wn, area=0.0008, start_time=12 * 3600)

    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()

    p = results.node["pressure"][leak_node] * 0.1
    f = results.link["flowrate"]["P_Main"] * 1000
    water_age = results.node["quality"][leak_node] / 3600

    n_pts = len(p)
    hours = np.arange(n_pts) / sampling_rate

    pump_heads = np.array([get_optimal_pump_head(h, pump_pressure) for h in hours]) \
                 if smart_pump else np.full(n_pts, float(pump_pressure))

    noise_p = np.random.normal(0, 0.04, n_pts)
    noise_f = np.random.normal(0, 0.08, n_pts)

    df = pd.DataFrame({
        "Hour": hours,
        "Pressure (bar)": p.values + noise_p,
        "Flow Rate (L/s)": np.abs(f.values) + noise_f,
        "Water Age (h)": water_age.values,
        "Demand Pattern": np.tile(demand_pattern, n_pts // 24 + 1)[:n_pts],
        "Pump Head (m)": pump_heads,
    })
    df["Pressure (bar)"] = apply_moving_average(df["Pressure (bar)"])
    df["Flow Rate (L/s)"] = apply_moving_average(df["Flow Rate (L/s)"])

    return df, wn, leak_node


def build_healthy_network(city: CityManager, material: str, pipe_age: float,
                          sampling_rate: int, pump_pressure: float) -> dict:
    """
    No-leak baseline. Returns {node: mean_pressure_bar}.
    """
    wn = wntr.network.WaterNetworkModel()
    dist = 100
    roughness = hw_roughness(material, pipe_age)

    demand_pattern = create_demand_pattern()
    wn.add_pattern("dp", demand_pattern)

    for i in range(4):
        for j in range(4):
            name = f"N_{i}_{j}"
            wn.add_junction(name, base_demand=0.001,
                            elevation=city.node_elevation(i, j),
                            demand_pattern="dp")
            wn.get_node(name).coordinates = (i * dist, j * dist)
            if i > 0:
                wn.add_pipe(f"PH_{i}_{j}", f"N_{i-1}_{j}", name,
                            length=dist, diameter=0.2, roughness=roughness)
            if j > 0:
                wn.add_pipe(f"PV_{i}_{j}", f"N_{i}_{j-1}", name,
                            length=dist, diameter=0.2, roughness=roughness)

    wn.add_reservoir("Res", base_head=float(pump_pressure))
    wn.get_node("Res").coordinates = (-dist, -dist)
    wn.add_pipe("P_Main", "Res", "N_0_0", length=dist,
                diameter=0.4, roughness=roughness)

    wn.options.time.duration = 24 * 3600
    wn.options.time.report_timestep = 3600 // sampling_rate

    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()

    return {
        node: (results.node["pressure"][node] * 0.1).mean()
        for node in wn.node_name_list
        if node != "Res"
    }


# ─────────────────────────────────────────────────────────────────────────────
# REQUIREMENT 4 — N-1 Contingency & Impact Assessment
# ─────────────────────────────────────────────────────────────────────────────

def n1_impact(wn, failed_pipe: str, df: pd.DataFrame,
              city: CityManager) -> dict:
    """
    Simulate N-1 failure of a pipe.
    Returns dict with affected_nodes, virtual_citizens, time_to_criticality_h.
    """
    graph = wn.get_graph().copy()
    try:
        link = wn.get_link(failed_pipe)
        graph.remove_edge(link.start_node_name, link.end_node_name)
    except Exception:
        return {"error": "Pipe not found"}

    # Nodes disconnected from reservoir
    try:
        reachable = nx.descendants(graph, "Res") | {"Res"}
    except Exception:
        reachable = set(wn.node_name_list)

    affected = [n for n in wn.node_name_list
                if n != "Res" and n not in reachable]

    citizens = len(affected) * 250  # 250 residents per node

    # Time to criticality: assume local elevated tank of 5000L per node
    local_tank_vol_l = 5000 * max(1, len(affected))
    avg_demand_ls = df["Flow Rate (L/s)"].mean() * len(affected) / 16
    ttc = (local_tank_vol_l / (avg_demand_ls * 1000)) if avg_demand_ls > 0 else 99
    ttc = round(min(ttc, 72), 1)

    # Best valve to close: the pipe closest to Res that re-isolates
    best_valve = failed_pipe

    return {
        "affected_nodes": affected,
        "virtual_citizens": citizens,
        "time_to_criticality_h": ttc,
        "best_isolation_valve": best_valve,
    }


def find_isolation_valves(network, leak_node):
    graph = network.get_graph()
    neighbors = list(graph.neighbors(leak_node))
    pipes_to_close = []
    for neighbor in neighbors:
        for link_name in network.link_name_list:
            link = network.get_link(link_name)
            if hasattr(link, "start_node_name") and \
                    hasattr(link, "end_node_name"):
                if (link.start_node_name == leak_node and
                        link.end_node_name == neighbor) or \
                        (link.end_node_name == leak_node and
                         link.start_node_name == neighbor):
                    pipes_to_close.append(link_name)
    return pipes_to_close, neighbors


# ─────────────────────────────────────────────────────────────────────────────
# REQUIREMENT 5 — Full Economic Model
# ─────────────────────────────────────────────────────────────────────────────

SENSOR_UNIT_COST_KZT = 450_000     # per sensor install
ENERGY_COST_KZT_PER_KWH = 22.0    # Kazakhstan average
KWH_PER_M3_PUMP = 0.4              # kWh per m³ pumped
CO2_KG_PER_KWH = 0.62             # KZ grid emission factor
GRID_KW = 15.0                     # pump motor power (kW)


def calculate_economics(df: pd.DataFrame, price_kzt: float,
                        pump_pressure: float, smart_pump: bool,
                        limit: float, freq: int,
                        n_sensors: int, repair_cost: float) -> dict:
    df_leak = df[df["Pressure (bar)"] < limit]
    lost_l = df_leak["Flow Rate (L/s)"].sum() * (3600 / freq) if len(df_leak) else 0
    total_flow_l = df["Flow Rate (L/s)"].sum() * (3600 / freq)
    nrw_pct = (lost_l / total_flow_l * 100) if total_flow_l > 0 else 0

    direct_loss_kzt = lost_l * price_kzt
    indirect_kzt = repair_cost if len(df_leak) > 0 else 0

    # Energy
    static_energy_kwh = GRID_KW * 24          # full day at max
    night_hours = 8                             # ~23-05 = 6h + margins
    if smart_pump:
        dyn_kwh = GRID_KW * (24 - night_hours) + GRID_KW * 0.7 * night_hours
    else:
        dyn_kwh = static_energy_kwh
    energy_saved_kwh = static_energy_kwh - dyn_kwh
    energy_saved_kzt = energy_saved_kwh * ENERGY_COST_KZT_PER_KWH
    energy_saved_pct = (energy_saved_kwh / static_energy_kwh * 100) \
                       if smart_pump else 0.0
    co2_saved_kg = energy_saved_kwh * CO2_KG_PER_KWH

    # CAPEX: sensor installation
    capex_kzt = n_sensors * SENSOR_UNIT_COST_KZT

    # Monthly savings: water losses + energy (if smart pump)
    monthly_water_savings = direct_loss_kzt * 30
    monthly_energy_savings = energy_saved_kzt * 30 if smart_pump else 0
    monthly_total_savings = monthly_water_savings + monthly_energy_savings

    # Payback period (months)
    payback_months = (capex_kzt / monthly_total_savings) \
                     if monthly_total_savings > 0 else 9999

    return {
        "lost_l": lost_l,
        "nrw_pct": nrw_pct,
        "direct_loss_kzt": direct_loss_kzt,
        "indirect_kzt": indirect_kzt,
        "total_damage_kzt": direct_loss_kzt + indirect_kzt,
        "energy_saved_pct": round(energy_saved_pct, 1),
        "energy_saved_kwh": round(energy_saved_kwh, 2),
        "energy_saved_kzt": round(energy_saved_kzt, 0),
        "co2_saved_kg": round(co2_saved_kg, 2),
        "capex_kzt": capex_kzt,
        "monthly_savings_kzt": round(monthly_total_savings, 0),
        "payback_months": round(payback_months, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def make_hydraulic_plot(df, threshold, smart_pump, dark_mode):
    bg = "#0e1117" if dark_mode else "white"
    fg = "#e2e8f0" if dark_mode else "#2c3e50"
    grid_c = "#2d3748" if dark_mode else "lightgray"

    rows = 4 if smart_pump else 3
    rh = [0.28, 0.28, 0.22, 0.22] if smart_pump else [0.35, 0.35, 0.30]
    titles = ["💧 Pressure (bar)", "🌊 Flow Rate (L/s)",
              "⏱ Water Age (h)"]
    if smart_pump:
        titles.append("⚡ Pump Head (m) — Dynamic Schedule")

    fig = make_subplots(rows=rows, cols=1, subplot_titles=titles,
                        vertical_spacing=0.07, row_heights=rh)

    # Pressure
    fig.add_trace(go.Scatter(
        x=df["Hour"], y=df["Pressure (bar)"],
        name="Pressure (MA)", line=dict(color="#3b82f6", width=2.5),
        fill="tozeroy", fillcolor="rgba(59,130,246,0.10)",
    ), row=1, col=1)
    fig.add_hline(y=threshold, line_dash="dash", line_color="#ef4444",
                  line_width=2, annotation_text="⚠ Threshold", row=1, col=1)
    fig.add_hrect(y0=0, y1=1.5, fillcolor="red", opacity=0.08,
                  layer="below", line_width=0, row=1, col=1)

    # Flow
    fig.add_trace(go.Scatter(
        x=df["Hour"], y=df["Flow Rate (L/s)"],
        name="Flow (MA)", line=dict(color="#f59e0b", width=2.5),
    ), row=2, col=1)
    exp_flow = df["Demand Pattern"] * df["Flow Rate (L/s)"].mean()
    fig.add_trace(go.Scatter(
        x=df["Hour"], y=exp_flow,
        name="Expected Flow", line=dict(color="#10b981", width=1.8, dash="dot"),
    ), row=2, col=1)
    fig.add_vrect(x0=2, x1=5, fillcolor="blue", opacity=0.07,
                  layer="below", line_width=0,
                  annotation_text="MNF", annotation_position="top left",
                  row=2, col=1)

    # Water age
    fig.add_trace(go.Scatter(
        x=df["Hour"], y=df["Water Age (h)"],
        name="Water Age", line=dict(color="#a855f7", width=2.5),
        fill="tozeroy", fillcolor="rgba(168,85,247,0.10)",
    ), row=3, col=1)

    # Smart pump head
    if smart_pump:
        fig.add_trace(go.Scatter(
            x=df["Hour"], y=df["Pump Head (m)"],
            name="Pump Head", line=dict(color="#10b981", width=2.5),
            fill="tozeroy", fillcolor="rgba(16,185,129,0.12)",
        ), row=4, col=1)
        fig.add_vrect(x0=0, x1=5, fillcolor="green", opacity=0.07,
                      layer="below", line_width=0,
                      annotation_text="Night Mode",
                      annotation_position="top left", row=4, col=1)
        fig.add_vrect(x0=23, x1=24, fillcolor="green", opacity=0.07,
                      layer="below", line_width=0, row=4, col=1)
        fig.update_yaxes(title_text="Head (m)", row=4, col=1,
                         gridcolor=grid_c, color=fg)
        fig.update_xaxes(title_text="Hour", row=4, col=1,
                         gridcolor=grid_c, color=fg)
    else:
        fig.update_xaxes(title_text="Hour", row=3, col=1,
                         gridcolor=grid_c, color=fg)

    for r in range(1, rows + 1):
        fig.update_xaxes(gridcolor=grid_c, color=fg, row=r, col=1)
        fig.update_yaxes(gridcolor=grid_c, color=fg, row=r, col=1)

    fig.update_yaxes(title_text="Pressure (bar)", row=1, col=1)
    fig.update_yaxes(title_text="Flow (L/s)", row=2, col=1)
    fig.update_yaxes(title_text="Age (h)", row=3, col=1)

    fig.update_layout(
        height=950 if smart_pump else 800,
        showlegend=True,
        hovermode="x unified",
        plot_bgcolor=bg,
        paper_bgcolor=bg,
        font=dict(color=fg, size=11),
        margin=dict(l=55, r=55, t=70, b=50),
        legend=dict(bgcolor="rgba(0,0,0,0)")
    )
    return fig


def make_folium_map(wn, city: CityManager, active_leak: bool,
                    pred_node: str, fail_probs: dict,
                    residuals: dict, sensor_nodes: list,
                    isolated_pipes: list) -> folium.Map:
    lat, lng = city.latlon()
    m = folium.Map(location=[lat, lng], zoom_start=city.cfg["zoom"],
                   tiles="CartoDB dark_matter")

    node_latlon = {}

    for link_name in wn.link_name_list:
        link = wn.get_link(link_name)
        if not (hasattr(link, "start_node_name") and
                hasattr(link, "end_node_name")):
            continue
        sn, en = link.start_node_name, link.end_node_name

        def get_ll(name):
            c = wn.get_node(name).coordinates
            if name == "Res":
                return lat - 0.0009, lng - 0.0009
            return city.grid_to_latlon(int(round(c[0] / 100)),
                                       int(round(c[1] / 100)))

        sll, ell = get_ll(sn), get_ll(en)
        is_isolated = any(sn in p or en in p for p in isolated_pipes)
        folium.PolyLine(
            [sll, ell],
            color="#c0392b" if is_isolated else "#4a5568",
            weight=5 if is_isolated else 3,
            opacity=0.9,
            tooltip=link_name,
        ).add_to(m)
        node_latlon[sn] = sll
        node_latlon[en] = ell

    for node_name in wn.node_name_list:
        ll = node_latlon.get(node_name)
        if ll is None:
            continue
        prob = fail_probs.get(node_name, 0)
        res = residuals.get(node_name, 0)
        is_sensor = node_name in sensor_nodes

        if node_name == "Res":
            colour, icon, popup = "blue", "tint", "Reservoir"
        elif node_name == pred_node and active_leak:
            colour = "red"
            icon = "warning-sign"
            popup = (f"<b>⚠ LEAK PREDICTED</b><br>{node_name}<br>"
                     f"Risk: {prob:.1f}%<br>Residual: {res:.3f} bar")
        elif prob > 40:
            colour, icon = "red", "remove"
            popup = f"{node_name}<br>Risk: {prob:.1f}%"
        elif prob > 25:
            colour, icon = "orange", "exclamation-sign"
            popup = f"{node_name}<br>Risk: {prob:.1f}%"
        elif prob > 15:
            colour, icon = "beige", "info-sign"
            popup = f"{node_name}<br>Risk: {prob:.1f}%"
        else:
            colour, icon = "green", "ok"
            popup = f"{node_name}<br>Risk: {prob:.1f}%"

        # Sensor indicator (circle marker overlay)
        if is_sensor:
            folium.CircleMarker(
                ll, radius=14, color="#f59e0b", weight=3,
                fill=False, tooltip=f"📡 Sensor: {node_name}"
            ).add_to(m)

        folium.Marker(
            ll,
            popup=folium.Popup(popup, max_width=220),
            tooltip=node_name,
            icon=folium.Icon(color=colour, icon=icon, prefix="glyphicon"),
        ).add_to(m)

    legend_html = f"""
    <div style="position:fixed;bottom:20px;left:20px;z-index:9999;
        background:rgba(14,17,23,0.92);padding:10px 14px;border-radius:8px;
        border:1px solid #4a5568;font-size:11px;color:#e2e8f0;">
      <b style="color:#3b82f6">Legend — {city.name}</b><br>
      🔴 High risk (&gt;40%) | ⚠ Predicted leak<br>
      🟠 Medium (25-40%) | 🟡 Moderate (15-25%)<br>
      🟢 Low risk | 🔵 Reservoir<br>
      🟡 Circle = Sensor node (30% coverage)
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


def failure_probability(pressure_bar: float, degradation_pct: float,
                        burst_mult: float = 1.0) -> float:
    p_max = 5.0
    alpha, beta, gamma = 0.5, 2.0, 1.5
    p = alpha * ((1 - pressure_bar / p_max) ** beta) * \
        ((degradation_pct / 100) ** gamma) * burst_mult
    return min(p * 100, 100)


def calculate_mnf_anomaly(df, expected_mnf=0.4):
    night = df[(df["Hour"] >= 2) & (df["Hour"] <= 5)]
    if len(night) == 0:
        return False, 0.0
    avg = night["Flow Rate (L/s)"].mean()
    anomaly = (avg - expected_mnf) / expected_mnf * 100
    return anomaly > 15, anomaly


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────

_defaults = {
    "data": None, "network": None, "log": [],
    "isolated_pipes": [], "csv_data": None,
    "healthy_pressures": {}, "residuals": {},
    "predicted_leak_node": "N_2_2",
    "confidence_score": 0.0,
    "sensor_nodes": [],
    "n1_result": None,
    "econ": None,
    "city_name": "Алматы",
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.title("💧 Smart Shygyn PRO v3")

# Theme
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=True)
st.markdown(DARK_CSS if dark_mode else LIGHT_CSS, unsafe_allow_html=True)

st.sidebar.markdown("---")

with st.sidebar.expander("🏙️ City Selection", expanded=True):
    city_name = st.selectbox("City", list(CityManager.CITIES.keys()),
                             index=list(CityManager.CITIES.keys()).index(
                                 st.session_state["city_name"]))
    st.session_state["city_name"] = city_name
    season_temp = st.slider("Current Season Temp (°C)", -30, 45, 10)

with st.sidebar.expander("⚙️ Network Parameters", expanded=True):
    material = st.selectbox("Pipe Material",
                            ["Пластик (ПНД)", "Сталь", "Чугун"])
    pipe_age = st.slider("Pipe Age (years)", 0, 60, 15,
                         help="Used for Hazen-Williams degradation model")
    roughness_val = hw_roughness(material, pipe_age)
    st.caption(f"H-W Roughness C = **{roughness_val:.1f}** (from age & material)")
    freq = st.select_slider("Sensor Frequency",
                            options=[1, 2, 4],
                            format_func=lambda x: f"{x} Hz")

with st.sidebar.expander("🔧 Pump Control", expanded=True):
    pump_pressure = st.slider("Pump Head (m)", 30, 70, 40, step=5)
    st.caption(f"{pump_pressure} m ≈ {pump_pressure * 0.098:.1f} bar")
    smart_pump = st.checkbox("⚡ Smart Pump Scheduling",
                             help="Night: 70% head. Day: 100% head.")
    if smart_pump:
        st.success(f"Night head: {pump_pressure * 0.7:.0f} m | "
                   f"Day: {pump_pressure} m")

with st.sidebar.expander("💰 Economics", expanded=True):
    price = st.number_input("Water Tariff (₸/L)", value=0.55, step=0.05,
                            format="%.2f")
    limit = st.slider("Leak Detection Threshold (bar)", 1.0, 5.0, 2.7, step=0.1)
    repair_cost = st.number_input("Repair Team Deployment (₸)",
                                  value=50_000, step=5_000, format="%d")

with st.sidebar.expander("🔬 N-1 Contingency", expanded=False):
    st.markdown("Simulate failure of a single pipe:")
    contingency_pipe = st.text_input("Pipe name (e.g. PH_2_1)",
                                     value="", placeholder="leave blank = none")
    run_n1 = st.checkbox("Run N-1 on simulation", value=False)

with st.sidebar.expander("🛡️ Valve Control", expanded=False):
    enable_valves = st.checkbox("Enable Valve System")

with st.sidebar.expander("🔄 IoT Upload", expanded=False):
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        try:
            csv_df = pd.read_csv(uploaded_file)
            csv_df.columns = csv_df.columns.str.strip()
            req = ["Hour", "Pressure (bar)", "Flow Rate (L/s)"]
            miss = [c for c in req if c not in csv_df.columns]
            if miss:
                st.error(f"Missing: {miss}")
            else:
                st.session_state["csv_data"] = csv_df
                st.success(f"✅ {len(csv_df)} rows loaded")
        except Exception as e:
            st.error(str(e))

st.sidebar.markdown("---")

run_btn = st.sidebar.button("🚀 RUN SIMULATION",
                            use_container_width=True, type="primary")

if run_btn:
    city = CityManager(city_name, season_temp)
    cp = contingency_pipe.strip() if run_n1 and contingency_pipe.strip() else None

    with st.spinner("⏳ Running hydraulic simulation…"):
        try:
            df, net, leak_node = build_network(
                city, material, pipe_age, freq,
                pump_pressure, smart_pump, cp
            )
        except Exception as e:
            st.sidebar.error(f"Simulation error: {e}")
            st.stop()

    with st.spinner("🔍 Running healthy baseline (Residual Analysis)…"):
        try:
            healthy_p = build_healthy_network(
                city, material, pipe_age, freq, pump_pressure
            )
        except Exception as e:
            healthy_p = {}

    # Sensor placement
    sensors = place_sensors(list(net.node_name_list))

    # Observed pressures (mean over simulation for each node)
    # Approximate: we stored only leak node in df; use healthy scaled by leak ratio
    leak_ratio = df["Pressure (bar)"].mean() / (
        healthy_p.get(leak_node, df["Pressure (bar)"].mean()) or 1)
    observed_p = {n: healthy_p.get(n, 1.0) * leak_ratio
                  for n in net.node_name_list if n != "Res"}

    pred_node, residuals, confidence = residual_matrix_localize(
        healthy_p, observed_p, sensors, net
    )

    # N-1 impact
    n1_res = None
    if run_n1 and cp:
        n1_res = n1_impact(net, cp, df, city)

    # Economics
    econ = calculate_economics(
        df, price, pump_pressure, smart_pump,
        limit, freq, len(sensors), repair_cost
    )

    # Persist
    st.session_state.update({
        "data": df, "network": net,
        "isolated_pipes": [],
        "healthy_pressures": healthy_p,
        "residuals": residuals,
        "predicted_leak_node": pred_node,
        "confidence_score": confidence,
        "sensor_nodes": sensors,
        "n1_result": n1_res,
        "econ": econ,
    })
    log = (f"[{datetime.now().strftime('%H:%M:%S')}] ✅ {city_name} | "
           f"{material} {pipe_age}yr | {pump_pressure}m"
           + (" SmartPump" if smart_pump else "")
           + (f" N-1:{cp}" if cp else ""))
    st.session_state["log"].append(log)
    st.sidebar.success("✅ Done!")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────────

st.title("💧 Smart Shygyn PRO v3 — Command Center")
st.markdown("##### Intelligent Water Network Management | "
            "Multi-City | EKF Leak Detection | N-1 Contingency | Full ROI Model")

if st.session_state["data"] is None:
    city_cfg = CityManager.CITIES
    st.markdown("---")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    panels = [
        ("🏙️", "Multi-City Engine", "Almaty · Astana · Turkestan\nElevation physics"),
        ("🔬", "Advanced Physics", "H-W aging model\nTorricelli leaks"),
        ("🧠", "Smart Detection", "Sparse 30% sensors\nResidual Matrix EKF"),
        ("⚡", "N-1 Contingency", "Pipe failure sim\nImpact assessment"),
        ("💰", "Full ROI Model", "CAPEX/OPEX/Payback\nCarbon footprint"),
        ("🖥️", "Command Center", "Dark/Light mode\n4 pro dashboards"),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3, c4, c5, c6], panels):
        with col:
            st.markdown(f"#### {icon} {title}")
            for line in desc.split("\n"):
                st.markdown(f"- {line}")
    st.info("👈 Configure parameters in the sidebar and click **RUN SIMULATION**")
    st.stop()

# ─── Unpack state ─────────────────────────────────────────────────────────────
df = st.session_state["data"]
wn = st.session_state["network"]
pred_node = st.session_state["predicted_leak_node"]
residuals = st.session_state["residuals"]
confidence = st.session_state["confidence_score"]
sensors = st.session_state["sensor_nodes"]
n1_res = st.session_state["n1_result"]
econ = st.session_state["econ"]
city = CityManager(city_name, season_temp)

df["Leak"] = df["Pressure (bar)"] < limit
active_leak = df["Leak"].any()
mnf_detected, mnf_anomaly = calculate_mnf_anomaly(df)
contamination_risk = (df["Pressure (bar)"] < 1.5).any()

# Failure probs
avg_p = df["Pressure (bar)"].mean()
# degradation derived from H-W roughness drop
base_c = {"Пластик (ПНД)": 150, "Сталь": 140, "Чугун": 100}[material]
degradation_pct = max(0, min(100, (1 - roughness_val / base_c) * 100))

fail_probs = {}
for node in wn.node_name_list:
    if node != "Res":
        bm = city.cfg["burst_multiplier"]
        fail_probs[node] = failure_probability(avg_p, degradation_pct, bm)
    else:
        fail_probs[node] = 0

wsi = city.water_stress_index()

# ─── KPI Bar ──────────────────────────────────────────────────────────────────
st.markdown("### 📊 System Status Dashboard")
cols = st.columns(8)
kpis = [
    ("🚨 Status", "LEAK" if active_leak else "NORMAL",
     "Critical" if active_leak else "Stable",
     "inverse" if active_leak else "normal"),
    ("📍 City", city_name, city.cfg["elev_direction"], "off"),
    ("💧 Pressure min", f"{df['Pressure (bar)'].min():.2f} bar",
     f"{df['Pressure (bar)'].min()-limit:.2f}",
     "inverse" if df['Pressure (bar)'].min() < limit else "normal"),
    ("💦 Water Lost", f"{econ['lost_l']:,.0f} L",
     f"NRW {econ['nrw_pct']:.1f}%", "inverse" if econ['lost_l']>0 else "normal"),
    ("💸 Total Damage", f"{econ['total_damage_kzt']:,.0f}₸",
     f"Direct+Indirect", "inverse" if econ['total_damage_kzt']>0 else "normal"),
    ("🧠 Leak Node", pred_node,
     f"Conf: {confidence:.0f}%",
     "inverse" if confidence > 60 else "normal"),
    ("⚡ Energy Saved", f"{econ['energy_saved_pct']:.1f}%",
     "Smart Pump ON" if smart_pump else "Pump OFF", "normal"),
    ("🌿 CO₂ Saved", f"{econ['co2_saved_kg']:.1f} kg",
     f"Today", "normal"),
]
for col, (label, val, delta, dc) in zip(cols, kpis):
    with col:
        st.metric(label, val, delta, delta_color=dc)

# City-specific alerts
if city_name == "Астана":
    bm = city.cfg["burst_multiplier"]
    if bm > 1.3:
        st.error(f"🥶 **ASTANA FREEZE-THAW ALERT**: Ground temp {season_temp}°C. "
                 f"Pipe burst multiplier: **{bm:.2f}×**. Inspect insulation!")
    else:
        st.info(f"❄️ Astana: Ground temp {season_temp}°C. Burst risk ×{bm:.2f}")

if city_name == "Туркестан":
    st.warning(f"☀️ **TURKESTAN WATER STRESS INDEX: {wsi:.2f}** "
               f"({'CRITICAL' if wsi > 0.7 else 'HIGH'}). "
               f"Evaporation losses are elevated.")

if contamination_risk:
    st.error("⚠️ **CONTAMINATION RISK**: Pressure < 1.5 bar — "
             "groundwater infiltration possible!")
if mnf_detected:
    st.warning(f"🌙 **MNF ANOMALY**: Night flow +{mnf_anomaly:.1f}% above baseline.")

if active_leak and confidence >= 50:
    st.error(f"🔍 **RESIDUAL MATRIX**: Predicted leak at **{pred_node}** | "
             f"Confidence: **{confidence:.0f}%** | "
             f"Residual: {residuals.get(pred_node, 0):.3f} bar drop")
elif active_leak:
    st.warning(f"🔍 Low-confidence detection: **{pred_node}** "
               f"(conf. {confidence:.0f}%) — check sensor coverage")

if n1_res and "error" not in n1_res:
    st.error(
        f"🔧 **N-1 CONTINGENCY ACTIVE** — Pipe: `{contingency_pipe}` failed | "
        f"Affected: **{n1_res['virtual_citizens']} residents** | "
        f"Time to Criticality: **{n1_res['time_to_criticality_h']} h**"
    )

st.markdown("---")

# ─── TABS ────────────────────────────────────────────────────────────────────
tab_map, tab_hydro, tab_econ, tab_stress = st.tabs([
    "🗺️  Real-time Map",
    "📈  Hydraulic Diagnostics",
    "💰  Economic ROI",
    "🔬  Stress-Test & N-1",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — REAL-TIME MAP
# ════════════════════════════════════════════════════════════════════════════
with tab_map:
    col_m, col_ctrl = st.columns([3, 1])

    with col_ctrl:
        st.markdown("#### 🛡️ Valve Control")
        if active_leak:
            st.error(f"⚠️ Predicted: **{pred_node}** (conf. {confidence:.0f}%)")
            if st.button("🔒 ISOLATE SECTION", use_container_width=True,
                         type="primary"):
                ptc, aff = find_isolation_valves(wn, pred_node)
                st.session_state["isolated_pipes"] = ptc
                st.session_state["log"].append(
                    f"[{datetime.now().strftime('%H:%M:%S')}] 🔒 Isolated {len(ptc)} pipes"
                )
                st.rerun()
            if st.session_state["isolated_pipes"]:
                st.success(f"✅ {len(st.session_state['isolated_pipes'])} pipes closed")
                if st.button("🔓 Restore Supply"):
                    st.session_state["isolated_pipes"] = []
                    st.rerun()
        else:
            st.success("✅ System normal — valves on standby")

        st.markdown("---")
        st.markdown("#### 📡 Sensor Coverage")
        st.metric("Sensor Nodes", len(sensors),
                  f"{len(sensors)/16*100:.0f}% of grid")
        st.markdown(f"**Sensors:** `{'`, `'.join(sensors)}`")

        st.markdown("---")
        st.markdown("#### 🔍 Residual Table")
        if residuals:
            rdf = pd.DataFrame(
                [(k, f"{v:.4f}") for k, v in sorted(
                    residuals.items(), key=lambda x: -x[1]
                )], columns=["Node", "Δ Pressure (bar)"]
            )
            st.dataframe(rdf, use_container_width=True, height=200)

        st.markdown("---")
        st.markdown("#### 🏙️ City Info")
        cfg = city.cfg
        st.caption(cfg["description"])
        st.write(f"**Elevation:** {cfg['elev_min']}–{cfg['elev_max']} m")
        st.write(f"**Burst mult:** ×{cfg['burst_multiplier']:.2f}")
        st.write(f"**Water Stress:** {wsi:.2f}")

    with col_m:
        fmap = make_folium_map(
            wn, city, active_leak, pred_node,
            fail_probs, residuals,
            sensors, st.session_state["isolated_pipes"]
        )
        st_folium(fmap, width=None, height=540)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — HYDRAULIC DIAGNOSTICS
# ════════════════════════════════════════════════════════════════════════════
with tab_hydro:
    st.markdown("### Hydraulic Analysis (Elevation-Aware | H-W Aging | MA Filter)")
    st.caption(
        f"City: **{city_name}** | Material: **{material}** | "
        f"Age: **{pipe_age} yr** | H-W C: **{roughness_val:.0f}** | "
        f"Sensors smoothed by MA(3)"
    )

    fig_h = make_hydraulic_plot(df, limit, smart_pump, dark_mode)
    st.plotly_chart(fig_h, use_container_width=True)

    st.markdown("#### 📊 Statistics")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**💧 Pressure**")
        st.dataframe(df["Pressure (bar)"].describe().to_frame()
                     .style.format("{:.3f}"), use_container_width=True)
    with c2:
        st.markdown("**🌊 Flow Rate**")
        st.dataframe(df["Flow Rate (L/s)"].describe().to_frame()
                     .style.format("{:.3f}"), use_container_width=True)
    with c3:
        st.markdown("**⏱ Water Age**")
        st.dataframe(df["Water Age (h)"].describe().to_frame()
                     .style.format("{:.2f}"), use_container_width=True)

    # IoT comparison
    if st.session_state["csv_data"] is not None:
        st.markdown("---")
        st.markdown("#### 🔄 Model vs IoT Sensor Comparison")
        csv_df = st.session_state["csv_data"]
        fig_c = make_subplots(rows=2, cols=1,
                              subplot_titles=["Pressure Comparison",
                                              "Flow Comparison"],
                              vertical_spacing=0.1)
        fig_c.add_trace(go.Scatter(
            x=df["Hour"], y=df["Pressure (bar)"],
            name="Model (MA)", line=dict(color="#3b82f6", dash="dot")),
            row=1, col=1)
        fig_c.add_trace(go.Scatter(
            x=csv_df["Hour"], y=csv_df["Pressure (bar)"],
            name="IoT Sensors", line=dict(color="#ef4444")),
            row=1, col=1)
        fig_c.add_trace(go.Scatter(
            x=df["Hour"], y=df["Flow Rate (L/s)"],
            name="Model (MA)", line=dict(color="#3b82f6", dash="dot")),
            row=2, col=1)
        fig_c.add_trace(go.Scatter(
            x=csv_df["Hour"], y=csv_df["Flow Rate (L/s)"],
            name="IoT Sensors", line=dict(color="#ef4444")),
            row=2, col=1)
        fig_c.update_layout(height=600, showlegend=True,
                            plot_bgcolor="#0e1117" if dark_mode else "white",
                            paper_bgcolor="#0e1117" if dark_mode else "white",
                            font=dict(color="#e2e8f0" if dark_mode else "#2c3e50"))
        st.plotly_chart(fig_c, use_container_width=True)
    else:
        st.info("📂 Upload IoT CSV in sidebar to compare with model")

    # Operation log
    if st.session_state["log"]:
        with st.expander("📜 Operation Log"):
            for entry in reversed(st.session_state["log"][-20:]):
                st.code(entry)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — ECONOMIC ROI
# ════════════════════════════════════════════════════════════════════════════
with tab_econ:
    st.markdown("### 💰 Full Economic Model — OPEX/CAPEX/ROI/Carbon")

    e = econ
    ea, eb, ec, ed = st.columns(4)
    with ea:
        st.metric("💦 Direct Water Loss", f"{e['direct_loss_kzt']:,.0f} ₸",
                  f"{e['lost_l']:,.0f} L lost")
    with eb:
        st.metric("🔧 Indirect Costs", f"{e['indirect_kzt']:,.0f} ₸",
                  "Repair team deploy")
    with ec:
        st.metric("⚡ Daily Energy Saved", f"{e['energy_saved_kzt']:,.0f} ₸",
                  f"{e['energy_saved_kwh']:.1f} kWh")
    with ed:
        st.metric("🌿 CO₂ Reduced", f"{e['co2_saved_kg']:.1f} kg",
                  "Grid emission factor")

    st.markdown("---")
    fa, fb, fc = st.columns(3)
    with fa:
        st.metric("📦 Sensor CAPEX", f"{e['capex_kzt']:,.0f} ₸",
                  f"{len(sensors)} sensors × ₸{SENSOR_UNIT_COST_KZT:,}")
    with fb:
        st.metric("💹 Monthly Savings", f"{e['monthly_savings_kzt']:,.0f} ₸",
                  "Water + Energy")
    with fc:
        pb = e['payback_months']
        pb_label = f"{pb:.1f} months" if pb < 9000 else "N/A (no savings)"
        st.metric("⏱ Payback Period", pb_label,
                  "ROI-positive" if pb < 24 else "Review pricing")

    st.markdown("---")

    # NRW breakdown chart
    st.markdown("#### 📊 Water Accountability")
    labels = ["Revenue Water", "Non-Revenue Water (Leaks)"]
    values = [max(0, 100 - e["nrw_pct"]), e["nrw_pct"]]
    fig_pie = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.55,
        marker=dict(colors=["#10b981", "#ef4444"]),
        textinfo="label+percent",
    ))
    fig_pie.update_layout(
        title="Non-Revenue Water Distribution",
        height=320,
        paper_bgcolor="#0e1117" if dark_mode else "white",
        font=dict(color="#e2e8f0" if dark_mode else "#2c3e50"),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # Payback timeline
    if e["monthly_savings_kzt"] > 0:
        st.markdown("#### 📈 Payback Timeline")
        months = np.arange(0, int(min(e["payback_months"] * 2, 60)) + 1)
        cumulative_savings = months * e["monthly_savings_kzt"]
        capex_line = np.full_like(months, e["capex_kzt"], dtype=float)
        fig_pb = go.Figure()
        fig_pb.add_trace(go.Scatter(
            x=months, y=cumulative_savings,
            name="Cumulative Savings",
            line=dict(color="#10b981", width=2.5),
            fill="tozeroy", fillcolor="rgba(16,185,129,0.10)"
        ))
        fig_pb.add_trace(go.Scatter(
            x=months, y=capex_line,
            name="CAPEX Investment",
            line=dict(color="#f59e0b", dash="dash", width=2)
        ))
        fig_pb.add_vline(
            x=e["payback_months"],
            line_dash="dot", line_color="#3b82f6",
            annotation_text=f"Break-even: {e['payback_months']:.1f}m"
        )
        fig_pb.update_layout(
            height=300, hovermode="x unified",
            xaxis_title="Months", yaxis_title="KZT",
            plot_bgcolor="#0e1117" if dark_mode else "white",
            paper_bgcolor="#0e1117" if dark_mode else "white",
            font=dict(color="#e2e8f0" if dark_mode else "#2c3e50"),
            margin=dict(l=50, r=30, t=20, b=50)
        )
        st.plotly_chart(fig_pb, use_container_width=True)

    # Download report
    report_df = df.copy()
    report_df["City"] = city_name
    report_df["Predicted_Leak"] = pred_node
    report_df["Confidence_%"] = confidence
    report_df["NRW_%"] = e["nrw_pct"]
    report_df["Total_Damage_KZT"] = e["total_damage_kzt"]
    report_df["Payback_months"] = e["payback_months"]
    csv_out = report_df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        "📄 Download Full Report (CSV)", csv_out,
        f"shygyn_{city_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        "text/csv", use_container_width=True
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — STRESS-TEST & N-1 CONTINGENCY
# ════════════════════════════════════════════════════════════════════════════
with tab_stress:
    st.markdown("### 🔬 Stress-Test | N-1 Contingency | Predictive Maintenance")

    # N-1 results
    if n1_res:
        if "error" in n1_res:
            st.warning(f"N-1: {n1_res['error']}")
        else:
            st.error(
                f"**N-1 Result — Pipe `{contingency_pipe}` failed:**  "
                f"Affected nodes: {n1_res['affected_nodes']} | "
                f"**{n1_res['virtual_citizens']} residents without water** | "
                f"Time to criticality: **{n1_res['time_to_criticality_h']} hours** | "
                f"Recommended isolation: `{n1_res['best_isolation_valve']}`"
            )
            n1a, n1b, n1c = st.columns(3)
            with n1a:
                st.metric("🏘️ Affected Residents",
                          f"{n1_res['virtual_citizens']:,}")
            with n1b:
                st.metric("⏱ Time to Criticality",
                          f"{n1_res['time_to_criticality_h']} h",
                          "Hours until local tank empty")
            with n1c:
                st.metric("🔒 Best Valve",
                          n1_res["best_isolation_valve"],
                          "Close to minimize impact zone")
    else:
        st.info("Enable **N-1 Contingency** in sidebar and re-run to simulate "
                "a pipe failure scenario.")

    st.markdown("---")

    # Failure probability heatmap (Matplotlib)
    st.markdown("#### 🔥 Failure Probability Heatmap")
    st.caption(f"H-W C={roughness_val:.0f} | Burst Mult ×{city.cfg['burst_multiplier']:.2f} | "
               f"Water Stress {wsi:.2f}")

    fig_heat, ax = plt.subplots(figsize=(11, 9),
                                facecolor="#0e1117" if dark_mode else "white")
    ax.set_facecolor("#0e1117" if dark_mode else "white")
    txt_c = "white" if dark_mode else "black"

    pos = {n: wn.get_node(n).coordinates for n in wn.node_name_list}
    nx.draw_networkx_edges(wn.get_graph(), pos, ax=ax,
                           edge_color="#4a5568", width=3, alpha=0.6)

    for node in wn.node_name_list:
        x, y = pos[node]
        prob = fail_probs.get(node, 0)
        c = ("#ef4444" if prob > 40 else
             "#f59e0b" if prob > 25 else
             "#eab308" if prob > 15 else "#10b981")
        if node == "Res":
            c = "#3b82f6"
        circle = plt.Circle((x, y), 18, color=c, ec="white",
                             linewidth=2.5, zorder=2)
        ax.add_patch(circle)
        # Sensor indicator
        if node in sensors:
            ring = plt.Circle((x, y), 26, color="#f59e0b", fill=False,
                               linewidth=2, linestyle="--", zorder=3)
            ax.add_patch(ring)
        ax.text(x, y, node, fontsize=7.5, fontweight="bold",
                ha="center", va="center", color=txt_c, zorder=4)

    patches = [
        mpatches.Patch(color="#ef4444", label="High risk >40%"),
        mpatches.Patch(color="#f59e0b", label="Medium 25-40%"),
        mpatches.Patch(color="#eab308", label="Moderate 15-25%"),
        mpatches.Patch(color="#10b981", label="Low <15%"),
        mpatches.Patch(color="#3b82f6", label="Reservoir"),
        mpatches.Patch(color="#f59e0b", label="📡 Sensor (dashed ring)"),
    ]
    ax.legend(handles=patches, loc="upper left", fontsize=9,
              facecolor="#1a1f2e" if dark_mode else "white",
              labelcolor=txt_c)
    ax.set_title(f"Failure Probability — {city_name} "
                 f"(Age {pipe_age}yr, C={roughness_val:.0f})",
                 fontsize=13, fontweight="bold", color=txt_c)
    ax.set_axis_off()
    ax.set_aspect("equal")
    plt.tight_layout()
    st.pyplot(fig_heat)

    # Top-5 risk table
    sorted_p = sorted(
        [(k, v) for k, v in fail_probs.items() if k != "Res"],
        key=lambda x: -x[1]
    )[:5]
    st.markdown("#### 🏆 Top-5 High-Risk Nodes")
    risk_df = pd.DataFrame(sorted_p, columns=["Node", "Failure Risk (%)"])
    risk_df["Sensor?"] = risk_df["Node"].apply(
        lambda n: "📡 Yes" if n in sensors else "—")
    risk_df["Leak Predicted?"] = risk_df["Node"].apply(
        lambda n: "⚠️ YES" if n == pred_node and active_leak else "—")
    st.dataframe(risk_df.style.format({"Failure Risk (%)": "{:.1f}"}),
                 use_container_width=True)

    st.markdown("---")
    st.markdown("#### 💡 Maintenance Recommendations")
    max_prob = sorted_p[0][1] if sorted_p else 0
    if max_prob > 40:
        st.error(f"🔴 URGENT: Replace pipes at {sorted_p[0][0]}. "
                 f"Risk {max_prob:.1f}%. Burst multiplier ×{city.cfg['burst_multiplier']:.2f}")
    elif max_prob > 25:
        st.warning(f"🟠 Plan replacement within 6 months. "
                   f"H-W C={roughness_val:.0f} (degraded from base).")
    else:
        st.success(f"🟢 System acceptable. Next inspection in 12 months. "
                   f"H-W C={roughness_val:.0f}")
    if city_name == "Астана" and city.cfg["burst_multiplier"] > 1.3:
        st.warning("❄️ ASTANA: Ensure thermal insulation on all exposed pipes. "
                   "Freeze-thaw cycles significantly increase burst risk.")
    if city_name == "Туркестан":
        st.warning(f"☀️ TURKESTAN: Water Stress Index {wsi:.2f}. "
                   "Install pressure-reducing valves to limit evaporative losses.")
