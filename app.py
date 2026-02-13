"""
Professional Water Network Digital Twin Dashboard
Unified Streamlit entry point for Hydraulic, Leak, and Risk analytics.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ═══════════════════════════════════════════════════════════════════════════
# BACKEND IMPORTS WITH SAFETY
# ═══════════════════════════════════════════════════════════════════════════

try:
    from risk_engine import DigitalTwinEngine, DigitalTwinAPIResponse
except Exception as exc:  # pragma: no cover - UI safety
    DigitalTwinEngine = None
    DigitalTwinAPIResponse = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

try:
    from hydraulic_intelligence import MaterialDatabase
except Exception:
    MaterialDatabase = None


# ═══════════════════════════════════════════════════════════════════════════
# STREAMLIT CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Water Network Digital Twin",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS & UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

CITY_OPTIONS = ["Алматы", "Астана", "Туркестан"]
DEFAULT_GRID_SIZE = 4
DEFAULT_PIPE_AGE = 25
DEFAULT_LEAK_AREA = 0.8

COLOR_OK = "#22c55e"
COLOR_WARN = "#f59e0b"
COLOR_ALERT = "#ef4444"
COLOR_INFO = "#38bdf8"
COLOR_NEUTRAL = "#e2e8f0"


def _get_materials() -> List[str]:
    if MaterialDatabase is None:
        return ["Пластик (ПНД)", "Сталь", "Чугун", "ПВХ", "Асбестоцемент"]
    return MaterialDatabase.list_materials()


def _safe_value(value: Optional[float], default: float = 0.0) -> float:
    return default if value is None else float(value)


def _status_badge(label: str, color: str) -> str:
    return f"<span style='color:{color}; font-weight:600;'>{label}</span>"


def _coerce_api_result(result: Any) -> Optional[DigitalTwinAPIResponse]:
    if result is None:
        return None
    return result


def _build_grid_coordinates(grid_size: int) -> pd.DataFrame:
    coords = [(i, j) for i in range(grid_size) for j in range(grid_size)]
    return pd.DataFrame(coords, columns=["row", "col"])


def _build_criticality_map(result: DigitalTwinAPIResponse) -> Dict[str, float]:
    if not result or not result.criticality_assessment:
        return {}
    return {
        node: data.get("criticality_index", 0.0)
        for node, data in result.criticality_assessment.network_criticality.items()
    }


def _pressure_distribution(
    grid_size: int,
    leak_node: Optional[str],
    criticality_map: Dict[str, float],
) -> pd.DataFrame:
    coords = _build_grid_coordinates(grid_size)
    pressures = []
    for _, row in coords.iterrows():
        node = f"N_{row['row']}_{row['col']}"
        base_pressure = 3.2
        risk_penalty = criticality_map.get(node, 0.0) * 1.2
        leak_penalty = 0.6 if leak_node == node else 0.0
        noise = np.random.normal(0, 0.05)
        pressures.append(max(0.5, base_pressure - risk_penalty - leak_penalty + noise))
    coords["pressure_bar"] = pressures
    coords["node"] = [f"N_{r}_{c}" for r, c in coords[["row", "col"]].to_numpy()]
    return coords


def _risk_map(grid_size: int, criticality_map: Dict[str, float]) -> pd.DataFrame:
    coords = _build_grid_coordinates(grid_size)
    risk_values = []
    for _, row in coords.iterrows():
        node = f"N_{row['row']}_{row['col']}"
        risk_values.append(criticality_map.get(node, 0.0))
    coords["criticality"] = risk_values
    coords["node"] = [f"N_{r}_{c}" for r, c in coords[["row", "col"]].to_numpy()]
    return coords


def _simulate_chlorine_decay(avg_age: float) -> Tuple[np.ndarray, np.ndarray]:
    time_hours = np.linspace(0, 72, 200)
    c0 = 0.5
    decay_rate = 0.05
    concentration = c0 * np.exp(-decay_rate * time_hours)
    if avg_age > 0:
        concentration = np.maximum(concentration, 0.0)
    return time_hours, concentration


def _render_metric(label: str, value: str, status: str, color: str) -> None:
    st.markdown(
        f"""
        <div style='background:rgba(15,23,42,0.6); padding:1rem; border-radius:12px; border:1px solid #1e293b;'>
            <div style='font-size:0.75rem; text-transform:uppercase; color:#94a3b8; margin-bottom:0.25rem;'>
                {label}
            </div>
            <div style='font-size:1.6rem; font-weight:700; color:{color};'>{value}</div>
            <div style='font-size:0.85rem; color:#cbd5f5;'>{status}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════

if "simulation_result" not in st.session_state:
    st.session_state.simulation_result = None
if "simulation_params" not in st.session_state:
    st.session_state.simulation_params = {}
if "simulation_error" not in st.session_state:
    st.session_state.simulation_error = None


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR INPUTS
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Simulation Controls")
    st.markdown("---")

    city = st.selectbox("City Selection", CITY_OPTIONS)
    material = st.selectbox("Pipe Material", _get_materials())
    pipe_age = st.slider("Installation Age (years)", 0, 100, DEFAULT_PIPE_AGE)
    grid_size = st.slider("Grid Dimensions (N×N)", 3, 8, DEFAULT_GRID_SIZE)

    st.markdown("#### Leak Coordinates")
    leak_row = st.number_input("Leak Row", min_value=0, max_value=grid_size - 1, value=1)
    leak_col = st.number_input("Leak Column", min_value=0, max_value=grid_size - 1, value=1)
    leak_area = st.slider("Leak Area (cm²)", 0.0, 5.0, DEFAULT_LEAK_AREA, step=0.1)

    st.markdown("---")
    run_simulation = st.button("🚀 Run Digital Twin Simulation", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# RUN ENGINE WITH CACHING
# ═══════════════════════════════════════════════════════════════════════════

if run_simulation:
    if _IMPORT_ERROR is not None or DigitalTwinEngine is None:
        st.session_state.simulation_error = (
            f"Backend import failed: {_IMPORT_ERROR}. Ensure risk_engine.py is available."
        )
        st.session_state.simulation_result = None
    else:
        st.session_state.simulation_error = None
        try:
            engine = DigitalTwinEngine(
                city=city,
                season_temp_celsius=10.0,
                material=material,
                pipe_age=float(pipe_age),
            )
            leak_node = f"N_{int(leak_row)}_{int(leak_col)}" if leak_area > 0 else None
            result = engine.run_complete_analysis(
                grid_size=int(grid_size),
                leak_node=leak_node,
                leak_area_cm2=float(leak_area),
                n_sensors=max(3, grid_size),
            )
            st.session_state.simulation_result = result
            st.session_state.simulation_params = {
                "city": city,
                "material": material,
                "pipe_age": pipe_age,
                "grid_size": grid_size,
                "leak_node": leak_node,
                "leak_area": leak_area,
            }
        except Exception as exc:  # pragma: no cover - UI safety
            st.session_state.simulation_error = str(exc)
            st.session_state.simulation_result = None


# ═══════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ═══════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <div style='padding:1rem 0;'>
        <h1 style='margin-bottom:0;'>💧 Water Network Digital Twin</h1>
        <p style='color:#94a3b8; margin-top:0.2rem;'>Professional hydraulic intelligence, leak analytics, and risk assessment dashboard.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.session_state.simulation_error:
    st.error(f"Simulation failed: {st.session_state.simulation_error}")

result = _coerce_api_result(st.session_state.simulation_result)

if result is None:
    st.info("Run the simulation from the sidebar to populate the dashboard.")
    st.stop()

if getattr(result, "status", "SUCCESS") != "SUCCESS":
    st.error("Simulation completed with errors.")
    if result.errors:
        st.write(result.errors)

leak_detection = result.leak_detection
water_quality = result.water_quality
criticality = result.criticality_assessment

leak_detected = bool(leak_detection.leak_detected) if leak_detection else False
leak_severity = _safe_value(leak_detection.severity_score if leak_detection else 0.0)
leak_status = "Leak Detected" if leak_detected else "No Active Leak"

quality_standard = water_quality.quality_standard if water_quality else "UNKNOWN"
avg_age = _safe_value(water_quality.avg_age_hours if water_quality else 0.0)
chlorine = _safe_value(water_quality.chlorine_residual_mg_l if water_quality else 0.0)

criticality_map = _build_criticality_map(result)

health_color = COLOR_OK
if leak_detected and leak_severity >= 70:
    health_color = COLOR_ALERT
elif leak_detected and leak_severity >= 40:
    health_color = COLOR_WARN

col1, col2, col3 = st.columns(3)
with col1:
    _render_metric("System Health", "Operational", _status_badge("Stable", health_color), health_color)
with col2:
    _render_metric("Leak Status", leak_status, f"Severity: {leak_severity:.1f}/100", COLOR_ALERT)
with col3:
    _render_metric("Pressure Stability", "Nominal", f"Avg Age: {avg_age:.1f}h", COLOR_INFO)


# ═══════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════

tab_overview, tab_hydraulic, tab_risk = st.tabs(
    ["System Overview", "Hydraulic Analytics", "Risk & Maintenance"]
)

with tab_overview:
    st.subheader("System Overview")
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("#### Key Performance Indicators")
        st.write(
            {
                "Leak Detected": leak_detected,
                "Leak Type": leak_detection.leak_type if leak_detection else "N/A",
                "Water Quality": quality_standard,
                "Compliance %": water_quality.compliance_percentage if water_quality else 0.0,
            }
        )

    with col_right:
        st.markdown("#### Status Summary")
        st.markdown(
            f"- **Leak Status:** {leak_status}\n"
            f"- **Quality Standard:** {quality_standard}\n"
            f"- **Chlorine Residual:** {chlorine:.2f} mg/L"
        )

    if result.recommendations:
        st.markdown("#### Recommendations")
        for rec in result.recommendations:
            st.markdown(f"- {rec}")

with tab_hydraulic:
    st.subheader("Hydraulic Analytics")
    grid_data = _pressure_distribution(
        st.session_state.simulation_params.get("grid_size", DEFAULT_GRID_SIZE),
        st.session_state.simulation_params.get("leak_node"),
        criticality_map,
    )

    fig_pressure = go.Figure(
        data=go.Heatmap(
            z=grid_data.pivot(index="row", columns="col", values="pressure_bar").values,
            x=sorted(grid_data["col"].unique()),
            y=sorted(grid_data["row"].unique()),
            colorscale="Blues",
            colorbar=dict(title="Pressure (bar)"),
        )
    )
    fig_pressure.update_layout(
        title="Pressure Distribution Map",
        xaxis_title="Grid Column",
        yaxis_title="Grid Row",
        height=450,
    )
    st.plotly_chart(fig_pressure, use_container_width=True)

    st.markdown("#### Water Quality (Chlorine Decay)")
    time_hours, concentration = _simulate_chlorine_decay(avg_age)
    fig_chlorine = go.Figure()
    fig_chlorine.add_trace(
        go.Scatter(
            x=time_hours,
            y=concentration,
            mode="lines",
            line=dict(color=COLOR_INFO, width=3),
            name="Chlorine Residual",
        )
    )
    fig_chlorine.add_trace(
        go.Scatter(
            x=[avg_age],
            y=[chlorine],
            mode="markers",
            marker=dict(size=12, color=COLOR_WARN),
            name="Current",
        )
    )
    fig_chlorine.update_layout(
        title="Chlorine Residual Decay Curve",
        xaxis_title="Time (hours)",
        yaxis_title="Residual (mg/L)",
        height=350,
    )
    st.plotly_chart(fig_chlorine, use_container_width=True)

with tab_risk:
    st.subheader("Risk and Maintenance")

    priorities = []
    if criticality and criticality.maintenance_priorities:
        priorities = criticality.maintenance_priorities

    if priorities:
        st.markdown("#### Maintenance Priorities")
        priority_df = pd.DataFrame(priorities)
        st.dataframe(priority_df, use_container_width=True)
    else:
        st.info("No maintenance priorities available for the current scenario.")

    st.markdown("#### Network Risk Map")
    risk_data = _risk_map(
        st.session_state.simulation_params.get("grid_size", DEFAULT_GRID_SIZE),
        criticality_map,
    )
    fig_risk = go.Figure(
        data=go.Scatter(
            x=risk_data["col"],
            y=risk_data["row"],
            mode="markers",
            marker=dict(
                size=12,
                color=risk_data["criticality"],
                colorscale="Reds",
                colorbar=dict(title="Criticality"),
                line=dict(width=1, color="white"),
            ),
            text=risk_data["node"],
            hovertemplate="Node %{text}<br>Criticality %{marker.color:.2f}<extra></extra>",
        )
    )
    fig_risk.update_layout(
        title="Spatial Criticality Scatter Map",
        xaxis_title="Grid Column",
        yaxis_title="Grid Row",
        height=450,
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_risk, use_container_width=True)

if result.warnings:
    st.warning("Warnings reported during simulation:")
    st.write(result.warnings)
