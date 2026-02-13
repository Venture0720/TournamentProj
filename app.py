"""Smart Shygyn PRO v3 Streamlit Application."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import folium
from streamlit_folium import st_folium
from typing import Dict, Tuple

from config import APP, UI, VIZ, MAPS, CITY, get_custom_css
from backend import SimulationInputs, DigitalTwinBackend, build_network_risk_map


st.set_page_config(page_title=APP.page_title, page_icon=APP.icon, layout=APP.layout)

st.markdown(get_custom_css(), unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def run_simulation_cached(params: SimulationInputs):
    backend = DigitalTwinBackend(params)
    response = backend.run_digital_twin()
    wn, results, analytics, df_pressure, df_flow, df_age = backend.run_hydraulic_analysis()
    mnf, leak_class, sensors, coverage = backend.build_leak_analytics(wn, df_pressure, df_flow)
    return response, wn, analytics, df_pressure, df_flow, df_age, mnf, leak_class, sensors, coverage


@st.cache_data(show_spinner=False)
def build_map_cached(city_name: str, graph_nodes: Dict[str, Tuple[float, float]], risk_map: Dict[str, float]):
    city = CITY.get_city(city_name)
    base_map = folium.Map(location=[city["lat"], city["lon"]], zoom_start=MAPS.default_zoom, tiles=MAPS.tile_style)

    for node, (lat, lon) in graph_nodes.items():
        risk = risk_map.get(node, 0.0)
        if risk >= 0.7:
            color = MAPS.risk_colors["CRITICAL"]
        elif risk >= 0.5:
            color = MAPS.risk_colors["HIGH"]
        elif risk >= 0.3:
            color = MAPS.risk_colors["MEDIUM"]
        else:
            color = MAPS.risk_colors["LOW"]
        folium.CircleMarker([lat, lon], radius=5, color=color, fill=True, fill_opacity=0.8, popup=f"{node} | Risk {risk:.2f}").add_to(base_map)

    return base_map



def render_sidebar():
    st.sidebar.markdown("## Simulation Controls")
    city = st.sidebar.selectbox("City", CITY.list_cities())
    material = st.sidebar.selectbox("Pipe Material", ["Пластик (ПНД)", "Сталь", "Чугун", "ПВХ", "Асбестоцемент"])
    pipe_age = st.sidebar.slider("Pipe Age (years)", 1, 60, 20)
    grid_size = st.sidebar.slider("Grid Size", 3, 8, 4)
    leak_node = st.sidebar.text_input("Leak Node (e.g., N_2_2)", "N_2_2")
    leak_area = st.sidebar.slider("Leak Area (cm²)", 0.1, 3.0, 0.8)
    pressure_setpoint = st.sidebar.slider("Reservoir Head (m)", 30.0, 60.0, 40.0)

    run = st.sidebar.button("Run Digital Twin", type="primary")

    params = SimulationInputs(
        city=city,
        pipe_age=float(pipe_age),
        material=material,
        grid_size=int(grid_size),
        leak_node=leak_node if leak_node else None,
        leak_area_cm2=float(leak_area),
        pressure_setpoint=float(pressure_setpoint),
    )
    return params, run



def render_kpis(response):
    cols = st.columns(4)
    pressure_mean = response.material_degradation.get("hazen_williams_current", 0.0) if response.material_degradation else 0.0
    leak_loss = response.leak_detection.severity_score if response.leak_detection else 0.0
    criticality = 0.0
    if response.criticality_assessment and response.criticality_assessment.high_risk_nodes:
        criticality = response.criticality_assessment.high_risk_nodes[0]["criticality_index"]
    savings = 2.4 + leak_loss * 0.1

    kpi_data = [
        ("Pressure Mean", f"{pressure_mean:.2f}", UI.cyan),
        ("Leak Loss %", f"{leak_loss:.1f}%", UI.rose),
        ("Criticality Index", f"{criticality:.2f}", UI.violet),
        ("Monthly Savings ₸M", f"{savings:.1f}", UI.emerald),
    ]

    for col, (label, value, color) in zip(cols, kpi_data):
        with col:
            st.markdown(
                f"""
                <div class="metric-glow">
                    <div class="metric-title">{label}</div>
                    <div class="metric-value" style="color:{color}">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )



def render_hydraulic_tab(df_pressure, df_flow, df_age):
    st.markdown("<div class='section-title'>Hydraulic Dynamics</div>", unsafe_allow_html=True)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("Pressure (bar)", "Flow (L/s)", "Water Age (hours)"))

    fig.add_trace(go.Scatter(x=df_pressure["time_hours"], y=df_pressure["pressure_bar"],
                             mode="lines", line=dict(color=UI.cyan, width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_flow["time_hours"], y=df_flow["flow_lps"],
                             mode="lines", line=dict(color=UI.emerald, width=3)), row=2, col=1)

    if not df_age.empty:
        fig.add_trace(go.Scatter(x=df_age["time_hours"], y=df_age["water_age_hours"],
                                 mode="lines", line=dict(color=UI.violet, width=3)), row=3, col=1)

    fig.update_layout(template=VIZ.plotly_template, height=VIZ.large_height, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)



def render_map_tab(city, graph, risk_map):
    st.markdown("<div class='section-title'>GIS Risk Map</div>", unsafe_allow_html=True)
    base = CITY.get_city(city)
    graph_nodes = {}
    for node in graph.nodes():
        if node == "Res":
            continue
        parts = node.split("_")
        if len(parts) == 3:
            i, j = int(parts[1]), int(parts[2])
            lat = base["lat"] + (i - 1.5) * 0.01
            lon = base["lon"] + (j - 1.5) * 0.01
            graph_nodes[node] = (lat, lon)

    fmap = build_map_cached(city, graph_nodes, risk_map)
    st_folium(fmap, width=900, height=520)



def render_economics_tab(response, df_flow):
    st.markdown("<div class='section-title'>Economic ROI</div>", unsafe_allow_html=True)
    months = np.arange(1, 25)
    capex = np.full_like(months, 25)
    savings = np.cumsum(np.random.uniform(0.5, 1.2, len(months)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=months, y=capex, mode="lines", line=dict(color=UI.rose, width=3), name="CAPEX"))
    fig.add_trace(go.Scatter(x=months, y=savings, mode="lines", line=dict(color=UI.emerald, width=3), name="Savings"))
    fig.update_layout(template=VIZ.plotly_template, height=VIZ.large_height)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-title'>Water Balance</div>", unsafe_allow_html=True)
    fig2 = go.Figure(data=[go.Pie(labels=["Revenue Water", "NRW"], values=[70, 30], hole=0.5,
                                  marker=dict(colors=[UI.cyan, UI.rose]))])
    fig2.update_layout(template=VIZ.plotly_template, height=VIZ.small_height)
    st.plotly_chart(fig2, use_container_width=True)



def main():
    st.markdown(f"# {APP.icon} {APP.name}")
    st.markdown(APP.tagline)

    params, run = render_sidebar()

    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    if run:
        with st.spinner("Running Digital Twin simulation..."):
            response, wn, analytics, df_pressure, df_flow, df_age, mnf, leak_class, sensors, coverage = run_simulation_cached(params)
            st.session_state.last_result = {
                "response": response,
                "wn": wn,
                "df_pressure": df_pressure,
                "df_flow": df_flow,
                "df_age": df_age,
            }

    if st.session_state.last_result is None:
        st.info("Run a simulation to load the digital twin dashboards.")
        return

    response = st.session_state.last_result["response"]
    wn = st.session_state.last_result["wn"]
    df_pressure = st.session_state.last_result["df_pressure"]
    df_flow = st.session_state.last_result["df_flow"]
    df_age = st.session_state.last_result["df_age"]

    render_kpis(response)

    tab1, tab2, tab3 = st.tabs(["\ud83d\udee0 Hydraulics", "\ud83c\udf0d GIS/Map", "\ud83d\udcb8 Economics"])

    with tab1:
        render_hydraulic_tab(df_pressure, df_flow, df_age)
    with tab2:
        risk_map = build_network_risk_map(wn.get_graph(), response.criticality_assessment)
        render_map_tab(params.city, wn.get_graph(), risk_map)
    with tab3:
        render_economics_tab(response, df_flow)


if __name__ == "__main__":
    main()
