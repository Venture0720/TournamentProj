"""
Smart Shygyn - Digital Twin Mission Control Dashboard
Sophisticated Streamlit UI for Kazakhstan's Water Networks
Astana Hub Competition 2026
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, List
import json

# Import backend modules
from config import (
    COLORS, METADATA, CONSTRAINTS, CITIES, MAPS, VIZ, RISK, PRESETS,
    get_temperature_emoji, get_status_emoji, format_number, get_severity_label
)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title=METADATA.PAGE_TITLE,
    page_icon=METADATA.ICON,
    layout=METADATA.LAYOUT,
    initial_sidebar_state="expanded"
)


# ═══════════════════════════════════════════════════════════════════════════
# CUSTOM CSS INJECTION
# ═══════════════════════════════════════════════════════════════════════════

def inject_custom_css():
    """Inject professional CSS styling using config colors"""
    css = f"""
    <style>
    /* ═══ GLOBAL STYLES ═══ */
    .stApp {{
        background: linear-gradient(135deg, {COLORS.BACKGROUND_DARK} 0%, #1e293b 100%);
    }}
    
    /* ═══ HEADER STYLING ═══ */
    .main-header {{
        background: {COLORS.GRADIENT_PRIMARY};
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }}
    
    .main-header h1 {{
        color: white;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }}
    
    .main-header p {{
        color: rgba(255, 255, 255, 0.9);
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }}
    
    /* ═══ METRIC CARDS ═══ */
    .metric-card {{
        background: linear-gradient(135deg, rgba(30, 58, 138, 0.2) 0%, rgba(6, 182, 212, 0.1) 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid {COLORS.SECONDARY_COLOR}40;
        box-shadow: 0 4px 16px rgba(6, 182, 212, 0.2);
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }}
    
    .metric-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(6, 182, 212, 0.3);
        border-color: {COLORS.SECONDARY_COLOR};
    }}
    
    .metric-label {{
        color: {COLORS.TEXT_SECONDARY};
        font-size: 0.9rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }}
    
    .metric-value {{
        color: white;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }}
    
    .metric-delta {{
        font-size: 0.85rem;
        font-weight: 500;
    }}
    
    .metric-delta.positive {{
        color: {COLORS.SUCCESS_COLOR};
    }}
    
    .metric-delta.negative {{
        color: {COLORS.DANGER_COLOR};
    }}
    
    /* ═══ BUTTONS ═══ */
    .stButton > button {{
        background: {COLORS.GRADIENT_PRIMARY};
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(6, 182, 212, 0.3);
        width: 100%;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(6, 182, 212, 0.5);
    }}
    
    /* ═══ SIDEBAR ═══ */
    .css-1d391kg, [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {COLORS.BACKGROUND_DARK} 0%, #1e293b 100%);
        border-right: 1px solid {COLORS.BORDER_COLOR}20;
    }}
    
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] h1, h2, h3 {{
        color: white !important;
        font-weight: 600;
    }}
    
    /* ═══ TABS ═══ */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: transparent;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background: rgba(30, 58, 138, 0.3);
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        color: white;
        border: 1px solid transparent;
        font-weight: 500;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: {COLORS.GRADIENT_PRIMARY};
        border-color: {COLORS.SECONDARY_COLOR};
    }}
    
    /* ═══ ALERTS ═══ */
    .alert-critical {{
        background: linear-gradient(135deg, {COLORS.DANGER_COLOR}20 0%, {COLORS.DANGER_COLOR}10 100%);
        border-left: 4px solid {COLORS.DANGER_COLOR};
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: white;
    }}
    
    .alert-warning {{
        background: linear-gradient(135deg, {COLORS.WARNING_COLOR}20 0%, {COLORS.WARNING_COLOR}10 100%);
        border-left: 4px solid {COLORS.WARNING_COLOR};
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: white;
    }}
    
    .alert-success {{
        background: linear-gradient(135deg, {COLORS.SUCCESS_COLOR}20 0%, {COLORS.SUCCESS_COLOR}10 100%);
        border-left: 4px solid {COLORS.SUCCESS_COLOR};
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: white;
    }}
    
    /* ═══ TABLES ═══ */
    .dataframe {{
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }}
    
    .dataframe thead tr th {{
        background: {COLORS.PRIMARY_COLOR} !important;
        color: white !important;
        font-weight: 600;
        padding: 12px !important;
    }}
    
    .dataframe tbody tr:hover {{
        background: {COLORS.SECONDARY_COLOR}15 !important;
    }}
    
    /* ═══ CHARTS ═══ */
    .js-plotly-plot {{
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }}
    
    /* ═══ EXPANDER ═══ */
    .streamlit-expanderHeader {{
        background: rgba(30, 58, 138, 0.2);
        border-radius: 8px;
        color: white;
        font-weight: 600;
    }}
    
    /* ═══ STATUS BADGES ═══ */
    .status-badge {{
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }}
    
    .status-critical {{
        background: {COLORS.DANGER_COLOR};
        color: white;
    }}
    
    .status-high {{
        background: {COLORS.WARNING_COLOR};
        color: white;
    }}
    
    .status-medium {{
        background: {COLORS.INFO_COLOR};
        color: white;
    }}
    
    .status-low {{
        background: {COLORS.SUCCESS_COLOR};
        color: white;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# STATE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════

def initialize_session_state():
    """Initialize Streamlit session state for persistent data"""
    if 'simulation_result' not in st.session_state:
        st.session_state.simulation_result = None
    if 'simulation_timestamp' not in st.session_state:
        st.session_state.simulation_timestamp = None
    if 'simulation_params' not in st.session_state:
        st.session_state.simulation_params = {}
    if 'history' not in st.session_state:
        st.session_state.history = []


# ═══════════════════════════════════════════════════════════════════════════
# HEADER & BRANDING
# ═══════════════════════════════════════════════════════════════════════════

def render_header():
    """Render professional header with branding"""
    st.markdown(f"""
    <div class="main-header">
        <h1>{METADATA.ICON} {METADATA.APP_NAME}</h1>
        <p>{METADATA.APP_TAGLINE}</p>
        <small style="color: rgba(255,255,255,0.7);">{METADATA.ORGANIZATION} | {METADATA.COMPETITION}</small>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR - MISSION CONTROL INPUTS
# ═══════════════════════════════════════════════════════════════════════════

def render_sidebar() -> Optional[Dict]:
    """Render professional sidebar with simulation controls"""
    with st.sidebar:
        st.markdown(f"### 🎛️ Mission Control")
        st.markdown("---")
        
        # ═══ SCENARIO PRESETS ═══
        st.markdown("#### 📋 Quick Scenarios")
        scenario = st.selectbox(
            "Select Preset",
            ["Custom"] + PRESETS.get_scenario_names(),
            help="Choose a pre-configured scenario or create custom"
        )
        
        preset_data = PRESETS.SCENARIOS.get(scenario, {}) if scenario != "Custom" else {}
        
        st.markdown("---")
        
        # ═══ LOCATION PARAMETERS ═══
        st.markdown("#### 📍 Location Settings")
        
        city = st.selectbox(
            "City",
            CITIES.get_city_list(),
            help="Select Kazakhstan city for simulation"
        )
        
        city_data = CITIES.get_city_data(city)
        default_temp = preset_data.get('soil_temp', city_data['default_temp'])
        
        soil_temp = st.slider(
            f"Soil Temperature {get_temperature_emoji(default_temp)}",
            min_value=CONSTRAINTS.MIN_SOIL_TEMP,
            max_value=CONSTRAINTS.MAX_SOIL_TEMP,
            value=float(default_temp),
            step=1.0,
            format="%.1f°C",
            help="Current soil temperature affecting pipe conditions"
        )
        
        st.markdown("---")
        
        # ═══ INFRASTRUCTURE PARAMETERS ═══
        st.markdown("#### 🏗️ Infrastructure")
        
        from config import PipeMaterial
        pipe_material = st.selectbox(
            "Pipe Material",
            PipeMaterial.get_materials_list(),
            help="Material composition of the water distribution pipes"
        )
        
        pipe_age = st.slider(
            "Pipe Age (years)",
            min_value=0,
            max_value=100,
            value=25,
            step=1,
            help="Age of the pipe infrastructure"
        )
        
        st.markdown("---")
        
        # ═══ NETWORK TOPOLOGY ═══
        st.markdown("#### 🗺️ Network Configuration")
        
        grid_size = st.slider(
            "Grid Size",
            min_value=CONSTRAINTS.MIN_GRID_SIZE,
            max_value=CONSTRAINTS.MAX_GRID_SIZE,
            value=preset_data.get('grid_size', CONSTRAINTS.DEFAULT_GRID_SIZE),
            step=1,
            help="Network grid dimensions (N×N nodes)"
        )
        
        st.info(f"📊 Total Nodes: {grid_size * grid_size}")
        
        st.markdown("---")
        
        # ═══ LEAK SIMULATION ═══
        st.markdown("#### 💧 Leak Parameters")
        
        leak_node = st.number_input(
            "Leak Node ID",
            min_value=0,
            max_value=(grid_size * grid_size) - 1,
            value=min(8, (grid_size * grid_size) - 1),
            step=1,
            help="Node where leak is simulated (0 = no leak)"
        )
        
        leak_area = st.slider(
            "Leak Area (cm²)",
            min_value=CONSTRAINTS.MIN_LEAK_AREA_CM2,
            max_value=CONSTRAINTS.MAX_LEAK_AREA_CM2,
            value=preset_data.get('leak_area_cm2', CONSTRAINTS.DEFAULT_LEAK_AREA_CM2),
            step=0.5,
            format="%.1f cm²",
            help="Equivalent leak opening area"
        )
        
        st.markdown("---")
        
        # ═══ SENSOR CONFIGURATION ═══
        st.markdown("#### 📡 Sensor Network")
        
        n_sensors = st.slider(
            "Active Sensors",
            min_value=CONSTRAINTS.MIN_SENSORS,
            max_value=CONSTRAINTS.MAX_SENSORS,
            value=preset_data.get('sensors', CONSTRAINTS.DEFAULT_SENSORS),
            step=1,
            help="Number of deployed pressure/flow sensors"
        )
        
        st.markdown("---")
        
        # ═══ RUN SIMULATION BUTTON ═══
        run_button = st.button(
            "🚀 Run Digital Twin Analysis",
            use_container_width=True,
            type="primary"
        )
        
        if run_button:
            params = {
                'city': city,
                'soil_temp': soil_temp,
                'pipe_material': pipe_material,
                'pipe_age': pipe_age,
                'grid_size': grid_size,
                'leak_node': leak_node,
                'leak_area': leak_area,
                'n_sensors': n_sensors,
                'timestamp': datetime.now()
            }
            return params
        
        # ═══ FOOTER INFO ═══
        st.markdown("---")
        st.markdown(f"""
        <small style='color: {COLORS.TEXT_SECONDARY};'>
        <b>Version:</b> {METADATA.VERSION}<br>
        <b>City:</b> {city}<br>
        <b>Coordinates:</b> {city_data['lat']:.4f}, {city_data['lon']:.4f}
        </small>
        """, unsafe_allow_html=True)
    
    return None


# ═══════════════════════════════════════════════════════════════════════════
# METRIC CARDS
# ═══════════════════════════════════════════════════════════════════════════

def render_metrics_row(result):
    """Render top-row metric cards with key KPIs"""
    
    # Extract data
    leak_detected = result.leak_detection.leak_detected
    severity = result.leak_detection.severity_score
    avg_age = result.water_quality.avg_age_hours
    chlorine = result.water_quality.chlorine_residual_mg_l
    
    # Determine system health
    if leak_detected and severity >= RISK.SEVERE_LEAK_THRESHOLD:
        health_status = "🚨 CRITICAL"
        health_color = COLORS.DANGER_COLOR
    elif leak_detected and severity >= RISK.MAJOR_LEAK_THRESHOLD:
        health_status = "⚠️ WARNING"
        health_color = COLORS.WARNING_COLOR
    elif leak_detected:
        health_status = "🔔 ALERT"
        health_color = COLORS.INFO_COLOR
    else:
        health_status = "✅ OPTIMAL"
        health_color = COLORS.SUCCESS_COLOR
    
    # Create 4 columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">System Health</div>
            <div class="metric-value" style="color: {health_color};">{health_status}</div>
            <div class="metric-delta">Real-time monitoring active</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        leak_status = "🔴 DETECTED" if leak_detected else "🟢 SECURE"
        leak_color = COLORS.DANGER_COLOR if leak_detected else COLORS.SUCCESS_COLOR
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Leak Status</div>
            <div class="metric-value" style="color: {leak_color};">{leak_status}</div>
            <div class="metric-delta">Severity: {severity:.2f}/1.00</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        age_status = "✅ FRESH" if avg_age < CONSTRAINTS.MAX_WATER_AGE_HOURS else "⚠️ AGED"
        age_color = COLORS.SUCCESS_COLOR if avg_age < CONSTRAINTS.MAX_WATER_AGE_HOURS else COLORS.WARNING_COLOR
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Water Age</div>
            <div class="metric-value" style="color: {age_color};">{avg_age:.1f}h</div>
            <div class="metric-delta">{age_status}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        chlorine_ok = CONSTRAINTS.CHLORINE_MIN_THRESHOLD <= chlorine <= CONSTRAINTS.CHLORINE_MAX_THRESHOLD
        chlorine_status = "✅ COMPLIANT" if chlorine_ok else "⚠️ OUT OF RANGE"
        chlorine_color = COLORS.SUCCESS_COLOR if chlorine_ok else COLORS.DANGER_COLOR
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Chlorine Residual</div>
            <div class="metric-value" style="color: {chlorine_color};">{chlorine:.2f} mg/L</div>
            <div class="metric-delta">{chlorine_status}</div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: HYDRAULIC NETWORK MAP
# ═══════════════════════════════════════════════════════════════════════════

def build_geo_grid(city_data: Dict, grid_size: int) -> pd.DataFrame:
    """Build geo-referenced grid around city center."""
    lat_center = city_data["lat"]
    lon_center = city_data["lon"]
    # Rough conversion: 1 grid step ~= 0.004 degrees (~400m)
    step = 0.004
    half = (grid_size - 1) / 2.0
    rows = []
    for i in range(grid_size):
        for j in range(grid_size):
            node_id = i * grid_size + j
            rows.append({
                "node_id": node_id,
                "row": i,
                "col": j,
                "lat": lat_center + (i - half) * step,
                "lon": lon_center + (j - half) * step,
            })
    return pd.DataFrame(rows)


def render_geo_network_map(result, grid_size: int, leak_node: int, city: str):
    """Render geo-referenced network map with risk + leak overlays."""
    st.markdown("### 🗺️ Geographic Network Map")

    city_data = CITIES.get_city_data(city)
    geo_df = build_geo_grid(city_data, grid_size)

    # Criticality by node
    criticality_map = {}
    if result.criticality_assessment and result.criticality_assessment.maintenance_priorities:
        for priority in result.criticality_assessment.maintenance_priorities:
            criticality_map[priority["node"]] = priority["criticality_index"]

    geo_df["criticality"] = geo_df["node_id"].map(lambda n: criticality_map.get(n, 0.0))
    geo_df["risk_class"] = geo_df["criticality"].apply(RISK.get_risk_class)

    # Base figure with mapbox
    fig = go.Figure()

    # Risk points
    fig.add_trace(go.Scattermapbox(
        lat=geo_df["lat"],
        lon=geo_df["lon"],
        mode="markers",
        marker=dict(
            size=12,
            color=geo_df["criticality"],
            colorscale=VIZ.RISK_COLORSCALE,
            cmin=0,
            cmax=1,
            colorbar=dict(title="Criticality"),
        ),
        text=[
            f"Node {row.node_id}<br>Risk: {row.risk_class}<br>CI: {row.criticality:.3f}"
            for row in geo_df.itertuples()
        ],
        hoverinfo="text",
        name="Criticality",
    ))

    # Leak overlay
    if leak_node is not None and leak_node >= 0:
        leak_row = geo_df[geo_df["node_id"] == leak_node]
        if not leak_row.empty:
            fig.add_trace(go.Scattermapbox(
                lat=leak_row["lat"],
                lon=leak_row["lon"],
                mode="markers",
                marker=dict(size=18, color=COLORS.DANGER_COLOR, symbol="x"),
                text=["Leak Detected"],
                hoverinfo="text",
                name="Leak",
            ))

    # Mapbox layout
    map_style = MAPS.DEFAULT_STYLE if hasattr(MAPS, "DEFAULT_STYLE") else "OpenStreetMap"
    fig.update_layout(
        mapbox=dict(
            style="open-street-map" if map_style.lower() == "openstreetmap" else "carto-darkmatter",
            center=dict(lat=city_data["lat"], lon=city_data["lon"]),
            zoom=MAPS.DEFAULT_ZOOM,
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=VIZ.LARGE_CHART_HEIGHT,
        template=VIZ.PLOTLY_TEMPLATE,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_hydraulic_map(result, grid_size: int, leak_node: int):
    """Render interactive hydraulic network visualization"""
    
    st.markdown("### 🗺️ Hydraulic Network Topology")
    
    # Generate grid coordinates
    nodes = []
    edges = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            node_id = i * grid_size + j
            nodes.append({
                'id': node_id,
                'x': j,
                'y': i,
                'row': i,
                'col': j
            })
            
            # Add edges (connections)
            if j < grid_size - 1:  # Right connection
                edges.append((node_id, node_id + 1))
            if i < grid_size - 1:  # Down connection
                edges.append((node_id, node_id + grid_size))
    
    # Get criticality data
    criticality_map = {}
    if hasattr(result, 'criticality_assessment') and result.criticality_assessment:
        for priority in result.criticality_assessment.maintenance_priorities:
            criticality_map[priority['node']] = priority['criticality_index']
    
    # Create figure
    fig = go.Figure()
    
    # Draw edges (pipes)
    for edge in edges:
        n1, n2 = edge
        node1 = nodes[n1]
        node2 = nodes[n2]
        
        fig.add_trace(go.Scatter(
            x=[node1['x'], node2['x']],
            y=[node1['y'], node2['y']],
            mode='lines',
            line=dict(color=COLORS.SECONDARY_COLOR, width=2, dash='solid'),
            hoverinfo='skip',
            showlegend=False,
            opacity=0.6
        ))
    
    # Prepare node data
    node_x = [n['x'] for n in nodes]
    node_y = [n['y'] for n in nodes]
    node_colors = []
    node_sizes = []
    node_text = []
    
    for node in nodes:
        node_id = node['id']
        criticality = criticality_map.get(node_id, 0.0)
        
        # Color based on criticality or leak status
        if node_id == leak_node:
            color = COLORS.DANGER_COLOR
            size = 25
        elif criticality >= RISK.CRITICAL_THRESHOLD:
            color = COLORS.DANGER_COLOR
            size = 18
        elif criticality >= RISK.HIGH_THRESHOLD:
            color = COLORS.WARNING_COLOR
            size = 15
        elif criticality >= RISK.MEDIUM_THRESHOLD:
            color = COLORS.INFO_COLOR
            size = 12
        else:
            color = COLORS.SUCCESS_COLOR
            size = 10
        
        node_colors.append(color)
        node_sizes.append(size)
        
        # Hover text
        risk_class = RISK.get_risk_class(criticality)
        text = f"Node {node_id}<br>Position: ({node['row']}, {node['col']})<br>"
        text += f"Criticality: {criticality:.3f}<br>Risk: {risk_class}"
        if node_id == leak_node:
            text += "<br><b>⚠️ LEAK DETECTED</b>"
        node_text.append(text)
    
    # Draw nodes
    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color='white'),
            symbol='circle'
        ),
        text=[str(n['id']) for n in nodes],
        textposition='middle center',
        textfont=dict(size=8, color='white', family='Arial Black'),
        hovertext=node_text,
        hoverinfo='text',
        showlegend=False
    ))
    
    # Layout
    fig.update_layout(
        title="Network Node Criticality Heatmap",
        template=VIZ.PLOTLY_TEMPLATE,
        height=VIZ.DEFAULT_CHART_HEIGHT,
        xaxis=dict(
            title="Column",
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            zeroline=False
        ),
        yaxis=dict(
            title="Row",
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            zeroline=False,
            autorange='reversed'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Legend
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<span class='status-badge status-critical'>🔴 Critical Risk</span>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<span class='status-badge status-high'>🟠 High Risk</span>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<span class='status-badge status-medium'>🔵 Medium Risk</span>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<span class='status-badge status-low'>🟢 Low Risk</span>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: LEAK ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════

def render_leak_analytics(result):
    """Render leak detection analytics with charts"""
    
    st.markdown("### 💧 Leak Detection & Flow Analysis")
    
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown("#### 📊 24-Hour Flow Demand Profile")
        
        # Generate 24-hour flow profile
        hours = np.arange(0, 24, 0.5)
        
        # Base demand pattern (typical residential)
        base_demand = 50 + 30 * np.sin((hours - 6) * np.pi / 12)  # L/s
        base_demand = np.maximum(base_demand, 20)  # Minimum 20 L/s
        
        # Add leak component if detected
        leak_flow = 0
        if result.leak_detection.leak_detected:
            leak_flow = result.leak_detection.estimated_flow_lps
        
        total_flow = base_demand + leak_flow
        
        # Create stacked area chart
        fig = go.Figure()
        
        # Base demand
        fig.add_trace(go.Scatter(
            x=hours,
            y=base_demand,
            name='Normal Demand',
            mode='lines',
            line=dict(width=0),
            fillcolor=COLORS.SUCCESS_COLOR,
            fill='tozeroy',
            stackgroup='one'
        ))
        
        # Leak component
        if leak_flow > 0:
            fig.add_trace(go.Scatter(
                x=hours,
                y=[leak_flow] * len(hours),
                name='Leak Loss',
                mode='lines',
                line=dict(width=0),
                fillcolor=COLORS.DANGER_COLOR,
                fill='tonexty',
                stackgroup='one'
            ))
        
        # MNF zone highlight (2-4 AM)
        fig.add_vrect(
            x0=2, x1=4,
            fillcolor=COLORS.INFO_COLOR,
            opacity=0.2,
            line_width=0,
            annotation_text="MNF Zone",
            annotation_position="top left"
        )
        
        fig.update_layout(
            title="Flow Demand with Leak Component",
            xaxis_title="Hour of Day",
            yaxis_title="Flow Rate (L/s)",
            template=VIZ.PLOTLY_TEMPLATE,
            height=VIZ.DEFAULT_CHART_HEIGHT,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # MNF Analysis
        if hasattr(result.leak_detection, 'mnf_analysis') and result.leak_detection.mnf_analysis:
            mnf = result.leak_detection.mnf_analysis
            st.info(f"""
            **Minimum Night Flow Analysis:**
            - MNF: {mnf.get('mnf_lps', 0):.2f} L/s
            - Expected: {mnf.get('expected_mnf_lps', 0):.2f} L/s
            - Anomaly: {mnf.get('anomaly_detected', False)}
            """)
    
    with col_right:
        st.markdown("#### 🎯 Severity Gauge")
        
        severity = result.leak_detection.severity_score
        
        # Create gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=severity,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Severity Score", 'font': {'size': 18, 'color': 'white'}},
            delta={'reference': RISK.MODERATE_LEAK_THRESHOLD, 'increasing': {'color': COLORS.DANGER_COLOR}},
            gauge={
                'axis': {'range': [None, 1], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': COLORS.SECONDARY_COLOR},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 2,
                'bordercolor': "white",
                'steps': [
                    {'range': [0, RISK.MINOR_LEAK_THRESHOLD], 'color': COLORS.SUCCESS_COLOR},
                    {'range': [RISK.MINOR_LEAK_THRESHOLD, RISK.MODERATE_LEAK_THRESHOLD], 'color': COLORS.INFO_COLOR},
                    {'range': [RISK.MODERATE_LEAK_THRESHOLD, RISK.MAJOR_LEAK_THRESHOLD], 'color': COLORS.WARNING_COLOR},
                    {'range': [RISK.MAJOR_LEAK_THRESHOLD, 1], 'color': COLORS.DANGER_COLOR}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': severity
                }
            }
        ))
        
        fig.update_layout(
            template=VIZ.PLOTLY_TEMPLATE,
            height=VIZ.DEFAULT_CHART_HEIGHT,
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': "white", 'family': "Arial"}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Leak details
        st.markdown("#### 📋 Leak Details")
        
        leak_type = result.leak_detection.leak_type
        leak_flow = result.leak_detection.estimated_flow_lps
        
        severity_label = get_severity_label(severity)
        
        st.markdown(f"""
        - **Type:** {leak_type}
        - **Status:** {severity_label}
        - **Flow Loss:** {leak_flow:.2f} L/s
        - **Daily Loss:** {leak_flow * 86.4:.1f} m³/day
        """)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: WATER QUALITY
# ═══════════════════════════════════════════════════════════════════════════

def render_water_quality(result):
    """Render water quality analysis with chlorine decay"""
    
    st.markdown("### 🧪 Water Quality Analysis")
    
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.markdown("#### 📉 Chlorine Decay Curve")
        
        # Chlorine decay: C(t) = C0 * e^(-kt)
        C0 = 0.8  # Initial chlorine (mg/L)
        k = 0.05  # Decay rate (1/hour)
        
        time_hours = np.linspace(0, 72, 200)
        chlorine_t = C0 * np.exp(-k * time_hours)
        
        # Create chlorine decay chart
        fig = go.Figure()
        
        # Decay curve
        fig.add_trace(go.Scatter(
            x=time_hours,
            y=chlorine_t,
            name='Chlorine Concentration',
            mode='lines',
            line=dict(color=COLORS.SECONDARY_COLOR, width=3),
            fill='tozeroy',
            fillcolor=f'rgba(6, 182, 212, 0.2)'
        ))
        
        # Minimum threshold line
        fig.add_hline(
            y=CONSTRAINTS.CHLORINE_MIN_THRESHOLD,
            line_dash="dash",
            line_color=COLORS.DANGER_COLOR,
            annotation_text=f"Min Threshold ({CONSTRAINTS.CHLORINE_MIN_THRESHOLD} mg/L)",
            annotation_position="right"
        )
        
        # Optimal level line
        fig.add_hline(
            y=CONSTRAINTS.CHLORINE_OPTIMAL,
            line_dash="dot",
            line_color=COLORS.SUCCESS_COLOR,
            annotation_text=f"Optimal ({CONSTRAINTS.CHLORINE_OPTIMAL} mg/L)",
            annotation_position="right"
        )
        
        # Current residual marker
        current_chlorine = result.water_quality.chlorine_residual_mg_l
        avg_age = result.water_quality.avg_age_hours
        
        fig.add_trace(go.Scatter(
            x=[avg_age],
            y=[current_chlorine],
            name='Current State',
            mode='markers',
            marker=dict(
                size=15,
                color=COLORS.WARNING_COLOR,
                symbol='diamond',
                line=dict(width=2, color='white')
            ),
            text=[f"Current: {current_chlorine:.2f} mg/L at {avg_age:.1f}h"],
            hoverinfo='text'
        ))
        
        # Add decay equation annotation
        fig.add_annotation(
            x=40, y=0.6,
            text=r"$C(t) = C_0 \cdot e^{-kt}$",
            showarrow=False,
            font=dict(size=16, color='white', family='Arial'),
            bgcolor=COLORS.PRIMARY_COLOR,
            bordercolor=COLORS.SECONDARY_COLOR,
            borderwidth=2,
            borderpad=10
        )
        
        fig.update_layout(
            title="Chlorine Residual Decay Over Time",
            xaxis_title="Time (hours)",
            yaxis_title="Chlorine Concentration (mg/L)",
            template=VIZ.PLOTLY_TEMPLATE,
            height=VIZ.DEFAULT_CHART_HEIGHT,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        st.markdown("#### 📊 Quality Metrics")
        
        quality_standard = result.water_quality.quality_standard
        avg_age = result.water_quality.avg_age_hours
        
        # Quality indicators
        st.metric(
            "Quality Standard",
            quality_standard,
            help="Kazakhstan GOST water quality classification"
        )
        
        st.metric(
            "Average Water Age",
            f"{avg_age:.1f} hours",
            delta=f"{avg_age - CONSTRAINTS.MAX_WATER_AGE_HOURS:.1f}h vs limit",
            delta_color="inverse"
        )
        
        st.metric(
            "Chlorine Residual",
            f"{result.water_quality.chlorine_residual_mg_l:.3f} mg/L",
            help=f"Range: {CONSTRAINTS.CHLORINE_MIN_THRESHOLD}-{CONSTRAINTS.CHLORINE_MAX_THRESHOLD} mg/L"
        )
        
        # Compliance status
        chlorine_ok = CONSTRAINTS.CHLORINE_MIN_THRESHOLD <= current_chlorine <= CONSTRAINTS.CHLORINE_MAX_THRESHOLD
        age_ok = avg_age <= CONSTRAINTS.MAX_WATER_AGE_HOURS
        
        if chlorine_ok and age_ok:
            st.success("✅ All quality parameters within acceptable ranges")
        else:
            if not chlorine_ok:
                st.error("❌ Chlorine residual out of compliance range")
            if not age_ok:
                st.warning("⚠️ Water age exceeds recommended maximum")
    
    # Stagnation zones table
    st.markdown("---")
    st.markdown("#### 🚰 Stagnation Zones")
    
    stagnation_zones = result.water_quality.stagnation_zones
    
    if stagnation_zones and len(stagnation_zones) > 0:
        df_stagnation = pd.DataFrame(stagnation_zones)
        
        # Add severity classification
        df_stagnation['severity'] = df_stagnation['water_age_hours'].apply(
            lambda x: '🔴 Critical' if x > 72 else ('🟠 High' if x > 48 else '🟡 Moderate')
        )
        
        st.dataframe(
            df_stagnation,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("✅ No significant stagnation zones detected in the network")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4: RISK & ROI
# ═══════════════════════════════════════════════════════════════════════════

def render_risk_roi(result):
    """Render risk assessment and maintenance roadmap"""
    
    st.markdown("### ⚠️ Risk Assessment & Maintenance Roadmap")
    
    # Maintenance priorities table
    priorities = result.criticality_assessment.maintenance_priorities
    
    if priorities and len(priorities) > 0:
        df_priorities = pd.DataFrame(priorities)
        
        # Add risk class
        df_priorities['risk_class'] = df_priorities['criticality_index'].apply(
            lambda x: RISK.get_risk_class(x)
        )
        
        # Add color indicator
        df_priorities['status_badge'] = df_priorities['risk_class'].apply(
            lambda x: f'🔴 {x}' if x == 'CRITICAL' else (
                f'🟠 {x}' if x == 'HIGH' else (
                    f'🟡 {x}' if x == 'MEDIUM' else f'🟢 {x}'
                )
            )
        )
        
        # Sort by criticality
        df_priorities = df_priorities.sort_values('criticality_index', ascending=False)
        
        st.markdown("#### 🛠️ Maintenance Priority Queue")
        
        # Style the dataframe
        def highlight_critical(row):
            if row['risk_class'] == 'CRITICAL':
                return [f'background-color: {COLORS.DANGER_COLOR}40'] * len(row)
            elif row['risk_class'] == 'HIGH':
                return [f'background-color: {COLORS.WARNING_COLOR}30'] * len(row)
            else:
                return [''] * len(row)
        
        styled_df = df_priorities.style.apply(highlight_critical, axis=1)
        
        st.dataframe(
            df_priorities[['node', 'criticality_index', 'risk_class', 'status_badge']],
            use_container_width=True,
            hide_index=True,
            column_config={
                'node': 'Node ID',
                'criticality_index': st.column_config.ProgressColumn(
                    'Criticality Index',
                    format="%.3f",
                    min_value=0,
                    max_value=1
                ),
                'risk_class': 'Risk Level',
                'status_badge': 'Status'
            }
        )
        
        # Risk distribution chart
        st.markdown("---")
        st.markdown("#### 📊 Risk Distribution")
        
        risk_counts = df_priorities['risk_class'].value_counts()
        
        fig = go.Figure(data=[
            go.Pie(
                labels=risk_counts.index,
                values=risk_counts.values,
                marker=dict(
                    colors=[
                        RISK.get_risk_color(label) for label in risk_counts.index
                    ],
                    line=dict(color='white', width=2)
                ),
                textinfo='label+percent',
                textfont=dict(size=14, color='white', family='Arial Black'),
                hole=0.4
            )
        ])
        
        fig.update_layout(
            title="Network Risk Distribution",
            template=VIZ.PLOTLY_TEMPLATE,
            height=VIZ.SMALL_CHART_HEIGHT,
            showlegend=True,
            annotations=[dict(
                text=f'{len(priorities)}<br>Nodes',
                x=0.5, y=0.5,
                font_size=20,
                showarrow=False,
                font_color='white'
            )]
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("No critical maintenance priorities identified")
    
    # Critical nodes highlight
    critical_nodes = [p for p in priorities if p['criticality_index'] >= RISK.CRITICAL_THRESHOLD]
    
    if critical_nodes:
        st.markdown("---")
        st.markdown(f"""
        <div class="alert-critical">
            <h4>🚨 CRITICAL ATTENTION REQUIRED</h4>
            <p><b>{len(critical_nodes)} node(s)</b> require immediate maintenance intervention:</p>
            <ul>
                {''.join([f"<li>Node {n['node']}: Criticality {n['criticality_index']:.3f}</li>" for n in critical_nodes[:5]])}
            </ul>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# ALERTS & RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════

def render_alerts_recommendations(result):
    """Render system alerts and recommendations"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🚨 Active Alerts")
        
        alerts = result.alerts
        
        if alerts and len(alerts) > 0:
            for alert in alerts:
                level = alert.get('level', 'INFO').upper()
                message = alert.get('message', '')
                node = alert.get('node', 'N/A')
                
                if level == 'CRITICAL':
                    alert_class = 'alert-critical'
                    emoji = '🚨'
                elif level == 'HIGH' or level == 'WARNING':
                    alert_class = 'alert-warning'
                    emoji = '⚠️'
                else:
                    alert_class = 'alert-success'
                    emoji = 'ℹ️'
                
                st.markdown(f"""
                <div class="{alert_class}">
                    <b>{emoji} {level}</b> - Node {node}<br>
                    {message}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ No active alerts - System operating normally")
    
    with col2:
        st.markdown("### 💡 Recommendations")
        
        recommendations = result.recommendations
        
        if recommendations and len(recommendations) > 0:
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"""
                <div style="background: rgba(6, 182, 212, 0.1); padding: 0.75rem; border-radius: 8px; margin-bottom: 0.5rem; border-left: 3px solid {COLORS.SECONDARY_COLOR};">
                    <b>{i}.</b> {rec}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No specific recommendations at this time")


# ═══════════════════════════════════════════════════════════════════════════
# ERROR HANDLING
# ═══════════════════════════════════════════════════════════════════════════

def render_error(error_message: str):
    """Render professional error message"""
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {COLORS.DANGER_COLOR}20 0%, {COLORS.DANGER_COLOR}10 100%);
        border: 2px solid {COLORS.DANGER_COLOR};
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
    ">
        <h2 style="color: {COLORS.DANGER_COLOR}; margin: 0;">❌ Analysis Error</h2>
        <p style="color: white; margin-top: 1rem; font-size: 1.1rem;">{error_message}</p>
        <small style="color: {COLORS.TEXT_SECONDARY};">Please check your input parameters and try again</small>
    </div>
    """, unsafe_allow_html=True)


def run_simulation(params: Dict):
    """Execute digital twin simulation with error handling"""
    try:
        with st.spinner('🔄 Running Digital Twin Analysis...'):
            # Initialize engine
            twin = DigitalTwinEngine(
                city=params['city'],
                season_temp_celsius=params['soil_temp'],
                material=params['pipe_material'],
                pipe_age=params['pipe_age']
            )
            
            # Run complete analysis
            result = twin.run_complete_analysis(
                grid_size=params['grid_size'],
                leak_node=params['leak_node'],
                leak_area_cm2=params['leak_area'],
                n_sensors=params['n_sensors']
            )
            
            # Store in session state
            st.session_state.simulation_result = result
            st.session_state.simulation_timestamp = params['timestamp']
            st.session_state.simulation_params = params
            
            # Add to history
            if len(st.session_state.history) >= 10:
                st.session_state.history.pop(0)
            st.session_state.history.append({
                'timestamp': params['timestamp'],
                'city': params['city'],
                'status': result.status
            })
            
            return result
            
    except Exception as e:
        st.error(f"Simulation failed: {str(e)}")
        import traceback
        with st.expander("🔍 Debug Information"):
            st.code(traceback.format_exc())
        return None


# ═══════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Main application entry point"""
    
    # Initialize
    initialize_session_state()
    inject_custom_css()
    
    # Header
    render_header()
    
    # Sidebar
    params = render_sidebar()
    
    # Run simulation if button clicked
    if params:
        result = run_simulation(params)
    else:
        result = st.session_state.simulation_result
    
    # Main content
    if result is None:
        # Welcome screen
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, rgba(30, 58, 138, 0.3) 0%, rgba(6, 182, 212, 0.2) 100%);
            padding: 3rem;
            border-radius: 16px;
            text-align: center;
            margin: 2rem 0;
            border: 1px solid {COLORS.SECONDARY_COLOR}40;
        ">
            <h2 style="color: white; margin-bottom: 1rem;">👋 Welcome to Smart Shygyn</h2>
            <p style="color: {COLORS.TEXT_SECONDARY}; font-size: 1.2rem; margin-bottom: 2rem;">
                Digital Twin for Kazakhstan's Water Distribution Networks
            </p>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 2rem;">
                <div style="background: rgba(255,255,255,0.05); padding: 1.5rem; border-radius: 8px;">
                    <div style="font-size: 2rem;">🗺️</div>
                    <div style="color: white; font-weight: 600; margin-top: 0.5rem;">Hydraulic Modeling</div>
                    <div style="color: {COLORS.TEXT_SECONDARY}; font-size: 0.9rem;">Real-time network simulation</div>
                </div>
                <div style="background: rgba(255,255,255,0.05); padding: 1.5rem; border-radius: 8px;">
                    <div style="font-size: 2rem;">💧</div>
                    <div style="color: white; font-weight: 600; margin-top: 0.5rem;">Leak Detection</div>
                    <div style="color: {COLORS.TEXT_SECONDARY}; font-size: 0.9rem;">AI-powered anomaly detection</div>
                </div>
                <div style="background: rgba(255,255,255,0.05); padding: 1.5rem; border-radius: 8px;">
                    <div style="font-size: 2rem;">🧪</div>
                    <div style="color: white; font-weight: 600; margin-top: 0.5rem;">Water Quality</div>
                    <div style="color: {COLORS.TEXT_SECONDARY}; font-size: 0.9rem;">GOST compliance monitoring</div>
                </div>
                <div style="background: rgba(255,255,255,0.05); padding: 1.5rem; border-radius: 8px;">
                    <div style="font-size: 2rem;">⚠️</div>
                    <div style="color: white; font-weight: 600; margin-top: 0.5rem;">Risk Analytics</div>
                    <div style="color: {COLORS.TEXT_SECONDARY}; font-size: 0.9rem;">Predictive maintenance</div>
                </div>
            </div>
            <div style="margin-top: 2rem; padding: 1rem; background: rgba(6, 182, 212, 0.1); border-radius: 8px; border-left: 4px solid {COLORS.SECONDARY_COLOR};">
                <p style="color: white; margin: 0;">
                    <b>Getting Started:</b> Configure your network parameters in the sidebar and click 
                    <b>"🚀 Run Digital Twin Analysis"</b> to begin
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show simulation history if available
        if st.session_state.history and len(st.session_state.history) > 0:
            st.markdown("### 📊 Recent Simulations")
            history_df = pd.DataFrame(st.session_state.history)
            st.dataframe(history_df, use_container_width=True, hide_index=True)
        
        return
    
    # Check for errors
    if result.status == "ERROR":
        render_error(f"Analysis returned error status. Please review input parameters.")
        return
    
    # Success - render metrics
    st.markdown("---")
    render_metrics_row(result)
    
    # Tabs
    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🗺️ Hydraulic Map",
        "💧 Leak Analytics", 
        "🧪 Water Quality",
        "⚠️ Risk & ROI"
    ])
    
    with tab1:
        render_geo_network_map(
            result,
            st.session_state.simulation_params['grid_size'],
            st.session_state.simulation_params['leak_node'],
            st.session_state.simulation_params['city']
        )
        st.markdown("---")
        render_hydraulic_map(
            result,
            st.session_state.simulation_params['grid_size'],
            st.session_state.simulation_params['leak_node']
        )
    
    with tab2:
        render_leak_analytics(result)
    
    with tab3:
        render_water_quality(result)
    
    with tab4:
        render_risk_roi(result)
    
    # Alerts and recommendations
    st.markdown("---")
    render_alerts_recommendations(result)
    
    # Footer with simulation info
    st.markdown("---")
    if st.session_state.simulation_timestamp:
        timestamp = st.session_state.simulation_timestamp.strftime("%Y-%m-%d %H:%M:%S")
        params = st.session_state.simulation_params
        
        st.markdown(f"""
        <div style="text-align: center; color: {COLORS.TEXT_SECONDARY}; padding: 1rem;">
            <small>
                <b>Last Analysis:</b> {timestamp} | 
                <b>City:</b> {params['city']} | 
                <b>Grid:</b> {params['grid_size']}x{params['grid_size']} | 
                <b>Sensors:</b> {params['n_sensors']} |
                <b>Status:</b> {result.status}
            </small>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# APPLICATION ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
