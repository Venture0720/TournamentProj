"""
Smart Shygyn - Digital Twin Mission Control Dashboard
Sophisticated Streamlit UI for Kazakhstan's Water Networks
Astana Hub Competition 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
from typing import Optional, Dict, List

# Backend imports
from risk_engine import DigitalTwinEngine, DigitalTwinAPIResponse
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
    """Inject professional CSS styling"""
    st.markdown(f"""
    <style>
        /* Main container */
        .stApp {{
            background: linear-gradient(135deg, {COLORS.BACKGROUND_LIGHT} 0%, #E0E7FF 100%);
        }}
        
        /* Headers */
        h1, h2, h3 {{
            color: {COLORS.PRIMARY_COLOR};
            font-weight: 700;
        }}
        
        /* Metric cards */
        [data-testid="stMetricValue"] {{
            font-size: 2rem;
            color: {COLORS.PRIMARY_COLOR};
        }}
        
        /* Buttons */
        .stButton>button {{
            background: {COLORS.GRADIENT_PRIMARY};
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 2rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }}
        
        .stButton>button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(30, 58, 138, 0.3);
        }}
        
        /* Sidebar */
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {COLORS.PRIMARY_COLOR} 0%, #1E293B 100%);
        }}
        
        [data-testid="stSidebar"] * {{
            color: white !important;
        }}
        
        /* Alerts */
        .stAlert {{
            border-radius: 8px;
            padding: 1rem;
        }}
    </style>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# STATE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'simulation_result' not in st.session_state:
        st.session_state.simulation_result = None
    if 'simulation_params' not in st.session_state:
        st.session_state.simulation_params = None


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
    """
    Render sidebar with simulation parameters.
    
    Returns:
        Dict with keys: city, grid_size, leak_node, leak_area_cm2, 
                       material, pipe_age_years, soil_temp_celsius, sensors
        or None if validation fails
    """
    st.sidebar.header("🎛️ Mission Control")
    
    # City selection
    city = st.sidebar.selectbox(
        "📍 Select City",
        options=CITIES.get_city_list(),
        help="Choose Kazakhstan city for simulation"
    )
    
    # Grid size
    grid_size = st.sidebar.slider(
        "🔲 Grid Size",
        min_value=CONSTRAINTS.MIN_GRID_SIZE,
        max_value=CONSTRAINTS.MAX_GRID_SIZE,
        value=CONSTRAINTS.DEFAULT_GRID_SIZE,
        help="Network grid dimensions (NxN)"
    )
    
    # Leak parameters
    st.sidebar.subheader("💧 Leak Scenario")
    leak_node = st.sidebar.text_input(
        "Leak Node ID",
        value=f"N_{grid_size//2}_{grid_size//2}",
        help="Format: N_row_col (e.g., N_2_2)"
    )
    
    leak_area_cm2 = st.sidebar.slider(
        "Leak Area (cm²)",
        min_value=CONSTRAINTS.MIN_LEAK_AREA_CM2,
        max_value=CONSTRAINTS.MAX_LEAK_AREA_CM2,
        value=CONSTRAINTS.DEFAULT_LEAK_AREA_CM2,
        step=0.1
    )
    
    # Material and age
    st.sidebar.subheader("🔧 Infrastructure")
    material = st.sidebar.selectbox(
        "Pipe Material",
        options=["Пластик (ПНД)", "Сталь", "Чугун", "ПВХ", "Асбестоцемент"]
    )
    
    pipe_age_years = st.sidebar.slider(
        "Pipe Age (years)",
        min_value=0,
        max_value=100,
        value=20
    )
    
    # Environmental
    soil_temp_celsius = st.sidebar.slider(
        "Soil Temperature (°C)",
        min_value=CONSTRAINTS.MIN_SOIL_TEMP,
        max_value=CONSTRAINTS.MAX_SOIL_TEMP,
        value=CONSTRAINTS.DEFAULT_SOIL_TEMP,
        help="Affects freeze-thaw damage"
    )
    
    # Sensors
    sensors = st.sidebar.slider(
        "Number of Sensors",
        min_value=CONSTRAINTS.MIN_SENSORS,
        max_value=CONSTRAINTS.MAX_SENSORS,
        value=CONSTRAINTS.DEFAULT_SENSORS
    )
    
    # Run button
    if st.sidebar.button("🚀 Run Simulation", type="primary"):
        return {
            "city": city,
            "grid_size": grid_size,
            "leak_node": leak_node,
            "leak_area_cm2": leak_area_cm2,
            "material": material,
            "pipe_age_years": pipe_age_years,
            "soil_temp_celsius": soil_temp_celsius,
            "sensors": sensors
        }
    return None


# ═══════════════════════════════════════════════════════════════════════════
# METRIC CARDS
# ═══════════════════════════════════════════════════════════════════════════

def render_metrics_row(result: DigitalTwinAPIResponse):
    """Display key metrics in a row of colored cards"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if result.leak_detection:
            leak_status = "🚨 DETECTED" if result.leak_detection.leak_detected else "✅ NORMAL"
            severity = result.leak_detection.severity_score if result.leak_detection.leak_detected else None
        else:
            leak_status = "⚠️ NO DATA"
            severity = None
        
        st.metric(
            "Leak Status",
            leak_status,
            delta=f"{severity:.1f} severity" if severity else None,
            delta_color="inverse" if severity else "normal"
        )
    
    with col2:
        # Use material degradation data as proxy for pressure
        if result.material_degradation:
            degradation_pct = result.material_degradation.get('degradation_percentage', 0)
            # Estimate pressure based on degradation (inverse relationship)
            estimated_pressure = CONSTRAINTS.OPTIMAL_PRESSURE_BAR * (1 - degradation_pct / 100)
            st.metric(
                "Est. Pressure",
                f"{estimated_pressure:.2f} bar",
                delta=f"{estimated_pressure - CONSTRAINTS.OPTIMAL_PRESSURE_BAR:+.2f} bar"
            )
        else:
            st.metric("Est. Pressure", "N/A")
    
    with col3:
        if result.water_quality:
            quality_emoji = {
                "EXCELLENT": "🌟",
                "GOOD": "✅",
                "ACCEPTABLE": "⚠️",
                "MARGINAL": "🔶",
                "POOR": "🚨"
            }.get(result.water_quality.quality_standard, "❓")
            
            st.metric(
                "Water Quality",
                f"{quality_emoji} {result.water_quality.quality_standard}",
                delta=f"{result.water_quality.avg_age_hours:.1f}h age"
            )
        else:
            st.metric("Water Quality", "⚠️ NO DATA")
    
    with col4:
        # Calculate economic loss based on leak flow rate
        if result.leak_detection and result.leak_detection.leak_detected:
            # Estimate: 1 L/s = ~86,400 L/day, water cost ~100 KZT/m³
            daily_loss_m3 = result.leak_detection.estimated_flow_lps * 86.4
            loss_kzt = daily_loss_m3 * 100
            st.metric(
                "Economic Loss",
                f"{loss_kzt:,.0f} ₸",
                delta="Per day"
            )
        else:
            st.metric("Economic Loss", "0 ₸", delta="No leaks")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: HYDRAULIC NETWORK MAP
# ═══════════════════════════════════════════════════════════════════════════

def render_hydraulic_map(result: DigitalTwinAPIResponse, grid_size: int, leak_node: str):
    """Render interactive Plotly network map"""
    st.subheader("🗺️ Hydraulic Network Visualization")
    
    # Extract node data
    nodes_df = pd.DataFrame(result.topology.nodes)
    pipes_df = pd.DataFrame(result.topology.pipes)
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Add pipes as lines
    for _, pipe in pipes_df.iterrows():
        start = nodes_df[nodes_df['node_id'] == pipe['start_node']].iloc[0]
        end = nodes_df[nodes_df['node_id'] == pipe['end_node']].iloc[0]
        
        fig.add_trace(go.Scatter(
            x=[start['x'], end['x']],
            y=[start['y'], end['y']],
            mode='lines',
            line=dict(color=COLORS.TEXT_SECONDARY, width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add nodes colored by pressure
    fig.add_trace(go.Scatter(
        x=nodes_df['x'],
        y=nodes_df['y'],
        mode='markers+text',
        marker=dict(
            size=15,
            color=nodes_df['pressure_bar'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Pressure (bar)")
        ),
        text=nodes_df['node_id'],
        textposition='top center',
        hovertemplate='<b>%{text}</b><br>Pressure: %{marker.color:.2f} bar<extra></extra>'
    ))
    
    # Highlight leak node
    leak_node_data = nodes_df[nodes_df['node_id'] == leak_node]
    if not leak_node_data.empty:
        fig.add_trace(go.Scatter(
            x=leak_node_data['x'],
            y=leak_node_data['y'],
            mode='markers',
            marker=dict(size=25, color=COLORS.DANGER_COLOR, symbol='x'),
            name='Leak Location',
            hovertemplate='<b>LEAK</b><extra></extra>'
        ))
    
    fig.update_layout(
        title="Network Topology with Pressure Distribution",
        xaxis_title="X Coordinate (m)",
        yaxis_title="Y Coordinate (m)",
        height=600,
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: LEAK ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════

def render_leak_analytics(result: DigitalTwinAPIResponse):
    """Render leak detection analytics"""
    st.subheader("🔍 Leak Detection & Analysis")
    
    if not result.leak_detection:
        st.warning("⚠️ Leak detection data not available")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Detection Summary")
        st.markdown(f"**Status:** {get_status_emoji('ERROR' if result.leak_detection.leak_detected else 'SUCCESS')} "
                   f"{'LEAK DETECTED' if result.leak_detection.leak_detected else 'NO LEAK'}")
        st.markdown(f"**Type:** {result.leak_detection.leak_type}")
        st.markdown(f"**Severity:** {result.leak_detection.severity_score:.1f}/100")
        st.markdown(f"**Confidence:** {result.leak_detection.confidence*100:.1f}%")
        st.markdown(f"**Flow Rate:** {result.leak_detection.estimated_flow_lps:.2f} L/s")
    
    with col2:
        # Severity gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result.leak_detection.severity_score,
            title={'text': "Severity Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': COLORS.DANGER_COLOR},
                'steps': [
                    {'range': [0, 25], 'color': COLORS.SUCCESS_COLOR},
                    {'range': [25, 50], 'color': COLORS.WARNING_COLOR},
                    {'range': [50, 75], 'color': 'orange'},
                    {'range': [75, 100], 'color': COLORS.DANGER_COLOR}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 75
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: WATER QUALITY
# ═══════════════════════════════════════════════════════════════════════════

def render_water_quality(result):
    """Render water quality analysis with chlorine decay"""
    
    st.markdown("### 🧪 Water Quality Analysis")
    
    if not result.water_quality:
        st.warning("⚠️ Water quality data not available")
        return
    
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
    
    if not result.criticality_assessment:
        st.warning("⚠️ Criticality assessment data not available")
        return
    
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
    """Display user-friendly error message"""
    st.error(f"❌ **Simulation Error**\n\n{error_message}")
    
    with st.expander("🔧 Troubleshooting"):
        st.markdown("""
        **Common Issues:**
        - Invalid leak node ID (must match format: N_row_col)
        - Grid size mismatch with leak node
        - Sensor count exceeds network size
        - Material parameters incompatible with pipe age
        
        **Solution:** Adjust parameters in sidebar and retry.
        """)


def run_simulation(params: Dict) -> DigitalTwinAPIResponse:
    """
    Execute digital twin simulation with error handling.
    
    Args:
        params: Dictionary from render_sidebar()
    
    Returns:
        DigitalTwinAPIResponse object
    
    Raises:
        Exception: If simulation fails
    """
    try:
        # Initialize engine
        twin = DigitalTwinEngine(
            city=params["city"],
            season_temp_celsius=params["soil_temp_celsius"],
            material=params["material"],
            pipe_age=params["pipe_age_years"]
        )
        
        # Run complete analysis
        result = twin.run_complete_analysis(
            grid_size=params["grid_size"],
            leak_node=params["leak_node"],
            leak_area_cm2=params["leak_area_cm2"],
            n_sensors=params["sensors"]
        )
        
        return result
        
    except Exception as e:
        st.error(f"Simulation failed: {str(e)}")
        raise


# ═══════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Main application entry point"""
    
    # Inject CSS
    inject_custom_css()
    
    # Initialize state
    initialize_session_state()
    
    # Header
    render_header()
    
    # Sidebar
    params = render_sidebar()
    
    # Run simulation if parameters provided
    if params:
        with st.spinner("🔄 Running Digital Twin Simulation..."):
            result = run_simulation(params)
            st.session_state.simulation_result = result
            st.session_state.simulation_params = params
    
    # Display results if available
    if st.session_state.simulation_result:
        result = st.session_state.simulation_result
        params = st.session_state.simulation_params
        
        # Metrics row
        render_metrics_row(result)
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "🗺️ Hydraulic Map",
            "🔍 Leak Analytics", 
            "💧 Water Quality",
            "📊 Risk & ROI"
        ])
        
        with tab1:
            render_hydraulic_map(result, params["grid_size"], params["leak_node"])
        
        with tab2:
            render_leak_analytics(result)
        
        with tab3:
            render_water_quality(result)
        
        with tab4:
            render_risk_roi(result)
        
        # Alerts section
        render_alerts_recommendations(result)

if __name__ == "__main__":
    main()
