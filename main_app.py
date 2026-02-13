"""
Smart Water Management Digital Twin - Streamlit Frontend
=========================================================
Main orchestrator for 3 backend modules.
Target: Astana Hub Competition
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import traceback
import warnings
import sys
import os

# ПРИНУДИТЕЛЬНОЕ ДОБАВЛЕНИЕ ПУТИ (Чтобы Python видел соседние файлы)
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Suppress warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Smart Shygyn Digital Twin",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional appearance
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1E88E5; text-align: center; padding: 1rem 0; }
    .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; color: white; text-align: center; margin: 0.5rem 0; }
    .stTabs [data-baseweb="tab-list"] { gap: 2rem; }
    .stTabs [data-baseweb="tab"] { padding: 1rem 2rem; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# BACKEND MODULE IMPORTS WITH ERROR HANDLING
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_backend_modules():
    modules = {}
    errors = []
    
    # Пытаемся импортировать каждый модуль отдельно
    try:
        from hydraulic_intelligence import HydraulicIntelligenceEngine
        modules['hydraulic'] = HydraulicIntelligenceEngine
    except Exception as e:
        errors.append(f"❌ hydraulic_intelligence.py: {str(e)}")
    
    try:
        from leak_analytics import LeakAnalyticsEngine
        modules['leak'] = LeakAnalyticsEngine
    except Exception as e:
        errors.append(f"❌ leak_analytics.py: {str(e)}")
    
    try:
        from risk_engine import DigitalTwinEngine, WaterAgeAnalyzer, CriticalityIndexCalculator
        modules['risk'] = DigitalTwinEngine
        modules['water_age'] = WaterAgeAnalyzer
        modules['criticality'] = CriticalityIndexCalculator
    except Exception as e:
        errors.append(f"❌ risk_engine.py: {str(e)}")
    
    if errors:
        return False, modules, "\n".join(errors)
    
    return True, modules, None

# Load modules at startup
with st.spinner("🚀 Loading backend modules..."):
    success, MODULES, error_msg = load_backend_modules()

if not success:
    st.error("### ⚠️ Backend Module Loading Failed")
    st.info(f"**Detected Errors:**\n{error_msg}")
    st.warning("Ensure all files are in the same folder and libraries (wntr, networkx) are installed.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════

if 'simulation_run' not in st.session_state:
    st.session_state.simulation_run = False
if 'api_response' not in st.session_state:
    st.session_state.api_response = None

# ═══════════════════════════════════════════════════════════════════════════
# MAIN INTERFACE
# ═══════════════════════════════════════════════════════════════════════════

st.markdown('<div class="main-header">Smart Shygyn: Digital Twin System</div>', unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.header("📍 Network Configuration")
    city = st.selectbox("Select City", ["Astana", "Almaty", "Turkestan"])
    grid_size = st.slider("Grid Complexity (Nodes)", 3, 6, 4)
    material = st.selectbox("Pipe Material", ["Steel", "Cast Iron", "HDPE", "PVC"])
    
    st.divider()
    
    st.header("🚨 Simulation Stress-Test")
    leak_mode = st.checkbox("Simulate Emergency (Leak)")
    if leak_mode:
        leak_type = st.select_slider("Leak Severity", options=["Small", "Burst", "Catastrophic"])
    
    run_btn = st.button("🚀 RUN SIMULATION", use_container_width=True, type="primary")

# Main Dashboard logic
if run_btn:
    with st.spinner("Calculating Hydraulics & Risk Metrics..."):
        # Здесь вызывается логика из твоих backend-файлов
        # Для примера создаем флаг запуска
        st.session_state.simulation_run = True
        st.success(f"Simulation for {city} completed successfully!")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Live Monitoring", "🛡️ Risk Assessment", "💰 Economic Impact"])

with tab1:
    if st.session_state.simulation_run:
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Pressure", "3.2 bar", "+0.2")
        col2.metric("Water Loss", "1.2 L/s", "-5%", delta_color="inverse")
        col3.metric("System Health", "94%", "Stable")
        
        st.subheader("Network Hydraulic Profile")
        # Тут будет твой Plotly график из NetworkX
        st.info("Interactive Graph Loading...")
    else:
        st.info("Please run the simulation from the sidebar to see results.")

with tab2:
    st.subheader("Criticality Index & Maintenance Priority")
    st.write("Analysis of failure probability vs social impact.")

with tab3:
    st.subheader("ROI & Energy Savings")
    st.write("Financial metrics based on water loss reduction.")
