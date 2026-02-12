"""
Smart Water Management Digital Twin - Streamlit Frontend
=========================================================

Main orchestrator for 3 backend modules:
- hydraulic_intelligence.py (HydraulicIntelligenceEngine)
- leak_analytics.py (LeakAnalyticsEngine)
- risk_engine.py (DigitalTwinEngine)

Author: Principal Software Engineer
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

# Suppress warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Smart Shygyn Digital Twin",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Smart Water Management Digital Twin for Astana Hub Competition"
    }
)

# Custom CSS for professional appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .alert-critical {
        background-color: #f44336;
        color: white;
        padding: 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .alert-warning {
        background-color: #ff9800;
        color: white;
        padding: 1rem;
        border-radius: 5px;
    }
    .alert-success {
        background-color: #4caf50;
        color: white;
        padding: 1rem;
        border-radius: 5px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# BACKEND MODULE IMPORTS WITH ERROR HANDLING
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_backend_modules():
    """
    Load backend modules with comprehensive error handling.
    Uses @st.cache_resource to prevent reloading on every interaction.
    
    Returns:
        tuple: (success, modules_dict, error_message)
    """
    modules = {}
    errors = []
    
    try:
        from hydraulic_intelligence import HydraulicIntelligenceEngine
        modules['hydraulic'] = HydraulicIntelligenceEngine
    except ImportError as e:
        errors.append(f"❌ hydraulic_intelligence.py: {str(e)}")
    except Exception as e:
        errors.append(f"❌ hydraulic_intelligence.py: {str(e)}")
    
    try:
        from leak_analytics import LeakAnalyticsEngine
        modules['leak'] = LeakAnalyticsEngine
    except ImportError as e:
        errors.append(f"❌ leak_analytics.py: {str(e)}")
    except Exception as e:
        errors.append(f"❌ leak_analytics.py: {str(e)}")
    
    try:
        from risk_engine import DigitalTwinEngine, WaterAgeAnalyzer, CriticalityIndexCalculator
        modules['risk'] = DigitalTwinEngine
        modules['water_age'] = WaterAgeAnalyzer
        modules['criticality'] = CriticalityIndexCalculator
    except ImportError as e:
        errors.append(f"❌ risk_engine.py: {str(e)}")
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
    st.error(error_msg)
    st.info("**Please ensure all 3 backend files are in the same directory:**")
    st.code("""
    - hydraulic_intelligence.py
    - leak_analytics.py
    - risk_engine.py
    - main_app.py (this file)
    """)
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════

def initialize_session_state():
    """Initialize all session state variables."""
    
    if 'simulation_run' not in st.session_state:
        st.session_state.simulation_run = False
    
    if 'simulation_results' not in st.session_state:
        st.session_state.simulation_results = None
    
    if 'api_response' not in st.session_state:
        st.session_state.api_response = None
    
    if 'network_data' not in st.session_state:
        st.session_state.network_data = None
    
    if 'leak_data' not in st.session_state:
        st.session_state.leak_data = None
    
    if 'risk_data' not in st.session_state:
        st.session_state.risk_data = None
    
    if 'simulation_config' not in st.session_state:
        st.session_state.simulation_config = {}

initialize_session_state()


