"""
Smart Shygyn PRO v3 — Professional Config Engine
Complete configuration with modern CSS injection, multi-city support, and production-ready constants.
Python 3.13+ compatible.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass(frozen=True)
class UIColors:
    """Professional color schemes for dark and light themes."""
    
    # Dark Theme (Cyberpunk Professional)
    DARK_BG: str = "#0e1117"
    DARK_CARD: str = "#1a1f2e"
    DARK_BORDER: str = "#2d3748"
    DARK_ACCENT: str = "#3b82f6"
    DARK_DANGER: str = "#ef4444"
    DARK_WARNING: str = "#f59e0b"
    DARK_SUCCESS: str = "#10b981"
    DARK_TEXT: str = "#e2e8f0"
    DARK_MUTED: str = "#94a3b8"
    
    # Light Theme (Professional Clean)
    LIGHT_BG: str = "#ffffff"
    LIGHT_CARD: str = "#f8f9fa"
    LIGHT_BORDER: str = "#e2e8f0"
    LIGHT_ACCENT: str = "#1f77b4"
    LIGHT_DANGER: str = "#dc3545"
    LIGHT_WARNING: str = "#fd7e14"
    LIGHT_SUCCESS: str = "#28a745"
    LIGHT_TEXT: str = "#2c3e50"
    LIGHT_MUTED: str = "#6c757d"


@dataclass(frozen=True)
class PhysicsConstants:
    """Hydraulic physics and simulation constants."""
    
    # Hazen-Williams roughness coefficients (Material: C-factor)
    ROUGHNESS_BASE: Dict[str, float] = field(default_factory=lambda: {
        "Пластик (ПНД)": 145.0,
        "Сталь": 120.0,
        "Чугун": 100.0
    })
    
    # Roughness degradation rates (Material: % per year)
    DEGRADATION_RATES: Dict[str, float] = field(default_factory=lambda: {
        "Пластик (ПНД)": 0.15,
        "Сталь": 0.80,
        "Чугун": 1.20
    })
    
    # Physical constants
    GRAVITY: float = 9.81  # m/s²
    WATER_DENSITY: float = 1000.0  # kg/m³
    BAR_TO_METER: float = 10.197  # 1 bar = 10.197 m of water
    METER_TO_BAR: float = 0.098  # 1 m = 0.098 bar
    
    # Leak detection thresholds
    PRESSURE_THRESHOLD_BAR: float = 2.7
    CONTAMINATION_THRESHOLD_BAR: float = 1.5
    MNF_ANOMALY_THRESHOLD_PCT: float = 15.0  # % above baseline


@dataclass(frozen=True)
class EconomicConstants:
    """Economic and financial parameters."""
    
    # Costs (KZT - Kazakhstani Tenge)
    SENSOR_UNIT_COST: float = 120_000.0  # per sensor
    INSTALLATION_COST_MULTIPLIER: float = 1.2  # 20% installation overhead
    DEFAULT_WATER_TARIFF: float = 0.55  # KZT per liter
    DEFAULT_REPAIR_COST: float = 50_000.0  # emergency repair deployment
    ENERGY_COST_PER_KWH: float = 18.5  # KZT per kWh
    
    # Environmental
    CO2_PER_KWH: float = 0.45  # kg CO₂ per kWh (Kazakhstan grid)
    
    # ROI thresholds
    ACCEPTABLE_PAYBACK_MONTHS: int = 24
    CRITICAL_PAYBACK_MONTHS: int = 36


@dataclass(frozen=True)
class SimulationDefaults:
    """Default simulation parameters."""
    
    GRID_SIZE: int = 4  # 4x4 network grid
    SIMULATION_DURATION_HOURS: int = 24
    TIMESTEP_SECONDS: int = 3600  # 1 hour
    
    # Pump defaults
    DEFAULT_PUMP_HEAD_M: int = 40
    SMART_PUMP_NIGHT_RATIO: float = 0.7  # 70% during night
    NIGHT_START_HOUR: int = 0
    NIGHT_END_HOUR: int = 6
    
    # Sensor configuration
    SENSOR_COVERAGE_RATIO: float = 0.30  # 30% of nodes
    DEFAULT_SAMPLING_RATE_HZ: int = 1
    
    # Leak defaults
    DEFAULT_LEAK_AREA_CM2: float = 0.8
    
    # Material defaults
    DEFAULT_MATERIAL: str = "Пластик (ПНД)"
    DEFAULT_PIPE_AGE: int = 15


@dataclass
class AppConfig:
    """
    Master application configuration.
    Provides complete CSS injection, constants, and utility methods.
    """
    
    # Sub-configurations
    colors: UIColors = field(default_factory=UIColors)
    physics: PhysicsConstants = field(default_factory=PhysicsConstants)
    economics: EconomicConstants = field(default_factory=EconomicConstants)
    simulation: SimulationDefaults = field(default_factory=SimulationDefaults)
    
    # Cache settings
    CACHE_TTL: int = 3600  # 1 hour
    
    def get_style(self, dark_mode: bool) -> str:
        """
        Generate complete CSS for Streamlit injection.
        Covers all UI elements with modern selectors.
        
        Args:
            dark_mode: If True, use dark theme; otherwise light theme
            
        Returns:
            Complete CSS string ready for st.markdown(unsafe_allow_html=True)
        """
        c = self.colors
        
        if dark_mode:
            bg = c.DARK_BG
            card = c.DARK_CARD
            border = c.DARK_BORDER
            accent = c.DARK_ACCENT
            danger = c.DARK_DANGER
            warning = c.DARK_WARNING
            success = c.DARK_SUCCESS
            text = c.DARK_TEXT
            muted = c.DARK_MUTED
        else:
            bg = c.LIGHT_BG
            card = c.LIGHT_CARD
            border = c.LIGHT_BORDER
            accent = c.LIGHT_ACCENT
            danger = c.LIGHT_DANGER
            warning = c.LIGHT_WARNING
            success = c.LIGHT_SUCCESS
            text = c.LIGHT_TEXT
            muted = c.LIGHT_MUTED
        
        return f"""
        <style>
            /* ============================================================
               ROOT VARIABLES
               ============================================================ */
            :root {{
                --bg: {bg};
                --card: {card};
                --border: {border};
                --accent: {accent};
                --danger: {danger};
                --warning: {warning};
                --success: {success};
                --text: {text};
                --muted: {muted};
            }}
            
            /* ============================================================
               MAIN CONTAINERS
               ============================================================ */
            [data-testid="stAppViewContainer"] {{
                background-color: var(--bg);
                color: var(--text);
            }}
            
            [data-testid="stSidebar"] {{
                background-color: var(--card);
                border-right: 2px solid var(--border);
            }}
            
            [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{
                color: var(--text);
            }}
            
            [data-testid="stHeader"] {{
                background-color: var(--bg);
                border-bottom: 1px solid var(--border);
            }}
            
            /* ============================================================
               METRICS
               ============================================================ */
            [data-testid="stMetric"] {{
                background: var(--card);
                padding: 16px;
                border-radius: 8px;
                border: 1px solid var(--border);
                backdrop-filter: blur(10px);
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }}
            
            [data-testid="stMetricValue"] {{
                font-size: 24px;
                font-weight: 700;
                color: var(--text);
            }}
            
            [data-testid="stMetricLabel"] {{
                font-size: 12px;
                color: var(--muted);
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            
            [data-testid="stMetricDelta"] {{
                font-size: 14px;
            }}
            
            /* ============================================================
               TYPOGRAPHY
               ============================================================ */
            h1 {{
                color: var(--accent) !important;
                text-align: center;
                padding: 16px 0;
                letter-spacing: 1px;
                border-bottom: 3px solid var(--accent);
                margin-bottom: 24px;
                font-family: 'Segoe UI', 'Roboto', sans-serif;
            }}
            
            h2 {{
                color: var(--text) !important;
                border-left: 4px solid var(--accent);
                padding-left: 12px;
                margin-top: 24px;
                font-family: 'Segoe UI', 'Roboto', sans-serif;
            }}
            
            h3 {{
                color: var(--text) !important;
                border-bottom: 2px solid var(--accent);
                padding-bottom: 8px;
                margin-top: 16px;
                font-family: 'Segoe UI', 'Roboto', sans-serif;
            }}
            
            h4, h5, h6 {{
                color: var(--text) !important;
                font-family: 'Segoe UI', 'Roboto', sans-serif;
            }}
            
            p, li, span {{
                color: var(--text);
            }}
            
            .stCaption {{
                color: var(--muted) !important;
            }}
            
            /* ============================================================
               ALERTS & MESSAGES
               ============================================================ */
            .stAlert {{
                border-radius: 8px;
                border-left-width: 4px;
                background-color: var(--card);
                color: var(--text);
            }}
            
            [data-testid="stAlert"] {{
                background-color: var(--card);
            }}
            
            /* ============================================================
               TABS
               ============================================================ */
            .stTabs [data-baseweb="tab-list"] {{
                gap: 8px;
                background-color: var(--bg);
            }}
            
            .stTabs [data-baseweb="tab"] {{
                height: 45px;
                font-size: 14px;
                font-weight: 600;
                padding: 12px 24px;
                border-radius: 8px 8px 0 0;
                background-color: var(--card);
                color: var(--text);
                border: 1px solid var(--border);
            }}
            
            .stTabs [data-baseweb="tab"]:hover {{
                background-color: var(--border);
            }}
            
            .stTabs [aria-selected="true"] {{
                background-color: var(--accent) !important;
                color: white !important;
                border-color: var(--accent) !important;
            }}
            
            /* ============================================================
               BUTTONS
               ============================================================ */
            .stButton > button {{
                width: 100%;
                font-weight: 600;
                border-radius: 6px;
                background-color: var(--accent);
                color: white;
                border: none;
                padding: 10px 20px;
                transition: all 0.3s ease;
            }}
            
            .stButton > button:hover {{
                background-color: var(--accent);
                opacity: 0.85;
                border: none;
                transform: translateY(-1px);
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
            }}
            
            .stButton > button[kind="primary"] {{
                background-color: var(--success);
            }}
            
            .stButton > button[kind="primary"]:hover {{
                background-color: var(--success);
                opacity: 0.85;
            }}
            
            .stButton > button[kind="secondary"] {{
                background-color: var(--card);
                color: var(--text);
                border: 1px solid var(--border);
            }}
            
            /* ============================================================
               INPUT WIDGETS
               ============================================================ */
            .stTextInput > div > div > input,
            .stNumberInput > div > div > input,
            .stSelectbox > div > div,
            .stSlider > div > div > div {{
                background-color: var(--card);
                color: var(--text);
                border-color: var(--border);
            }}
            
            .stTextInput > div > div > input:focus,
            .stNumberInput > div > div > input:focus {{
                border-color: var(--accent);
                box-shadow: 0 0 0 1px var(--accent);
            }}
            
            /* Slider styling */
            .stSlider > div > div > div > div {{
                background-color: var(--accent);
            }}
            
            /* ============================================================
               EXPANDERS
               ============================================================ */
            .streamlit-expanderHeader {{
                font-weight: 600;
                font-size: 15px;
                color: var(--text);
                background-color: var(--card);
                border-radius: 6px;
                border: 1px solid var(--border);
            }}
            
            .streamlit-expanderHeader:hover {{
                background-color: var(--border);
            }}
            
            .streamlit-expanderContent {{
                background-color: var(--card);
                border: 1px solid var(--border);
                border-top: none;
                border-radius: 0 0 6px 6px;
            }}
            
            /* ============================================================
               DATAFRAMES & TABLES
               ============================================================ */
            [data-testid="stDataFrame"] {{
                background-color: var(--card);
                border: 1px solid var(--border);
                border-radius: 6px;
            }}
            
            [data-testid="stDataFrame"] table {{
                color: var(--text);
            }}
            
            [data-testid="stDataFrame"] th {{
                background-color: var(--accent);
                color: white;
                font-weight: 600;
            }}
            
            /* ============================================================
               CODE BLOCKS
               ============================================================ */
            .stCodeBlock {{
                background-color: var(--card) !important;
                color: var(--text) !important;
                border: 1px solid var(--border);
                border-radius: 6px;
            }}
            
            code {{
                background-color: var(--card);
                color: var(--text);
                padding: 2px 6px;
                border-radius: 4px;
            }}
            
            /* ============================================================
               PROGRESS & SPINNERS
               ============================================================ */
            .stProgress > div > div > div > div {{
                background-color: var(--accent);
            }}
            
            .stSpinner > div {{
                border-top-color: var(--accent) !important;
            }}
            
            /* ============================================================
               DIVIDERS
               ============================================================ */
            hr {{
                border-color: var(--border);
                margin: 24px 0;
            }}
            
            /* ============================================================
               MARKDOWN & TEXT
               ============================================================ */
            [data-testid="stMarkdownContainer"] p {{
                color: var(--text);
            }}
            
            [data-testid="stMarkdownContainer"] a {{
                color: var(--accent);
                text-decoration: none;
            }}
            
            [data-testid="stMarkdownContainer"] a:hover {{
                text-decoration: underline;
            }}
            
            /* ============================================================
               FORMS
               ============================================================ */
            [data-testid="stForm"] {{
                background-color: var(--card);
                border: 1px solid var(--border);
                border-radius: 8px;
                padding: 16px;
            }}
            
            /* ============================================================
               TOOLTIPS
               ============================================================ */
            [data-testid="stTooltipIcon"] {{
                color: var(--muted);
            }}
            
            /* ============================================================
               SCROLLBARS (Webkit browsers)
               ============================================================ */
            ::-webkit-scrollbar {{
                width: 10px;
                height: 10px;
            }}
            
            ::-webkit-scrollbar-track {{
                background: var(--bg);
            }}
            
            ::-webkit-scrollbar-thumb {{
                background: var(--border);
                border-radius: 5px;
            }}
            
            ::-webkit-scrollbar-thumb:hover {{
                background: var(--accent);
            }}
            
            /* ============================================================
               CUSTOM UTILITY CLASSES
               ============================================================ */
            .success-box {{
                background-color: var(--success);
                color: white;
                padding: 12px;
                border-radius: 6px;
                font-weight: 500;
            }}
            
            .warning-box {{
                background-color: var(--warning);
                color: white;
                padding: 12px;
                border-radius: 6px;
                font-weight: 500;
            }}
            
            .danger-box {{
                background-color: var(--danger);
                color: white;
                padding: 12px;
                border-radius: 6px;
                font-weight: 500;
            }}
        </style>
        """
    
    def get_map_tile(self, dark_mode: bool) -> str:
        """
        Get appropriate Folium map tile based on theme.
        
        Args:
            dark_mode: If True, return dark map tiles
            
        Returns:
            Map tile provider string
        """
        return "CartoDB dark_matter" if dark_mode else "OpenStreetMap"
    
    def get_plotly_template(self, dark_mode: bool) -> str:
        """
        Get Plotly template name based on theme.
        
        Args:
            dark_mode: If True, return dark template
            
        Returns:
            Plotly template name
        """
        return "plotly_dark" if dark_mode else "plotly"
    
    def get_color_palette(self, dark_mode: bool) -> Tuple[str, ...]:
        """
        Get color palette for charts and visualizations.
        
        Args:
            dark_mode: If True, return dark-theme colors
            
        Returns:
            Tuple of color hex codes
        """
        c = self.colors
        if dark_mode:
            return (
                c.DARK_ACCENT,
                c.DARK_SUCCESS,
                c.DARK_WARNING,
                c.DARK_DANGER,
                "#a855f7",  # Purple
                "#ec4899",  # Pink
            )
        else:
            return (
                c.LIGHT_ACCENT,
                c.LIGHT_SUCCESS,
                c.LIGHT_WARNING,
                c.LIGHT_DANGER,
                "#7c3aed",  # Purple
                "#db2777",  # Pink
            )


# ============================================================
# GLOBAL SINGLETON INSTANCE
# ============================================================
CONFIG = AppConfig()


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def get_roughness(material: str, age_years: float) -> float:
    """
    Calculate Hazen-Williams roughness with age degradation.
    
    Args:
        material: Pipe material name
        age_years: Age of pipe in years
        
    Returns:
        Degraded roughness coefficient
    """
    base_roughness = CONFIG.physics.ROUGHNESS_BASE.get(material, 120.0)
    degradation_rate = CONFIG.physics.DEGRADATION_RATES.get(material, 0.5)
    
    # Apply degradation: C_degraded = C_base * (1 - rate * age / 100)
    degradation_factor = 1.0 - (degradation_rate * age_years / 100.0)
    degraded_roughness = base_roughness * max(0.5, degradation_factor)  # Floor at 50% of base
    
    return round(degraded_roughness, 1)


def get_degradation_percentage(material: str, age_years: float) -> float:
    """
    Calculate percentage degradation of pipe roughness.
    
    Args:
        material: Pipe material name
        age_years: Age of pipe in years
        
    Returns:
        Degradation percentage (0-100)
    """
    base_roughness = CONFIG.physics.ROUGHNESS_BASE.get(material, 120.0)
    current_roughness = get_roughness(material, age_years)
    
    degradation_pct = ((base_roughness - current_roughness) / base_roughness) * 100.0
    return round(degradation_pct, 1)


def validate_config() -> bool:
    """
    Validate configuration integrity.
    
    Returns:
        True if configuration is valid
    """
    try:
        # Check all materials have roughness and degradation
        materials = CONFIG.physics.ROUGHNESS_BASE.keys()
        assert all(m in CONFIG.physics.DEGRADATION_RATES for m in materials)
        
        # Check economic constants are positive
        assert CONFIG.economics.SENSOR_UNIT_COST > 0
        assert CONFIG.economics.DEFAULT_WATER_TARIFF > 0
        assert CONFIG.economics.ENERGY_COST_PER_KWH > 0
        
        # Check simulation defaults are reasonable
        assert 2 <= CONFIG.simulation.GRID_SIZE <= 10
        assert 0 < CONFIG.simulation.SENSOR_COVERAGE_RATIO <= 1.0
        
        return True
    except AssertionError:
        return False


# Validate on import
if not validate_config():
    raise ValueError("Configuration validation failed! Check CONFIG constants.")
