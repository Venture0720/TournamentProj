"""
Smart Shygyn - Digital Twin Configuration
Professional configuration module for Kazakhstan's Water Networks Digital Twin
Astana Hub Competition 2026
"""

from dataclasses import dataclass, field
from typing import Dict, List
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════════
# BRANDING & THEME CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class BrandColors:
    """Official Smart Shygyn color palette - Corporate identity"""
    PRIMARY_COLOR: str = "#1E3A8A"      # Deep Blue - Trust & Reliability
    SECONDARY_COLOR: str = "#06B6D4"    # Cyan - Water & Technology
    SUCCESS_COLOR: str = "#10B981"      # Green - Operational Excellence
    WARNING_COLOR: str = "#F59E0B"      # Amber - Attention Required
    DANGER_COLOR: str = "#EF4444"       # Red - Critical Alert
    INFO_COLOR: str = "#3B82F6"         # Blue - Information
    BACKGROUND_DARK: str = "#0F172A"    # Slate - Dark Mode
    BACKGROUND_LIGHT: str = "#F8FAFC"   # Light Slate - Light Mode
    TEXT_PRIMARY: str = "#1E293B"       # Dark Slate
    TEXT_SECONDARY: str = "#64748B"     # Slate Gray
    BORDER_COLOR: str = "#E2E8F0"       # Light Border
    
    # Gradient Definitions
    GRADIENT_PRIMARY: str = "linear-gradient(135deg, #1E3A8A 0%, #06B6D4 100%)"
    GRADIENT_SUCCESS: str = "linear-gradient(135deg, #10B981 0%, #34D399 100%)"
    GRADIENT_DANGER: str = "linear-gradient(135deg, #EF4444 0%, #F87171 100%)"


@dataclass(frozen=True)
class AppMetadata:
    """Application metadata and branding"""
    APP_NAME: str = "Smart Shygyn"
    APP_TAGLINE: str = "Digital Twin for Kazakhstan's Water Networks"
    VERSION: str = "1.0.0"
    ORGANIZATION: str = "Astana Hub Innovation"
    COMPETITION: str = "Water Infrastructure Digitalization 2026"
    ICON: str = "💧"
    PAGE_TITLE: str = "Smart Shygyn | Water Network Digital Twin"
    LAYOUT: str = "wide"


# ═══════════════════════════════════════════════════════════════════════════
# TECHNICAL CONSTRAINTS & STANDARDS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SystemConstraints:
    """Default operational constraints for Kazakhstan water networks"""
    
    # Network Topology
    DEFAULT_GRID_SIZE: int = 4
    MIN_GRID_SIZE: int = 3
    MAX_GRID_SIZE: int = 8
    
    # Pressure Standards (bar)
    MIN_PRESSURE_BAR: float = 2.0
    MAX_PRESSURE_BAR: float = 6.0
    OPTIMAL_PRESSURE_BAR: float = 3.5
    
    # Water Quality Standards (Kazakhstan GOST)
    CHLORINE_MIN_THRESHOLD: float = 0.2   # mg/L - Minimum residual
    CHLORINE_MAX_THRESHOLD: float = 1.2   # mg/L - Maximum safe level
    CHLORINE_OPTIMAL: float = 0.5         # mg/L - Target level
    MAX_WATER_AGE_HOURS: float = 48.0     # Maximum acceptable water age
    
    # Leak Detection Parameters
    MIN_LEAK_AREA_CM2: float = 0.1
    MAX_LEAK_AREA_CM2: float = 100.0
    DEFAULT_LEAK_AREA_CM2: float = 5.0
    
    # Sensor Configuration
    MIN_SENSORS: int = 2
    MAX_SENSORS: int = 20
    DEFAULT_SENSORS: int = 6
    
    # Temperature Ranges (°C)
    MIN_SOIL_TEMP: float = -40.0  # Kazakhstan winter extreme
    MAX_SOIL_TEMP: float = 40.0   # Summer extreme
    DEFAULT_SOIL_TEMP: float = 10.0


@dataclass(frozen=True)
class KazakhstanCities:
    """Major cities in Kazakhstan with default parameters"""
    ASTANA: Dict = field(default_factory=lambda: {
        "name": "Astana",
        "lat": 51.1694,
        "lon": 71.4491,
        "default_temp": -15.0,  # Winter average
        "population": 1_200_000,
        "timezone": "Asia/Almaty"
    })
    
    ALMATY: Dict = field(default_factory=lambda: {
        "name": "Almaty",
        "lat": 43.2220,
        "lon": 76.8512,
        "default_temp": 5.0,
        "population": 2_000_000,
        "timezone": "Asia/Almaty"
    })
    
    TURKESTAN: Dict = field(default_factory=lambda: {
        "name": "Turkestan",
        "lat": 43.2978,
        "lon": 68.2517,
        "default_temp": 8.0,
        "population": 170_000,
        "timezone": "Asia/Almaty"
    })
    
    SHYMKENT: Dict = field(default_factory=lambda: {
        "name": "Shymkent",
        "lat": 42.3417,
        "lon": 69.5901,
        "default_temp": 10.0,
        "population": 1_000_000,
        "timezone": "Asia/Almaty"
    })
    
    @classmethod
    def get_city_list(cls) -> List[str]:
        """Return list of city names"""
        return ["Astana", "Almaty", "Turkestan", "Shymkent"]
    
    @classmethod
    def get_city_data(cls, city_name: str) -> Dict:
        """Get city data by name"""
        city_map = {
            "Astana": cls.ASTANA,
            "Almaty": cls.ALMATY,
            "Turkestan": cls.TURKESTAN,
            "Shymkent": cls.SHYMKENT
        }
        return city_map.get(city_name, cls.ASTANA)


class PipeMaterial(Enum):
    """Supported pipe materials with properties"""
    CAST_IRON = "Cast Iron"
    DUCTILE_IRON = "Ductile Iron"
    PVC = "PVC"
    HDPE = "HDPE"
    STEEL = "Steel"
    CONCRETE = "Concrete"
    
    @classmethod
    def get_materials_list(cls) -> List[str]:
        """Return list of material names"""
        return [material.value for material in cls]


# ═══════════════════════════════════════════════════════════════════════════
# MAP CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

class MapStyles:
    """Map tile providers and styles"""
    
    STYLES: Dict[str, Dict] = {
        "Dark": {
            "url": "https://cartodb-basemaps-{s}.global.ssl.fastly.net/dark_all/{z}/{x}/{y}.png",
            "attribution": "© OpenStreetMap contributors © CARTO",
            "name": "Dark Matter"
        },
        "Light": {
            "url": "https://cartodb-basemaps-{s}.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png",
            "attribution": "© OpenStreetMap contributors © CARTO",
            "name": "Positron"
        },
        "Satellite": {
            "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            "attribution": "© Esri",
            "name": "ESRI Satellite"
        },
        "OpenStreetMap": {
            "url": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
            "attribution": "© OpenStreetMap contributors",
            "name": "OpenStreetMap"
        },
        "Terrain": {
            "url": "https://stamen-tiles-{s}.a.ssl.fastly.net/terrain/{z}/{x}/{y}.jpg",
            "attribution": "Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap",
            "name": "Stamen Terrain"
        }
    }
    
    DEFAULT_STYLE: str = "Dark"
    DEFAULT_ZOOM: int = 12
    
    @classmethod
    def get_style_names(cls) -> List[str]:
        """Return list of available map styles"""
        return list(cls.STYLES.keys())


# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class VisualizationConfig:
    """Chart and visualization settings"""
    
    # Plotly Theme
    PLOTLY_THEME: str = "plotly_dark"
    PLOTLY_TEMPLATE: str = "plotly_dark"
    
    # Chart Colors
    PRESSURE_COLORSCALE: List[List] = field(default_factory=lambda: [
        [0.0, "#EF4444"],    # Low pressure - Red
        [0.5, "#F59E0B"],    # Medium - Amber
        [1.0, "#10B981"]     # High pressure - Green
    ])
    
    RISK_COLORSCALE: List[List] = field(default_factory=lambda: [
        [0.0, "#10B981"],    # Low risk - Green
        [0.5, "#F59E0B"],    # Medium - Amber
        [1.0, "#EF4444"]     # High risk - Red
    ])
    
    # Chart Dimensions
    DEFAULT_CHART_HEIGHT: int = 400
    LARGE_CHART_HEIGHT: int = 600
    SMALL_CHART_HEIGHT: int = 300
    
    # Animation Settings
    ANIMATION_DURATION: int = 750
    TRANSITION_DURATION: int = 500


# ═══════════════════════════════════════════════════════════════════════════
# RISK & CRITICALITY THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════

class RiskThresholds:
    """Criticality and risk classification thresholds"""
    
    # Criticality Index Ranges
    CRITICAL_THRESHOLD: float = 0.7
    HIGH_THRESHOLD: float = 0.5
    MEDIUM_THRESHOLD: float = 0.3
    LOW_THRESHOLD: float = 0.0
    
    # Severity Score Ranges
    SEVERE_LEAK_THRESHOLD: float = 0.8
    MAJOR_LEAK_THRESHOLD: float = 0.6
    MODERATE_LEAK_THRESHOLD: float = 0.4
    MINOR_LEAK_THRESHOLD: float = 0.2
    
    # Alert Levels
    ALERT_LEVELS: Dict[str, str] = {
        "CRITICAL": "#EF4444",
        "HIGH": "#F59E0B",
        "MEDIUM": "#3B82F6",
        "LOW": "#10B981",
        "INFO": "#06B6D4"
    }
    
    @classmethod
    def get_risk_class(cls, criticality: float) -> str:
        """Determine risk class from criticality index"""
        if criticality >= cls.CRITICAL_THRESHOLD:
            return "CRITICAL"
        elif criticality >= cls.HIGH_THRESHOLD:
            return "HIGH"
        elif criticality >= cls.MEDIUM_THRESHOLD:
            return "MEDIUM"
        else:
            return "LOW"
    
    @classmethod
    def get_risk_color(cls, risk_class: str) -> str:
        """Get color for risk class"""
        return cls.ALERT_LEVELS.get(risk_class, cls.ALERT_LEVELS["INFO"])


# ═══════════════════════════════════════════════════════════════════════════
# SIMULATION PRESETS
# ═══════════════════════════════════════════════════════════════════════════

class SimulationPresets:
    """Pre-configured simulation scenarios"""
    
    SCENARIOS: Dict[str, Dict] = {
        "Normal Operations": {
            "description": "Routine monitoring with no anomalies",
            "leak_area_cm2": 0.0,
            "grid_size": 4,
            "sensors": 6
        },
        "Minor Leak Detection": {
            "description": "Small leak scenario for early detection testing",
            "leak_area_cm2": 2.0,
            "grid_size": 4,
            "sensors": 8
        },
        "Major Incident": {
            "description": "Large leak requiring immediate response",
            "leak_area_cm2": 25.0,
            "grid_size": 5,
            "sensors": 10
        },
        "Winter Stress Test": {
            "description": "Extreme cold conditions in Astana",
            "leak_area_cm2": 5.0,
            "grid_size": 6,
            "sensors": 12,
            "soil_temp": -35.0
        },
        "Summer Peak Demand": {
            "description": "High demand summer scenario",
            "leak_area_cm2": 3.0,
            "grid_size": 6,
            "sensors": 10,
            "soil_temp": 30.0
        }
    }
    
    @classmethod
    def get_scenario_names(cls) -> List[str]:
        """Return list of scenario names"""
        return list(cls.SCENARIOS.keys())


# ═══════════════════════════════════════════════════════════════════════════
# EXPORT CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ExportConfig:
    """Data export and reporting settings"""
    
    REPORT_FORMATS: List[str] = field(default_factory=lambda: ["PDF", "JSON", "CSV", "Excel"])
    DEFAULT_FORMAT: str = "PDF"
    
    # Export File Naming
    REPORT_PREFIX: str = "SmartShygyn_Report"
    TIMESTAMP_FORMAT: str = "%Y%m%d_%H%M%S"
    
    # Data Retention
    MAX_HISTORY_DAYS: int = 90
    ARCHIVE_AFTER_DAYS: int = 30


# ═══════════════════════════════════════════════════════════════════════════
# SINGLETON INSTANCES
# ═══════════════════════════════════════════════════════════════════════════

# Create singleton instances for easy access
COLORS = BrandColors()
METADATA = AppMetadata()
CONSTRAINTS = SystemConstraints()
CITIES = KazakhstanCities()
MAPS = MapStyles()
VIZ = VisualizationConfig()
RISK = RiskThresholds()
PRESETS = SimulationPresets()
EXPORT = ExportConfig()


# ═══════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def get_temperature_emoji(temp: float) -> str:
    """Return emoji based on temperature"""
    if temp < -20:
        return "🥶"
    elif temp < 0:
        return "❄️"
    elif temp < 15:
        return "🌡️"
    elif temp < 30:
        return "☀️"
    else:
        return "🔥"


def get_status_emoji(status: str) -> str:
    """Return emoji based on status"""
    status_map = {
        "SUCCESS": "✅",
        "ERROR": "❌",
        "WARNING": "⚠️",
        "INFO": "ℹ️",
        "CRITICAL": "🚨"
    }
    return status_map.get(status.upper(), "ℹ️")


def format_number(value: float, decimals: int = 2, suffix: str = "") -> str:
    """Format number with proper decimal places and suffix"""
    return f"{value:.{decimals}f}{suffix}"


def get_severity_label(score: float) -> str:
    """Convert severity score to human-readable label"""
    if score >= RISK.SEVERE_LEAK_THRESHOLD:
        return "🚨 SEVERE"
    elif score >= RISK.MAJOR_LEAK_THRESHOLD:
        return "⛔ MAJOR"
    elif score >= RISK.MODERATE_LEAK_THRESHOLD:
        return "⚠️ MODERATE"
    elif score >= RISK.MINOR_LEAK_THRESHOLD:
        return "🔔 MINOR"
    else:
        return "✅ NORMAL"


if __name__ == "__main__":
    # Configuration validation
    print(f"🚀 {METADATA.APP_NAME} v{METADATA.VERSION}")
    print(f"📍 Cities: {CITIES.get_city_list()}")
    print(f"🎨 Primary Color: {COLORS.PRIMARY_COLOR}")
    print(f"🗺️ Map Styles: {MAPS.get_style_names()}")
    print(f"🔧 Grid Size Range: {CONSTRAINTS.MIN_GRID_SIZE}-{CONSTRAINTS.MAX_GRID_SIZE}")
    print(f"✅ Configuration validated successfully!")
