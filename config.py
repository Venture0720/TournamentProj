"""Configuration and design tokens for Smart Shygyn PRO v3.
Python 3.13 compatible (safe defaults for mutable fields).
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class AppMetadata:
    name: str = "Smart Shygyn PRO v3"
    tagline: str = "Cyberpunk Digital Twin for Water Distribution"
    version: str = "3.0.0"
    icon: str = "💧"
    page_title: str = "Smart Shygyn PRO v3"
    layout: str = "wide"


@dataclass(frozen=True)
class UIPalette:
    background: str = "#0F172A"  # Slate 900
    surface: str = "#111827"     # Gray 900
    surface_alt: str = "#1F2937" # Gray 800
    border: str = "#1E293B"      # Slate 800
    text: str = "#E2E8F0"        # Slate 200
    text_muted: str = "#94A3B8"  # Slate 400
    cyan: str = "#06B6D4"        # Cyan 500
    emerald: str = "#10B981"     # Emerald 500
    rose: str = "#EF4444"        # Rose 500
    violet: str = "#8B5CF6"      # Violet 500
    amber: str = "#F59E0B"       # Amber 500


@dataclass(frozen=True)
class PhysicsConfig:
    headloss: str = "H-W"
    units: str = "LPS"
    min_pressure_bar: float = 2.0
    base_reservoir_head_m: float = 40.0
    default_pipe_diameter_m: float = 0.2
    default_pipe_length_m: float = 100.0


@dataclass(frozen=True)
class VisualizationConfig:
    plotly_template: str = "plotly_dark"
    default_height: int = 380
    large_height: int = 560
    small_height: int = 260
    pressure_colorscale: List[List] = field(default_factory=lambda: [
        [0.0, "#EF4444"],
        [0.5, "#F59E0B"],
        [1.0, "#10B981"],
    ])
    risk_colorscale: List[List] = field(default_factory=lambda: [
        [0.0, "#10B981"],
        [0.5, "#F59E0B"],
        [1.0, "#EF4444"],
    ])


@dataclass(frozen=True)
class MapConfig:
    default_zoom: int = 11
    tile_style: str = "CartoDB dark_matter"
    risk_colors: Dict[str, str] = field(default_factory=lambda: {
        "LOW": "#10B981",
        "MEDIUM": "#F59E0B",
        "HIGH": "#EF4444",
        "CRITICAL": "#EF4444",
    })


@dataclass(frozen=True)
class CityManager:
    cities: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "Almaty": {"lat": 43.238949, "lon": 76.889709, "ground_temp": 12.0},
        "Astana": {"lat": 51.169392, "lon": 71.449074, "ground_temp": -2.5},
        "Turkistan": {"lat": 43.29733, "lon": 68.25175, "ground_temp": 22.0},
    })

    def list_cities(self) -> List[str]:
        return list(self.cities.keys())

    def get_city(self, city_name: str) -> Dict[str, float]:
        return self.cities.get(city_name, self.cities["Almaty"])


UI = UIPalette()
APP = AppMetadata()
PHYSICS = PhysicsConfig()
VIZ = VisualizationConfig()
MAPS = MapConfig()
CITY = CityManager()


def get_custom_css() -> str:
    """Return custom CSS for cyberpunk glassmorphism UI."""
    return f"""
    <style>
    html, body, [class*="stApp"] {{
        background: radial-gradient(circle at 20% 20%, #111827, {UI.background});
        color: {UI.text};
    }}
    .block-container {{
        padding-top: 1.2rem;
    }}
    .glass-card {{
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid {UI.border};
        border-radius: 15px;
        padding: 1rem 1.2rem;
        box-shadow: 0 0 20px rgba(6, 182, 212, 0.08);
    }}
    .metric-glow {{
        border: 1px solid rgba(6, 182, 212, 0.6);
        border-radius: 15px;
        padding: 0.8rem 1rem;
        background: rgba(17, 24, 39, 0.75);
        box-shadow: 0 0 18px rgba(6, 182, 212, 0.35);
    }}
    .metric-title {{
        font-size: 0.75rem;
        text-transform: uppercase;
        color: {UI.text_muted};
        letter-spacing: 0.12em;
    }}
    .metric-value {{
        font-size: 1.6rem;
        font-weight: 700;
        color: {UI.cyan};
    }}
    .highlight-danger {{ color: {UI.rose}; font-weight: 700; }}
    .highlight-success {{ color: {UI.emerald}; font-weight: 700; }}
    .highlight-warning {{ color: {UI.amber}; font-weight: 700; }}
    .section-title {{
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: {UI.text};
    }}
    </style>
    """
