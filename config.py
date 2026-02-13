"""
Smart Shygyn PRO v3 — GLOBAL CONFIGURATION HUB (Fixed)
Исправлена ошибка с mutable default для работы на Python 3.13.
"""

from dataclasses import dataclass, field
from typing import Dict, List

@dataclass(frozen=True)
class UIColors:
    """Палитра цветов для экспертного интерфейса."""
    BG_DARK: str = "#0F172A"
    BG_CARD: str = "rgba(30, 41, 59, 0.7)"
    PRIMARY: str = "#3B82F6"
    SECONDARY: str = "#6366F1"
    ACCENT: str = "#06B6D4"
    SUCCESS: str = "#10B981"
    WARNING: str = "#F59E0B"
    DANGER: str = "#EF4444"
    INFO: str = "#8B5CF6"
    TEXT_MAIN: str = "#F1F5F9"
    TEXT_MUTED: str = "#94A3B8"

@dataclass(frozen=True)
class HydraulicDefaults:
    """Константы для гидравлики."""
    UNITS: str = "LPS"
    HEADLOSS: str = "H-W"
    MIN_PRESSURE: float = 2.0
    MAX_PRESSURE: float = 7.0
    DEFAULT_ROUGHNESS: int = 120

@dataclass(frozen=True)
class RiskThresholds:
    """Пороги рисков."""
    CRITICAL_AGE: int = 40
    LEAK_PROBABILITY_HIGH: float = 0.75
    BREAK_REPAIR_COST_AVG: int = 150000

@dataclass
class MapSettings:
    """Настройки ГИС."""
    DEFAULT_ZOOM: int = 12
    # ИСПОЛЬЗУЕМ default_factory ЧТОБЫ ИЗБЕЖАТЬ ValueError
    TILE_LAYERS: Dict[str, str] = field(default_factory=lambda: {
        "Dark Flow": "cartodb dark_matter",
        "Light Tech": "cartodb positron",
        "Satellite Intelligence": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    })
    NODE_RADIUS: int = 6
    PIPE_WEIGHT: int = 4

@dataclass
class AppConfig:
    """Главный объект конфигурации."""
    VERSION: str = "3.0.1-PRO"
    APP_NAME: str = "Smart Shygyn Intelligence"
    
    UI: UIColors = field(default_factory=UIColors)
    PHYSICS: HydraulicDefaults = field(default_factory=HydraulicDefaults)
    RISK: RiskThresholds = field(default_factory=RiskThresholds)
    MAP: MapSettings = field(default_factory=MapSettings)
    
    CACHE_TTL: int = 3600
    WATER_UNIT_COST: float = 145.5
    ENERGY_COST_KWT: float = 24.8
    
    def get_custom_css(self) -> str:
        return f"""
        <style>
            .stApp {{ background: {self.UI.BG_DARK}; color: {self.UI.TEXT_MAIN}; }}
            [data-testid="stMetricValue"] {{ color: {self.UI.ACCENT}; font-weight: 700; }}
            div[data-testid="stExpander"] {{ background-color: {self.UI.BG_CARD}; border-radius: 15px; }}
        </style>
        """

# Глобальный экземпляр
CONFIG = AppConfig()
