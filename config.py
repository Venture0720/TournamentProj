"""
Smart Shygyn PRO v3 — GLOBAL CONFIGURATION HUB
Централизованный модуль управления всеми параметрами системы: 
от цветовой палитры интерфейса до физических констант гидравлики.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any

@dataclass(frozen=True)
class UIColors:
    """Палитра цветов для экспертного интерфейса (Cyberpunk/Enterprise Dark)."""
    # Базовые фоновые цвета
    BG_DARK: str = "#0F172A"          # Deep Slate (основной фон)
    BG_CARD: str = "rgba(30, 41, 59, 0.7)" # Прозрачный фон для карточек
    
    # Семантические цвета (статусы)
    PRIMARY: str = "#3B82F6"          # Royal Blue (основные элементы)
    SECONDARY: str = "#6366F1"        # Indigo (второстепенные)
    ACCENT: str = "#06B6D4"           # Cyan (акценты, данные)
    
    # Статусы рисков и аномалий
    SUCCESS: str = "#10B981"          # Emerald (норма, экономия)
    WARNING: str = "#F59E0B"          # Amber (предупреждение)
    DANGER: str = "#EF4444"           # Rose (прорыв, критическое давление)
    INFO: str = "#8B5CF6"             # Violet (возраст воды, химия)
    
    # Текст
    TEXT_MAIN: str = "#F1F5F9"        # Почти белый
    TEXT_MUTED: str = "#94A3B8"       # Серый (подписи)

@dataclass(frozen=True)
class HydraulicDefaults:
    """Константы для модулей hydraulic_intelligence.py и wntr."""
    UNITS: str = "GPM"                # Единицы измерения (Gallons per minute или LPS)
    HEADLOSS: str = "H-W"             # Формула Хейзена-Вильямса
    MIN_PRESSURE: float = 2.0         # Минимальное допустимое давление (bar)
    MAX_PRESSURE: float = 7.0         # Максимальное допустимое давление (bar)
    DEFAULT_ROUGHNESS: int = 120      # Коэффициент шероховатости для новых труб
    DEMAND_MULTIPLIER: float = 1.0    # Коэффициент пикового потребления

@dataclass(frozen=True)
class RiskThresholds:
    """Пороги для risk_engine.py и leak_analytics.py."""
    CRITICAL_AGE: int = 40            # Возраст трубы, считающийся критическим (лет)
    LEAK_PROBABILITY_HIGH: float = 0.75 # Порог высокой вероятности утечки
    BREAK_REPAIR_COST_AVG: int = 150000 # Средняя стоимость ремонта порыва (₸)

@dataclass
class MapSettings:
    """Настройки ГИС и визуализации карт."""
    DEFAULT_ZOOM: int = 12
    TILE_LAYERS: Dict[str, str] = field(default_factory=lambda: {
        "Dark Flow": "cartodb dark_matter",
        "Light Tech": "cartodb positron",
        "Satellite Intelligence": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    })
    # Атрибуты иконок
    NODE_RADIUS: int = 6
    PIPE_WEIGHT: int = 4

@dataclass
class AppConfig:
    """Главный объект конфигурации."""
    
    # Метаданные
    VERSION: str = "3.0.0-PRO"
    APP_NAME: str = "Smart Shygyn Intelligence"
    
    # Инициализация подразделов
    UI: UIColors = UIColors()
    PHYSICS: HydraulicDefaults = HydraulicDefaults()
    RISK: RiskThresholds = RiskThresholds()
    MAP: MapSettings = MapSettings()
    
    # Системные настройки
    CACHE_TTL: int = 3600             # Кэш на 1 час для тяжелых расчетов
    DEBUG_MODE: bool = False
    
    # Экономика проекта
    WATER_UNIT_COST: float = 145.5    # Тенге за м³
    ENERGY_COST_KWT: float = 24.8     # Тенге за кВт/ч
    
    # CSS Стиль для Streamlit (Glassmorphism & Cyberpunk)
    def get_custom_css(self) -> str:
        return f"""
        <style>
            .stApp {{
                background: {self.UI.BG_DARK};
                color: {self.UI.TEXT_MAIN};
            }}
            [data-testid="stMetricValue"] {{
                color: {self.UI.ACCENT};
                font-family: 'JetBrains Mono', monospace;
                font-weight: 700;
            }}
            .stTabs [data-baseweb="tab-list"] {{
                gap: 24px;
            }}
            .stTabs [data-baseweb="tab"] {{
                height: 50px;
                white-space: pre-wrap;
                background-color: {self.UI.BG_CARD};
                border-radius: 10px 10px 0px 0px;
                color: {self.UI.TEXT_MUTED};
            }}
            .stTabs [aria-selected="true"] {{
                color: {self.UI.ACCENT} !important;
                border-bottom: 2px solid {self.UI.ACCENT} !important;
            }}
            div[data-testid="stExpander"] {{
                background-color: {self.UI.BG_CARD};
                border: 1px solid rgba(255,255,255,0.1);
                border-radius: 15px;
            }}
        </style>
        """

# Единственный экземпляр для всего проекта
CONFIG = AppConfig()
