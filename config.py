"""
Smart Shygyn PRO v3 — CONFIGURATION
Центральные настройки интерфейса, визуализации и симуляции.
"""
from dataclasses import dataclass
from typing import Dict

@dataclass
class AppConfig:
    """Глобальные константы приложения."""
    
    # --- Брендинг и Цвета (Modern Dark UI) ---
    PRIMARY_COLOR: str = "#1E3A8A"    # Глубокий синий
    SECONDARY_COLOR: str = "#06B6D4"  # Циан
    ACCENT_COLOR: str = "#3B82F6"     # Ярко-синий (для графиков давления)
    
    # Статусные цвета
    DANGER_COLOR: str = "#EF4444"     # Красный (утечки, критический риск)
    WARNING_COLOR: str = "#F59E0B"    # Янтарный (предупреждения)
    SUCCESS_COLOR: str = "#10B981"    # Зеленый (норма, экономия)
    INFO_COLOR: str = "#8B5CF6"       # Фиолетовый (возраст воды)
    
    # --- Настройки производительности ---
    CACHE_TTL: int = 3600             # Время жизни кэша в секундах (1 час)
    
    # --- Параметры карты (Folium) ---
    MAP_TILE_OPTIONS: Dict[str, str] = None
    
    def __post_init__(self):
        # Инициализация словаря после создания объекта
        self.MAP_TILE_OPTIONS = {
            "dark": "cartodbdarkmatter",
            "light": "cartodbpositron",
            "satellite": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
        }

    # --- Экономические константы ---
    WATER_COST_KZT: float = 150.0     # Стоимость куба воды для расчетов
    ENERGY_COST_KZT: float = 25.0      # Стоимость кВт/ч для насосов
    
    # --- Симуляция ---
    DEFAULT_GRID_SIZE: int = 4        # Размер сетки узлов (4x4)
    MIN_PRESSURE_THRESHOLD: float = 2.5 # Минимальное давление в bar

# Создаем глобальный экземпляр, который импортируется в другие модули
CONFIG = AppConfig()
