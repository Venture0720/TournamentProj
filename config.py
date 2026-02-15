"""
Smart Shygyn PRO v3 — Professional Config Engine
Supports Python 3.13, Modern CSS Injection, and Multi-City GIS.
"""
from dataclasses import dataclass, field
from typing import Dict

@dataclass(frozen=True)
class UIColors:
    # Cyberpunk Dark Theme
    DARK_BG: str = "#0F172A"
    DARK_CARD: str = "rgba(30, 41, 59, 0.7)"
    DARK_ACCENT: str = "#06B6D4"  # Cyan
    DARK_TEXT: str = "#E2E8F0"
    
    # Professional Light Theme
    LIGHT_BG: str = "#F8FAFC"
    LIGHT_CARD: str = "rgba(255, 255, 255, 0.9)"
    LIGHT_ACCENT: str = "#1E3A8A" # Deep Blue
    LIGHT_TEXT: str = "#1E293B"

@dataclass
class AppConfig:
    """Application-wide configuration with modern UI injection."""
    
    # Цвета и стили
    colors: UIColors = field(default_factory=UIColors)
    
    # Симуляция и физика
    CACHE_TTL: int = 3600
    DEFAULT_GRID_SIZE: int = 4
    
    # Коэффициенты Хейзена-Вильямса (Материал: C-фактор)
    ROUGHNESS_MAP: Dict[str, float] = field(default_factory=lambda: {
        "Пластик (ПНД)": 145.0,
        "Сталь": 120.0,
        "Чугун": 100.0
    })

    def get_style(self, dark_mode: bool) -> str:
        """Генерирует CSS для инъекции в Streamlit."""
        c = self.colors
        bg = c.DARK_BG if dark_mode else c.LIGHT_BG
        card = c.DARK_CARD if dark_mode else c.LIGHT_CARD
        text = c.DARK_TEXT if dark_mode else c.LIGHT_TEXT
        accent = c.DARK_ACCENT if dark_mode else c.LIGHT_ACCENT
        
        return f"""
        <style>
            /* Главный контейнер */
            [data-testid="stAppViewContainer"] {{
                background-color: {bg};
                color: {text};
            }}
            
            /* Боковая панель */
            [data-testid="stSidebar"] {{
                background-color: {bg};
                border-right: 1px solid {accent}33;
            }}

            /* Стилизация карточек (метрики и блоки) */
            div[data-testid="stMetric"] {{
                background: {card};
                padding: 15px;
                border-radius: 10px;
                border: 1px solid {accent}44;
                backdrop-filter: blur(10px);
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            }}

            /* Заголовки */
            h1, h2, h3 {{
                color: {accent} !important;
                font-family: 'Segoe UI', Roboto, sans-serif;
            }}

            /* Табы */
            .stTabs [data-baseweb="tab-list"] {{
                gap: 8px;
                background-color: transparent;
            }}

            .stTabs [data-baseweb="tab"] {{
                height: 40px;
                background-color: {card};
                border-radius: 4px 4px 0px 0px;
                border: 1px solid {accent}33;
                color: {text};
            }}

            .stTabs [aria-selected="true"] {{
                background-color: {accent} !important;
                color: white !important;
            }}
        </style>
        """

    def get_map_tile(self, dark_mode: bool) -> str:
        return "CartoDB dark_matter" if dark_mode else "OpenStreetMap"

# Глобальный экземпляр настроек
CONFIG = AppConfig()
