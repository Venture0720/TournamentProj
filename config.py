"""
Configuration constants and settings.
"""
from dataclasses import dataclass
from typing import Dict


@dataclass
class AppConfig:
    """Application-wide configuration."""

    # Visual Theme  (оригинал — не тронут)
    PRIMARY_COLOR   = "#1E3A8A"
    SECONDARY_COLOR = "#06B6D4"
    ACCENT_COLOR    = "#3B82F6"
    DANGER_COLOR    = "#EF4444"
    WARNING_COLOR   = "#F59E0B"
    SUCCESS_COLOR   = "#10B981"

    # Performance  (оригинал)
    CACHE_TTL = 3600

    # Export  (оригинал)
    EXPORT_DPI    = 300
    EXPORT_FORMAT = "PDF"

    # Simulation  (оригинал)
    DEFAULT_GRID_SIZE       = 4
    DEFAULT_SENSOR_COVERAGE = 0.30

    # Mapping  (оригинал)
    MAP_TILE_OPTIONS = {
        "dark":      "CartoDB dark_matter",
        "light":     "OpenStreetMap",
        "satellite": "Esri WorldImagery",
    }

    # BattLeDIM датасет (добавлено)
    BATTLEDIM_DATA_DIR   = "data/battledim"
    BATTLEDIM_GDRIVE_URL = "https://drive.google.com/drive/folders/1OL2xEGTKEA-eoaxRgd0n8vUEsGzj9Ngq"
    BATTLEDIM_DOI        = "10.5281/zenodo.4017659"
    BATTLEDIM_CITATION   = "Vrachimis et al. (2020). BattLeDIM. CCWI/WDSA 2020."
    BATTLEDIM_NODES      = 782
    BATTLEDIM_PIPES      = 909
    BATTLEDIM_KM         = 42.6
    BATTLEDIM_LEAKS      = 23

    # Реальные данные Казахстана (добавлено)
    TARIFF_ALMATY_KZT_M3     = 91.96
    TARIFF_ASTANA_KZT_M3     = 85.00
    TARIFF_TURKESTAN_KZT_M3  = 70.00
    WEAR_ALMATY_PCT          = 54.5
    WEAR_ASTANA_PCT          = 48.0
    WEAR_TURKESTAN_PCT       = 62.0
    SENSOR_PRESSURE_COST_KZT = 380_000
    SENSOR_FLOW_COST_KZT     = 520_000
    SMART_METER_COST_KZT     =  45_000
    ELECTRICITY_TARIFF_KZT_KWH      = 22.0
    CO2_FACTOR_KG_KWH                = 0.62
    MIN_PRESSURE_RESIDENTIAL_BAR     = 2.5
    MAX_PRESSURE_RESIDENTIAL_BAR     = 6.0


# Global instance
CONFIG = AppConfig()
