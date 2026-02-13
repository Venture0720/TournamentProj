"""
Configuration constants and settings.
"""
from dataclasses import dataclass
from typing import Dict

@dataclass
class AppConfig:
    """Application-wide configuration."""
    
    # Visual Theme (Tailwind-like colors)
    PRIMARY_COLOR = "#1E3A8A"      # Deep Blue
    SECONDARY_COLOR = "#06B6D4"    # Cyan
    ACCENT_COLOR = "#3B82F6"       # Blue
    DANGER_COLOR = "#EF4444"       # Red
    WARNING_COLOR = "#F59E0B"      # Amber
    SUCCESS_COLOR = "#10B981"      # Green
    
    # Performance
    CACHE_TTL = 3600  # 1 hour cache for static elements
    
    # Export
    EXPORT_DPI = 300
    EXPORT_FORMAT = "PDF"
    
    # Simulation
    DEFAULT_GRID_SIZE = 4
    DEFAULT_SENSOR_COVERAGE = 0.30
    
    # Mapping
    MAP_TILE_OPTIONS = {
        "dark": "CartoDB dark_matter",
        "light": "OpenStreetMap",
        "satellite": "Esri WorldImagery"
    }

# Global instance
CONFIG = AppConfig()
