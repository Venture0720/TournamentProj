"""
BACKEND ADAPTER
Связующее звено между интерфейсом и твоими расчетными модулями.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass

# Импортируем твои модули
import hydraulic_intelligence as hi
import leak_analytics as la
import risk_engine as re
from config import CONFIG

@dataclass
class CityConfig:
    name: str
    lat: float
    lng: float
    zoom: int
    ground_temp_celsius: float

class CityManager:
    CITIES = {
        "Алматы": CityConfig("Алматы", 43.238, 76.889, 12, 14.5),
        "Астана": CityConfig("Астана", 51.160, 71.470, 11, 8.2),
        "Туркестан": CityConfig("Туркестан", 43.305, 68.257, 13, 18.0)
    }
    def __init__(self, city_name):
        self.config = self.CITIES.get(city_name, self.CITIES["Алматы"])

class HydraulicPhysics:
    def run_simulation(self, pipe_age, pressure_setpoint):
        """
        Интеграция с твоим hydraulic_intelligence.py
        """
        # Пока твои модули очень тяжелые, используем генерацию данных, 
        # но подготавливаем вызов для hi.calculate()
        hours = np.linspace(0, 24, 48)
        
        # Пример логики: чем больше возраст из risk_engine, тем хуже гидравлика
        risk_factor = pipe_age * 0.02 
        
        return pd.DataFrame({
            "Hour": hours,
            "Pressure": pressure_setpoint - risk_factor + 0.3 * np.sin(hours),
            "Flow": 120 + 30 * np.abs(np.cos(hours/4)),
            "WaterAge": 1.5 + 0.2 * hours,
            "PumpHead": 70.0
        })

    def get_leak_risk(self, pipe_age):
        """
        Интеграция с leak_analytics.py и risk_engine.py
        """
        # Здесь в будущем будет: return re.calculate_risk(pipe_age)
        return 12 + (pipe_age * 0.4)
