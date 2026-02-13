import streamlit as st
import pandas as pd
import numpy as np
import os

# Проверка наличия файлов перед импортом (для отладки в логах)
files_in_dir = os.listdir('.')
st.write(f"Файлы в директории: {files_in_dir}") # Это поможет увидеть реальные имена в облаке

try:
    # Проверь, что файл на GitHub называется именно risk_engine.py (маленькими буквами)
    from risk_engine import RiskEngine, DigitalTwinEngine, DigitalTwinAPIResponse
    from hydraulic_intelligence import HydraulicIntelligenceEngine
    from leak_analytics import LeakAnalyticsEngine
except ImportError as e:
    st.error(f"❌ Ошибка импорта модуля: {e}")
    st.info("Проверьте, что файлы risk_engine.py, hydraulic_intelligence.py и leak_analytics.py лежат в корне репозитория.")
    st.stop()
class GlobalSettings:
    """Все настройки проекта в одном месте (замена config.py)"""
    CHLORINE_THRESHOLD = 0.2  # мг/л (норма РК)
    CRITICAL_PRESSURE = 2.0    # бар
    CITY_DATA = {
        "Astana": {"temp": 5, "soil": "clay"},
        "Almaty": {"temp": 12, "soil": "rocky"},
        "Turkestan": {"temp": 18, "soil": "sandy"}
    }
    COLORS = {"primary": "#1E3A8A", "danger": "#EF4444", "success": "#10B981"}

class MasterOrchestrator:
    """
    Класс-интегратор. 
    Сюда интерфейс будет подавать вводные данные, 
    а на выходе получать полный отчет.
    """
    def __init__(self, city: str, pipe_material: str, pipe_age: int):
        self.settings = GlobalSettings()
        self.city_info = self.settings.CITY_DATA.get(city, self.settings.CITY_DATA["Astana"])
        
        # Инициализируем главный движок из Part 3
        self.engine = DigitalTwinEngine(
            city=city,
            season_temp=self.city_info["temp"],
            material=pipe_material,
            age=pipe_age
        )

    def compute_full_cycle(self, grid_size: int, leak_node: int, leak_size: float) -> DigitalTwinAPIResponse:
        """
        Запуск полной цепочки анализа без потери деталей:
        1. Гидравлика (Part 1)
        2. Утечки (Part 2) 
        3. Риски и Хлор (Part 3)
        """
        try:
            # Вызываем метод, который ты отлаживал в Part 3
            result = self.engine.run_complete_analysis(
                grid_size=grid_size,
                leak_node=leak_node,
                leak_area_cm2=leak_size,
                n_sensors=max(2, grid_size // 2)
            )
            return result
        except Exception as e:
            print(f"Критическая ошибка интеграции: {e}")
            return None

# Пример запуска для проверки (можно удалить при подключении UI)
if __name__ == "__main__":
    orchestrator = MasterOrchestrator("Astana", "Cast Iron", 35)
    report = orchestrator.compute_full_cycle(grid_size=5, leak_node=12, leak_size=5.5)
    if report:
        print(f"Статус: {report.status}")
        print(f"Найдена утечка: {report.leak_detection.leak_detected}")
        print(f"Приоритетов к ремонту: {len(report.criticality_assessment.maintenance_priorities)}")
