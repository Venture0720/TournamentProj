import streamlit as st
import pandas as pd
import numpy as np
import os

# 1. Настройка страницы
st.set_page_config(page_title="Digital Twin Dashboard", layout="wide")

# Проверка наличия файлов (для отладки)
files_in_dir = os.listdir('.')
st.sidebar.write(f"📁 Файлы в системе: {files_in_dir}")

# 2. Безопасный импорт бэкенда
try:
    from risk_engine import DigitalTwinEngine, DigitalTwinAPIResponse
    from hydraulic_intelligence import HydraulicIntelligenceEngine
    from leak_analytics import LeakAnalyticsEngine
except ImportError as e:
    st.error(f"❌ Ошибка импорта модуля: {e}")
    st.stop()

# 3. Классы настроек и оркестратора
class GlobalSettings:
    CHLORINE_THRESHOLD = 0.2
    CITY_DATA = {
        "Astana": {"temp": 5, "soil": "clay"},
        "Almaty": {"temp": 12, "soil": "rocky"},
        "Turkestan": {"temp": 18, "soil": "sandy"}
    }

class MasterOrchestrator:
    def __init__(self, city: str, pipe_material: str, pipe_age: int):
        self.settings = GlobalSettings()
        self.city_info = self.settings.CITY_DATA.get(city, self.settings.CITY_DATA["Astana"])
        self.engine = DigitalTwinEngine(
            city=city,
            season_temp=self.city_info["temp"],
            material=pipe_material,
            age=pipe_age
        )

    def compute_full_cycle(self, grid_size: int, leak_node: int, leak_size: float) -> DigitalTwinAPIResponse:
        try:
            return self.engine.run_complete_analysis(
                grid_size=grid_size,
                leak_node=leak_node,
                leak_area_cm2=leak_size,
                n_sensors=max(2, grid_size // 2)
            )
        except Exception as e:
            st.error(f"Ошибка в расчетах: {e}")
            return None

# 4. Интерфейс управления (Sidebar)
with st.sidebar:
    st.header("⚙️ Управление двойником")
    city = st.selectbox("Город", list(GlobalSettings.CITY_DATA.keys()))
    material = st.selectbox("Материал", ["Cast Iron", "HDPE", "Steel"])
    age = st.slider("Возраст труб", 0, 60, 25)
    
    st.markdown("---")
    grid_size = st.number_input("Размер сети", 5, 20, 10)
    leak_node = st.number_input("Узел утечки", 0, grid_size**2 - 1, 5)
    leak_size = st.slider("Размер утечки (см2)", 0.1, 10.0, 2.5)

    if st.button("🚀 Запустить расчет", use_container_width=True):
        orchestrator = MasterOrchestrator(city, material, age)
        with st.spinner("Синхронизация с датчиками..."):
            report = orchestrator.compute_full_cycle(grid_size, leak_node, leak_size)
            st.session_state.report = report

# 5. Вывод результатов (Твой исправленный блок)
st.title("🌊 Smart Water Digital Twin")

if "report" in st.session_state and st.session_state.report:
    res = st.session_state.report
    
    try:
        col1, col2, col3 = st.columns(3)
        
        # Безопасное извлечение данных
        status_val = getattr(res, 'status', 'N/A')
        leak_data = getattr(res, 'leak_detection', None)
        is_leak = getattr(leak_data, 'leak_detected', False) if leak_data else False
        
        quality_data = getattr(res, 'water_quality', None)
        chlorine = getattr(quality_data, 'chlorine_residual_mg_l', 0.0) if quality_data else 0.0

        col1.metric("Статус системы", status_val)
        col2.metric("Детектор утечек", "⚠ ОБНАРУЖЕНА" if is_leak else "✅ НОРМА")
        col3.metric("Хлор (остаток)", f"{chlorine} мг/л")

        # Дополнительно: Таблица рисков
        risk_data = getattr(res, 'criticality_assessment', None)
        if risk_data and hasattr(risk_data, 'maintenance_priorities'):
            st.subheader("📋 Приоритеты обслуживания")
            st.table(pd.DataFrame(risk_data.maintenance_priorities))
            
    except Exception as e:
        st.warning(f"Ошибка отображения данных: {e}")
        st.write("Сырые данные отчета:", res)
else:
    st.info("Настройте параметры в боковой панели и нажмите кнопку 'Запустить расчет'.")
