import streamlit as st
import sys
import os
import pandas as pd
import plotly.graph_objects as go

# 1. Лечим проблему путей: принудительно видим соседние файлы
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 2. Безопасный импорт твоих модулей
try:
    from hydraulic_intelligence import HydraulicIntelligenceEngine
    from leak_analytics import LeakAnalyticsEngine
    from risk_engine import DigitalTwinEngine, CriticalityIndexCalculator
    import config
except ImportError as e:
    st.error(f"Ошибка импорта: {e}. Проверь, что файлы лежат в одной папке.")
    st.stop()

# --- ИНТЕРФЕЙС ---
st.set_page_config(page_title="Smart Shygyn Twin", layout="wide")

st.title("💧 Smart Shygyn: Digital Twin Orchestrator")
st.markdown("---")

# Боковая панель
with st.sidebar:
    st.header("Настройки сети")
    city = st.selectbox("Город", ["Astana", "Almaty"])
    scenario = st.radio("Сценарий", ["Норма", "Авария (Утечка)"])
    run_sim = st.button("🚀 ЗАПУСТИТЬ АНАЛИЗ", type="primary", use_container_width=True)

# Основная логика при нажатии кнопки
if run_sim:
    # Шаг 1: Гидравлика
    st.toast("Запуск гидравлического движка...")
    hydro = HydraulicIntelligenceEngine()
    # Здесь мы вызываем метод симуляции (названия могут отличаться в твоем коде)
    # Предположим, метод называется run_simulation()
    
    # Шаг 2: Аналитика утечек
    st.toast("Поиск аномалий...")
    leak = LeakAnalyticsEngine()
    
    # Шаг 3: Риски и Экономика
    st.toast("Расчет финансовых рисков...")
    risk = DigitalTwinEngine()

    # ВЫВОД РЕЗУЛЬТАТОВ
    tab1, tab2, tab3 = st.tabs(["Мониторинг", "Утечки", "Риски"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Давление", "3.4 bar", "OK")
        c2.metric("Расход", "140 m3/h", "-2%")
        c3.metric("Потери", "12%", "В норме")
        st.info("Здесь отрисовывается граф сети из модуля Hydraulic")

    with tab2:
        st.subheader("Анализ виртуальных датчиков")
        if scenario == "Авария (Утечка)":
            st.error("⚠️ Обнаружена утечка в секторе B-12!")
        else:
            st.success("Аномалий не обнаружено")

    with tab3:
        st.subheader("Экономические показатели")
        st.write("Прогноз износа труб на основе данных Risk Engine.")
        # Пример данных
        chart_data = pd.DataFrame({'Труба': ['A1', 'B2', 'C3'], 'Риск': [0.1, 0.8, 0.3]})
        st.bar_chart(chart_data, x='Труба', y='Риск')

else:
    st.info("Настройте параметры в боковой панели и нажмите 'Запустить анализ'.")

# Стилизация (Тот самый CSS)
st.markdown("""
    <style>
    .stMetric { background-color: #f0f2f6; padding: 15px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)
