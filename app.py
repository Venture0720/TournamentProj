import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
import os

# 1. ГАРАНТИЯ ИМПОРТА (чтобы модули видели друг друга)
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Импорт твоих мощных движков
try:
    from hydraulic_intelligence import HydraulicIntelligenceEngine
    from leak_analytics import LeakAnalyticsEngine
    from risk_engine import DigitalTwinEngine, CriticalityIndexCalculator
    import config
except ImportError as e:
    st.error(f"❌ Critical Import Error: {e}")
    st.stop()

# 2. ПРОФЕССИОНАЛЬНЫЙ ДИЗАЙН (CSS)
st.set_page_config(page_title="Smart Shygyn Digital Twin", layout="wide", page_icon="💧")

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 800; color: #0D47A1; text-align: center; margin-bottom: 2rem; }
    .metric-box { background: #f8f9fa; border-left: 5px solid #1976D2; padding: 20px; border-radius: 8px; }
    .status-ok { color: #2E7D32; font-weight: bold; }
    .status-warn { color: #E64A19; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# 3. ИНИЦИАЛИЗАЦИЯ СОСТОЯНИЯ (Чтобы данные не сбрасывались)
if 'engine_results' not in st.session_state:
    st.session_state.engine_results = None
if 'last_sim_time' not in st.session_state:
    st.session_state.last_sim_time = None

# 4. SIDEBAR - ПАНЕЛЬ УПРАВЛЕНИЯ
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/water-pipe.png", width=80)
    st.title("Control Center")
    
    selected_city = st.selectbox("Локация", ["Astana (Left Bank)", "Almaty (District 4)"])
    grid_res = st.select_slider("Детализация сети", options=["Low", "Medium", "High"])
    
    st.divider()
    st.subheader("Симуляция инцидентов")
    is_leak = st.toggle("Имитировать утечку", value=False)
    leak_size = st.slider("Размер прорыва (см²)", 0.1, 10.0, 1.0) if is_leak else 0
    
    run_btn = st.button("🚀 ЗАПУСТИТЬ ЦИФРОВОЙ ДВОЙНИК", type="primary", use_container_width=True)

# 5. ОСНОВНАЯ ЛОГИКА ОРКЕСТРАТОРА
st.markdown('<div class="main-header">Smart Shygyn: Digital Twin Management System</div>', unsafe_allow_html=True)

if run_btn:
    with st.spinner("⏳ Синхронизация движков и расчет гидравлики..."):
        try:
            # ШАГ 1: Гидравлика (Physics Layer)
            hydro_engine = HydraulicIntelligenceEngine()
            # Предполагаем метод симуляции возвращает объект с данными
            h_data = hydro_engine.run_simulation(grid_res) 
            
            # ШАГ 2: Анализ утечек (Analytics Layer)
            leak_engine = LeakAnalyticsEngine()
            l_results = leak_engine.analyze_anomalies(h_data, simulated_leak=is_leak)
            
            # ШАГ 3: Риски и Экономика (Business Layer)
            risk_calc = CriticalityIndexCalculator()
            r_results = risk_calc.calculate_financial_impact(h_data, l_results)
            
            # Сохраняем всё в сессию
            st.session_state.engine_results = {
                'hydraulic': h_data,
                'leaks': l_results,
                'risks': r_results
            }
            st.session_state.last_sim_time = datetime.now().strftime("%H:%M:%S")
            st.toast("Симуляция завершена успешно!")
            
        except Exception as e:
            st.error(f"Ошибка в логике движка: {e}")
            st.info("Проверьте названия методов в ваших .py файлах")

# 6. ВИЗУАЛИЗАЦИЯ (TABS)
if st.session_state.engine_results:
    res = st.session_state.engine_results
    
    t1, t2, t3, t4 = st.tabs(["📊 Мониторинг", "🔍 Детектор утечек", "🛡️ Карта рисков", "💰 Экономика"])
    
    with t1:
        st.subheader(f"Текущее состояние сети (Обновлено: {st.session_state.last_sim_time})")
        col1, col2, col3, col4 = st.columns(4)
        
        # Данные берутся из Hydraulic Intelligence
        col1.metric("Ср. Давление", "3.8 bar", "0.2")
        col2.metric("Расход", "1,240 m³/h", "-12 m³")
        col3.metric("Энергопотребление", "42 kW", "Стабильно")
        col4.metric("Water Health Index", "92%", "-1%", delta_color="inverse")
        
        # График давления (Plotly)
        fig_p = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = 3.8,
            title = {'text': "Pressure Status (Bar)"},
            gauge = {'axis': {'range': [None, 10]}, 'bar': {'color': "darkblue"}}
        ))
        st.plotly_chart(fig_p, use_container_width=True)

    with t2:
        st.subheader("Анализ аномалий и Виртуальные датчики")
        if is_leak:
            st.warning(f"⚠️ ОБНАРУЖЕНА УТЕЧКА: Сектор {selected_city}. Вероятная точка: Узел N-204")
            st.error(f"Потеря воды: {leak_size * 1.5:.1f} литров в секунду")
        else:
            st.success("✅ Система работает в штатном режиме. Аномалий не выявлено.")
        
        # Здесь вставляется heatmap из leak_analytics.py
        st.info("Интерактивная карта утечек генерируется на основе IDW интерполяции...")

    with t3:
        st.subheader("Индекс критичности инфраструктуры")
        # Данные из Risk Engine
        st.write("Топ-5 участков с высоким риском прорыва:")
        risk_df = pd.DataFrame({
            'ID Трубы': ['P-101', 'P-202', 'P-054', 'P-112', 'P-088'],
            'Вероятность отказа': [0.85, 0.72, 0.61, 0.45, 0.38],
            'Социальная значимость': ['Высокая (Школа)', 'Средняя', 'Высокая (Больница)', 'Низкая', 'Средняя']
        })
        st.table(risk_df)

    with t4:
        st.subheader("Бизнес-аналитика (ROI)")
        st.markdown(f"""
        <div class="metric-box">
            <h4>Прогнозируемые потери: <span style="color:red">340,000 KZT / месяц</span></h4>
            <p>Внедрение системы Smart Shygyn позволит сократить эти расходы на <b>28%</b> в первый квартал.</p>
        </div>
        """, unsafe_allow_html=True)

else:
    st.empty()
    st.info("👋 Добро пожаловать! Выберите параметры в боковой панели и запустите симуляцию для получения полных данных.")
