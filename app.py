import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Импорт твоих модулей
from config import CONFIG
try:
    from risk_engine import DigitalTwinEngine
except ImportError:
    st.error("Ошибка: Модули логики (risk_engine.py и др.) не найдены в папке.")

# --- КОНФИГУРАЦИЯ СТРАНИЦЫ ---
st.set_page_config(
    page_title="Smart Shygyn | Digital Twin",
    page_icon="💧",
    layout="wide"
)

# --- ГЛОБАЛЬНАЯ СТИЛИЗАЦИЯ (через CONFIG) ---
st.markdown(f"""
    <style>
    .stApp {{ background-color: #f8fafc; }}
    .stButton>button {{
        background-color: {CONFIG.PRIMARY_COLOR};
        color: white;
        border-radius: 8px;
        border: none;
    }}
    .stMetric {{
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }}
    [data-testid="stSidebar"] {{
        background-color: white;
        border-right: 1px solid #e2e8f0;
    }}
    </style>
    """, unsafe_allow_html=True)

# --- ИНИЦИАЛИЗАЦИЯ СОСТОЯНИЯ ---
if 'analysis' not in st.session_state:
    st.session_state.analysis = None

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/water-pipe.png", width=60)
    st.title("Smart Shygyn")
    st.caption("v3.0 Digital Twin Orchestrator")
    
    with st.expander("🌍 Локация и Среда", expanded=True):
        city = st.selectbox("Город", ["Астана", "Алматы", "Туркестан"])
        temp = st.slider("Температура почвы (°C)", -30, 40, 10)
        map_style = st.selectbox("Стиль карты", list(CONFIG.MAP_TILE_OPTIONS.keys()))
    
    with st.expander("🏗️ Параметры сети", expanded=True):
        material = st.selectbox("Материал труб", ["Пластик (ПНД)", "Сталь", "Чугун"])
        age = st.slider("Возраст труб (лет)", 0, 60, 25)
        grid_size = st.number_input("Размер сетки (N x N)", 2, 10, CONFIG.DEFAULT_GRID_SIZE)
    
    if st.button("🚀 ЗАПУСТИТЬ АНАЛИЗ", use_container_width=True):
        with st.spinner("Синхронизация данных..."):
            twin = DigitalTwinEngine(city=city, season_temp_celsius=temp, material=material, pipe_age=age)
            st.session_state.analysis = twin.run_complete_analysis(grid_size=grid_size, leak_node="N_2_2")
            st.toast("Анализ завершен!", icon="✅")

# --- ГЛАВНЫЙ ЭКРАН ---
if st.session_state.analysis is None:
    st.markdown(f"""
    ## 👋 Система готова к работе
    Настройте параметры сети в левой панели и нажмите **Запустить анализ**.
    
    **Текущие настройки по умолчанию:**
    - Размер сетки: `{CONFIG.DEFAULT_GRID_SIZE}x{CONFIG.DEFAULT_GRID_SIZE}`
    - Охват сенсоров: `{CONFIG.DEFAULT_SENSOR_COVERAGE*100}%`
    - Формат экспорта: `{CONFIG.EXPORT_FORMAT}`
    """)
    st.image("https://images.unsplash.com/photo-1581094794329-c8112a89af12?q=80&w=1000", caption="Цифровой двойник Smart Shygyn")
else:
    res = st.session_state.analysis
    
    # 1. МЕТРИКИ (Используем цвета из CONFIG)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Статус", "ONLINE", delta="Stable", delta_color="normal")
    
    leak_detected = res.leak_detection.leak_detected
    m2.metric("Утечки", "КРИТИЧНО" if leak_detected else "НЕТ", 
              delta="-15% давление" if leak_detected else None, 
              delta_color="inverse")
    
    m3.metric("Хлор (Residual)", f"{res.water_quality.chlorine_residual_mg_l} mg/L")
    m4.metric("Compliance", f"{res.water_quality.compliance_percentage}%")

    # 2. ВКЛАДКИ
    tab_hyd, tab_qual, tab_risk = st.tabs(["💧 Гидравлика", "🧪 Качество воды", "⚖️ Риски и Ремонты"])
    
    with tab_hyd:
        col_map, col_data = st.columns([2, 1])
        with col_map:
            st.write(f"**Топология сети (Подложка: {CONFIG.MAP_TILE_OPTIONS[map_style]})**")
            # Заглушка для карты
            st.info("Здесь отрисовывается граф NetworkX с цветовой индикацией давления.")
            # Пример графика Plotly с цветами из конфига
            fig_p = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = 3.5,
                title = {'text': "Давление в узле утечки (bar)"},
                gauge = {'axis': {'range': [0, 6]}, 'bar': {'color': CONFIG.ACCENT_COLOR}}
            ))
            st.plotly_chart(fig_p, use_container_width=True)
        
        with col_data:
            st.write("**Детали утечки**")
            if leak_detected:
                st.error(f"Тип: {res.leak_detection.leak_type}")
                st.write(f"Локация: {res.leak_detection.predicted_location}")
                st.write(f"Потери: {res.leak_detection.estimated_flow_lps} л/с")
            else:
                st.success("Утечек не обнаружено")

    with tab_qual:
        st.write("**Прогноз дезинфекции (Хлор)**")
        # График распада хлора с использованием DANGER_COLOR
        x_age = np.linspace(0, 48, 100)
        y_cl = 0.5 * np.exp(-0.05 * x_age)
        fig_cl = px.line(x=x_age, y=y_cl, labels={'x':'Возраст воды (ч)', 'y':'Хлор (мг/л)'})
        fig_cl.add_hline(y=0.2, line_dash="dash", line_color=CONFIG.DANGER_COLOR, annotation_text="Минимум РК")
        fig_cl.update_traces(line_color=CONFIG.SECONDARY_COLOR)
        st.plotly_chart(fig_cl, use_container_width=True)

    with tab_risk:
        st.write("**Приоритеты технического обслуживания**")
        prio_df = pd.DataFrame(res.criticality_assessment.maintenance_priorities)
        st.dataframe(prio_df.style.highlight_max(axis=0, color=CONFIG.WARNING_COLOR), use_container_width=True)
        
        st.subheader("💡 Рекомендации ИИ")
        for rec in res.recommendations:
            st.info(rec)

    # 3. ЭКСПОРТ (Используем настройки из CONFIG)
    st.divider()
    if st.button(f"📥 Экспортировать отчет ({CONFIG.EXPORT_FORMAT}, {CONFIG.EXPORT_DPI} DPI)"):
        st.download_button("Скачать PDF", data="dummy_data", file_name=f"Report_{city}.pdf")

# FOOTER
st.markdown(f"""
    <div style="text-align: center; color: #94a3b8; padding: 20px;">
        Smart Shygyn &copy; 2026 | Powered by Digital Twin Engine | Theme: <span style="color:{CONFIG.PRIMARY_COLOR}">Corporate Blue</span>
    </div>
    """, unsafe_allow_html=True)
