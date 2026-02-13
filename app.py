import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from datetime import datetime

# Импорт твоих модулей
try:
    from risk_engine import DigitalTwinEngine, SocialImpactFactors
except ImportError:
    st.error("Ошибка: Файлы модулей (risk_engine.py и др.) не найдены в папке.")

# --- КОНФИГУРАЦИЯ СТРАНИЦЫ ---
st.set_page_config(
    page_title="Smart Shygyn | Digital Twin",
    page_icon="💧",
    layout="wide"
)

# --- ИНИЦИАЛИЗАЦИЯ ДВИЖКА ---
if 'twin' not in st.session_state:
    st.session_state.twin = None

# --- СТИЛИЗАЦИЯ ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR: УПРАВЛЕНИЕ ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/water-pipe.png", width=80)
    st.title("Smart Shygyn v3.0")
    st.subheader("Параметры системы")
    
    city = st.selectbox("Регион (Казахстан)", ["Астана", "Алматы", "Туркестан"])
    material = st.selectbox("Материал магистрали", ["Пластик (ПНД)", "Сталь", "Чугун"])
    age = st.slider("Возраст труб (лет)", 0, 60, 25)
    temp = st.slider("Температура среды (°C)", -30, 40, 10)
    
    st.divider()
    st.subheader("Симуляция инцидентов")
    grid_size = st.slider("Размер сетки (N x N)", 2, 6, 4)
    leak_enabled = st.toggle("Активировать утечку", value=True)
    leak_node = st.text_input("Узел утечки (напр. N_2_2)", "N_2_2")
    
    if st.button("🚀 ЗАПУСТИТЬ ЦИФРОВОЙ ДВОЙНИК", use_container_width=True):
        st.session_state.twin = DigitalTwinEngine(
            city=city, 
            season_temp_celsius=temp, 
            material=material, 
            pipe_age=age
        )
        # Запуск анализа
        st.session_state.analysis = st.session_state.twin.run_complete_analysis(
            grid_size=grid_size,
            leak_node=leak_node if leak_enabled else None
        )

# --- ГЛАВНАЯ ПАНЕЛЬ ---
if st.session_state.twin is None:
    st.info("👋 Добро пожаловать! Настройте параметры в боковой панели и нажмите 'Запустить Цифровой Двойник' для начала анализа сети.")
    st.image("https://images.unsplash.com/photo-1581094794329-c8112a89af12?auto=format&fit=crop&q=80&w=1000", caption="Digital Twin Engine для водоканалов РК")
else:
    res = st.session_state.analysis
    
    # 1. МЕТРИКИ ВЕРХНЕГО УРОВНЯ
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Статус системы", res.status, delta=None)
    with col2:
        leak_status = "⚠️ ОБНАРУЖЕНО" if res.leak_detection.leak_detected else "✅ НОРМА"
        st.metric("Детектор утечек", leak_status)
    with col3:
        st.metric("Качество воды", res.water_quality.quality_standard)
    with col4:
        st.metric("Compliance (РК)", f"{res.water_quality.compliance_percentage}%")

    # 2. ВИЗУАЛИЗАЦИЯ СЕТИ (ГРАФ)
    st.subheader("Интерактивная топология сети")
    
    # Генерация визуализации графа через Plotly
    fig = go.Figure()
    # (Здесь логика отрисовки узлов и ребер из res.network_topology)
    # Для краткости выводим уведомление о зонах риска
    st.info(f"Анализ завершен для {res.city}. Обнаружено узлов: {res.water_quality.avg_age_hours} ч. (средний возраст воды).")
    
    # 3. АНАЛИТИЧЕСКИЕ ВКЛАДКИ
    tab1, tab2, tab3, tab4 = st.tabs(["💧 Гидравлика & Утечки", "🧪 Качество & Хлор", "⚖️ Риски & Критичность", "📄 Отчет API"])
    
    with tab1:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.write("**Анализ ночного потока (MNF):**")
            # Симуляция графика MNF
            chart_data = pd.DataFrame(np.random.normal(0.4, 0.05, size=(24, 1)), columns=['Flow (L/s)'])
            if res.leak_detection.leak_detected:
                chart_data.iloc[2:6] += res.leak_detection.estimated_flow_lps
            st.line_chart(chart_data)
        with c2:
            st.json(res.leak_detection.mnf_analysis)
            st.metric("Эст. поток утечки", f"{res.leak_detection.estimated_flow_lps} L/s")

    with tab2:
        col_a, col_b = st.columns(2)
        with col_a:
            st.write("**Распад хлора (First-order decay):**")
            # Визуализация твоей формулы из Part 3
            time_axis = np.linspace(0, 48, 100)
            chlorine = 0.5 * np.exp(-0.05 * time_axis)
            fig_cl = px.line(x=time_axis, y=chlorine, labels={'x':'Часы', 'y':'Cl (mg/L)'}, title="Прогноз дезинфекции")
            fig_cl.add_hline(y=0.2, line_dash="dash", line_color="red", annotation_text="Стандарт РК (0.2)")
            st.plotly_chart(fig_cl, use_container_width=True)
        with col_b:
            st.write("**Зоны застоя:**")
            st.table(res.water_quality.stagnation_zones)

    with tab3:
        st.write("**План приоритетного обслуживания:**")
        crit_df = pd.DataFrame(res.criticality_assessment.maintenance_priorities)
        if not crit_df.empty:
            st.dataframe(crit_df[['node', 'criticality_index', 'risk_class', 'priority']], use_container_width=True)
        
        st.write("**Рекомендация системы:**")
        for rec in res.recommendations:
            st.success(f"💡 {rec}")

    with tab4:
        st.write("Сгенерированный API Response (JSON):")
        st.json(res.to_dict())

    # 4. АЛЕРТЫ
    if res.alerts:
        st.sidebar.divider()
        st.sidebar.subheader("🔔 Уведомления")
        for alert in res.alerts:
            if alert['level'] == "CRITICAL":
                st.sidebar.error(f"{alert['message']} (Узел: {alert['node']})")
            else:
                st.sidebar.warning(alert['message'])

# --- FOOTER ---
st.divider()
st.caption(f"Smart Shygyn Digital Twin Core | API v3.0.0 | {datetime.now().year} Astana Hub Competition")
