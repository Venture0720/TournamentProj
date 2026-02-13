"""
Smart Shygyn PRO v3 — FRONTEND VISUALIZATION
Интерактивный дашборд для мониторинга систем водоснабжения.
Полная версия: Графики, Карта, Экономика.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster, Fullscreen, LocateControl

# Импорт конфигурации и бэкенда
# Убедитесь, что файлы config.py и backend.py лежат рядом
try:
    from config import CONFIG
    from backend import CityManager, HydraulicPhysics
except ImportError as e:
    st.error(f"Ошибка импорта: {e}. Убедитесь, что файлы config.py и backend.py существуют.")
    st.stop()

# ==============================================================================
# 1. ФУНКЦИИ ОТРИСОВКИ (VISUALIZATION ENGINE)
# ==============================================================================

@st.cache_data(ttl=CONFIG.CACHE_TTL)
def create_hydraulic_plot(df: pd.DataFrame, threshold_bar: float, smart_pump: bool, dark_mode: bool) -> go.Figure:
    """Создает сложный интерактивный график гидравлики."""
    bg = "#0F172A" if dark_mode else "#F8FAFC"
    fg = "#F1F5F9" if dark_mode else "#0F172A"
    grid_c = "#334155" if dark_mode else "#E2E8F0"
    
    rows = 4 if smart_pump else 3
    row_heights = [0.3, 0.3, 0.2, 0.2] if smart_pump else [0.35, 0.35, 0.30]
    titles = ["💧 Давление (bar)", "🌊 Расход (L/s)", "⏱ Возраст воды (h)"]
    if smart_pump: titles.append("⚡ Напор насоса (m)")

    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=row_heights, subplot_titles=titles)

    # 1. Давление
    fig.add_trace(go.Scatter(x=df["Hour"], y=df["Pressure"], name="Давление",
                             line=dict(color=CONFIG.ACCENT_COLOR, width=3), fill='tozeroy', 
                             fillcolor="rgba(59, 130, 246, 0.1)"), row=1, col=1)
    fig.add_hline(y=threshold_bar, line_dash="dash", line_color=CONFIG.DANGER_COLOR, annotation_text="Порог утечки", row=1, col=1)

    # 2. Расход
    fig.add_trace(go.Scatter(x=df["Hour"], y=df["Flow"], name="Расход",
                             line=dict(color=CONFIG.SUCCESS_COLOR, width=3)), row=2, col=1)
    # Ожидаемый расход (пунктир)
    fig.add_trace(go.Scatter(x=df["Hour"], y=df["Flow"]*0.8, name="Норма",
                             line=dict(color="gray", width=1, dash="dot"), opacity=0.7), row=2, col=1)

    # 3. Возраст воды
    fig.add_trace(go.Scatter(x=df["Hour"], y=df["WaterAge"], name="Возраст",
                             line=dict(color=CONFIG.INFO_COLOR, width=2), fill='tozeroy'), row=3, col=1)

    # 4. Насос
    if smart_pump:
        fig.add_trace(go.Scatter(x=df["Hour"], y=df["PumpHead"], name="Насос",
                                 line=dict(color=CONFIG.WARNING_COLOR, width=2, shape='hv')), row=4, col=1)

    fig.update_layout(height=900 if smart_pump else 700, 
                      template="plotly_dark" if dark_mode else "plotly_white",
                      paper_bgcolor=bg, plot_bgcolor=bg,
                      font=dict(color=fg, family="Inter"),
                      margin=dict(l=20, r=20, t=60, b=20),
                      hovermode="x unified")
    fig.update_xaxes(showgrid=True, gridcolor=grid_c)
    fig.update_yaxes(showgrid=True, gridcolor=grid_c)
    return fig

@st.cache_data(ttl=CONFIG.CACHE_TTL)
def create_payback_chart(economics: dict, dark_mode: bool) -> go.Figure:
    """График окупаемости ROI."""
    bg = "#0F172A" if dark_mode else "#F8FAFC"
    fg = "#F1F5F9" if dark_mode else "#0F172A"
    
    months = np.arange(0, 24)
    savings = months * economics["monthly_savings"]
    capex = np.full_like(months, economics["capex"])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=months, y=savings, name="Накопленная экономия", 
                             fill='tozeroy', line=dict(color=CONFIG.SUCCESS_COLOR)))
    fig.add_trace(go.Scatter(x=months, y=capex, name="Инвестиции (CAPEX)", 
                             line=dict(color=CONFIG.DANGER_COLOR, dash='dash')))
    
    # Точка безубыточности
    payback_m = economics["payback_months"]
    fig.add_vline(x=payback_m, line_dash="dot", annotation_text=f"Окупаемость: {payback_m:.1f} мес.")

    fig.update_layout(title="Финансовая модель (ROI)", height=350, 
                      paper_bgcolor=bg, plot_bgcolor=bg, font=dict(color=fg))
    return fig

@st.cache_data(ttl=CONFIG.CACHE_TTL)
def create_map(city_config, pipe_age, dark_mode):
    """Создает карту Folium."""
    tiles = CONFIG.MAP_TILE_OPTIONS["dark" if dark_mode else "light"]
    m = folium.Map(location=[city_config.lat, city_config.lng], 
                   zoom_start=city_config.zoom, tiles=tiles)
    
    Fullscreen().add_to(m)
    
    # Эмуляция узлов сети (для демонстрации)
    # В реальности данные берутся из wntr graph
    center_lat, center_lng = city_config.lat, city_config.lng
    
    # Рисуем "трубы" (случайная генерация для демо)
    for i in range(5):
        lat_offset = (np.random.random() - 0.5) * 0.01
        lng_offset = (np.random.random() - 0.5) * 0.01
        
        # Определяем риск на основе возраста труб
        risk_color = CONFIG.DANGER_COLOR if pipe_age > 40 else CONFIG.SUCCESS_COLOR
        if 20 < pipe_age <= 40: risk_color = CONFIG.WARNING_COLOR
            
        folium.PolyLine(
            locations=[[center_lat, center_lng], [center_lat + lat_offset, center_lng + lng_offset]],
            color=risk_color, weight=4, opacity=0.8, tooltip=f"Труба ID: {i}"
        ).add_to(m)
        
        folium.CircleMarker(
            location=[center_lat + lat_offset, center_lng + lng_offset],
            radius=6, color=risk_color, fill=True, popup=f"Узел {i}"
        ).add_to(m)

    # Резервуар
    folium.Marker(
        [center_lat, center_lng], 
        icon=folium.Icon(color="blue", icon="tint"),
        popup="Резервуар"
    ).add_to(m)
    
    return m

# ==============================================================================
# 2. ОСНОВНАЯ ЛОГИКА (MAIN APP)
# ==============================================================================

def main():
    # 1. Настройка страницы
    st.set_page_config(
        page_title="Smart Shygyn PRO v3",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # CSS Стиль
    st.markdown(f"""
        <style>
        .stApp {{ background-color: {CONFIG.PRIMARY_COLOR if True else '#ffffff'}; }}
        h1, h2, h3 {{ font-family: 'Inter', sans-serif; }}
        div[data-testid="stMetricValue"] {{ font-size: 28px; color: {CONFIG.ACCENT_COLOR}; }}
        </style>
    """, unsafe_allow_html=True)

    # 2. Сайдбар
    st.sidebar.image("https://img.icons8.com/fluency/96/water-pipe.png", width=60)
    st.sidebar.title("Smart Shygyn PRO")
    
    city_name = st.sidebar.selectbox("Город", ["Алматы", "Астана", "Туркестан"])
    dark_mode = st.sidebar.toggle("Dark Mode", value=True)
    
    st.sidebar.divider()
    
    st.sidebar.subheader("Параметры сети")
    pipe_age = st.sidebar.slider("Ср. возраст труб (лет)", 0, 80, 25)
    pressure_setpoint = st.sidebar.slider("Давление на выходе (bar)", 2.0, 6.0, 3.5)
    smart_pump = st.sidebar.checkbox("Smart Pump Control", value=True)
    
    # 3. Инициализация Бэкенда (Данные)
    city_mgr = CityManager(city_name)
    
    # Генерация демо-данных (чтобы графики были живыми)
    hours = np.linspace(0, 24, 48)
    base_pressure = pressure_setpoint - (pipe_age * 0.015) # Старые трубы снижают давление
    
    mock_df = pd.DataFrame({
        "Hour": hours,
        "Pressure": base_pressure + 0.5 * np.sin(hours) + np.random.normal(0, 0.05, 48),
        "Flow": 150 + 50 * np.sin(hours/2) + np.random.normal(0, 5, 48),
        "WaterAge": 2 + 0.1 * hours,
        "PumpHead": [60 if (h < 6 or h > 22) and smart_pump else 80 for h in hours]
    })
    
    # Расчет экономики (Demo)
    leak_rate = 15 + (pipe_age * 0.4) # % утечек
    monthly_loss_kzt = 500000 * (leak_rate / 100)
    capex_needed = 10000000
    
    economics = {
        "monthly_savings": monthly_loss_kzt * 0.8, # Экономим 80% потерь
        "capex": capex_needed,
        "payback_months": capex_needed / (monthly_loss_kzt * 0.8) if monthly_loss_kzt > 0 else 0
    }

    # 4. Основной экран
    st.title(f"🏙 Цифровой двойник: {city_name}")
    st.caption(f"Координаты: {city_mgr.config.lat}, {city_mgr.config.lng} | Температура грунта: {city_mgr.config.ground_temp_celsius}°C")
    
    # Метрики (KPI)
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Среднее давление", f"{mock_df['Pressure'].mean():.2f} bar", delta="-0.1")
    kpi2.metric("Уровень утечек (NRW)", f"{leak_rate:.1f}%", delta_color="inverse", delta=f"+{pipe_age*0.1:.1f}%")
    kpi3.metric("Прогноз прорывов", "Высокий" if pipe_age > 40 else "Низкий")
    kpi4.metric("Экономия (мес)", f"{economics['monthly_savings']/1000:.0f} тыс ₸")

    st.divider()

    # Вкладки интерфейса
    tab1, tab2, tab3 = st.tabs(["📊 Гидравлика", "🗺 Карта сети", "💰 Экономика"])

    with tab1:
        st.subheader("Динамика гидравлических параметров (24ч)")
        fig_hyd = create_hydraulic_plot(mock_df, 2.7, smart_pump, dark_mode)
        st.plotly_chart(fig_hyd, use_container_width=True)

    with tab2:
        st.subheader("Геоинформационная система")
        col_map, col_legend = st.columns([3, 1])
        with col_map:
            map_obj = create_map(city_mgr.config, pipe_age, dark_mode)
            st_folium(map_obj, height=500, use_container_width=True)
        with col_legend:
            st.info("Легенда карты")
            st.markdown(f"🔴 **Критический риск**: Трубы > 40 лет")
            st.markdown(f"🟠 **Высокий риск**: Трубы 20-40 лет")
            st.markdown(f"🟢 **Норма**: Трубы < 20 лет")

    with tab3:
        st.subheader("ROI и Инвестиционный анализ")
        col_roi, col_pie = st.columns(2)
        with col_roi:
            fig_roi = create_payback_chart(economics, dark_mode)
            st.plotly_chart(fig_roi, use_container_width=True)
        with col_pie:
            # Быстрый пи-чарт потерь
            fig_pie = go.Figure(go.Pie(
                labels=['Полезный отпуск', 'Потери'], 
                values=[100-leak_rate, leak_rate],
                marker_colors=[CONFIG.SUCCESS_COLOR, CONFIG.DANGER_COLOR],
                hole=0.6
            ))
            fig_pie.update_layout(title="Баланс воды", template="plotly_dark" if dark_mode else "plotly_white", paper_bgcolor="#0F172A" if dark_mode else "white")
            st.plotly_chart(fig_pie, use_container_width=True)

if __name__ == "__main__":
    main()
