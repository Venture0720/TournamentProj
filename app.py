"""
Smart Shygyn PRO v3 — Main Application
Центральный модуль управления интерфейсом и интеграции с гидравлическим движком.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import networkx as nx

# Импорт наших модулей
from config import CONFIG
from backend import CityManager, HydraulicPhysics

# ==========================================
# 1. ФУНКЦИИ ВИЗУАЛИЗАЦИИ (Оптимизировано)
# ==========================================

@st.cache_data(ttl=CONFIG.CACHE_TTL)
def render_hydraulic_charts(df, threshold_bar, smart_pump, dark_mode):
    bg = "#0F172A" if dark_mode else "#F8FAFC"
    fg = "#F1F5F9" if dark_mode else "#0F172A"
    
    fig = make_subplots(
        rows=3, cols=1, 
        subplot_titles=("💧 Давление (bar)", "🌊 Расход (L/s)", "⏱ Возраст воды (h)"),
        vertical_spacing=0.1,
        shared_xaxes=True
    )
    
    # График давления
    fig.add_trace(go.Scatter(x=df["Hour"], y=df["Pressure"], name="Давление",
                             line=dict(color=CONFIG.ACCENT_COLOR, width=3), fill='tozeroy'), row=1, col=1)
    fig.add_hline(y=threshold_bar, line_dash="dash", line_color=CONFIG.DANGER_COLOR, row=1, col=1)

    # График расхода
    fig.add_trace(go.Scatter(x=df["Hour"], y=df["Flow"], name="Расход",
                             line=dict(color=CONFIG.SUCCESS_COLOR, width=3)), row=2, col=1)

    # График возраста воды
    fig.add_trace(go.Scatter(x=df["Hour"], y=df["WaterAge"], name="Возраст",
                             line=dict(color="#A855F7", width=3)), row=3, col=1)

    fig.update_layout(height=800, template="plotly_dark" if dark_mode else "plotly_white",
                      paper_bgcolor=bg, plot_bgcolor=bg, font=dict(color=fg))
    return fig

@st.cache_data(ttl=CONFIG.CACHE_TTL)
def render_nrw_pie(nrw_val, dark_mode):
    bg = "#0F172A" if dark_mode else "#F8FAFC"
    fig = go.Figure(data=[go.Pie(labels=['Revenue Water', 'Losses (NRW)'], 
                                 values=[100-nrw_val, nrw_val], hole=.6,
                                 marker_colors=[CONFIG.SUCCESS_COLOR, CONFIG.DANGER_COLOR])])
    fig.update_layout(height=350, margin=dict(t=30, b=30, l=0, r=0), paper_bgcolor=bg)
    return fig

# ==========================================
# 2. ОСНОВНАЯ ЛОГИКА ПРИЛОЖЕНИЯ
# ==========================================

def main():
    # Настройка страницы ДОЛЖНА быть первой
    st.set_page_config(
        page_title="Smart Shygyn PRO v3",
        page_icon="🌊",
        layout="wide"
    )

    # Применяем кастомный CSS (Glassmorphism)
    st.markdown(f"""
        <style>
        .stApp {{ background-color: #0F172A; color: white; }}
        [data-testid="stMetricValue"] {{ color: {CONFIG.ACCENT_COLOR}; font-size: 32px; }}
        </style>
    """, unsafe_allow_html=True)

    # --- SIDEBAR ---
    st.sidebar.title("🎮 Управление")
    city_name = st.sidebar.selectbox("Выберите город", ["Алматы", "Астана", "Туркестан"])
    dark_mode = st.sidebar.toggle("Темная тема", value=True)
    
    st.sidebar.divider()
    
    pipe_age = st.sidebar.slider("Возраст труб (лет)", 0, 100, 25)
    leak_threshold = st.sidebar.slider("Порог утечки (bar)", 1.0, 5.0, 2.7)
    smart_pump = st.sidebar.checkbox("Smart Pump Scheduling", value=True)

    # --- ИНИЦИАЛИЗАЦИЯ БЭКЕНДА ---
    city_mgr = CityManager(city_name)
    physics = HydraulicPhysics()
    
    # Генерация данных (в будущем здесь будет вызов тяжелой симуляции WNTR)
    hours = np.linspace(0, 24, 25)
    mock_data = pd.DataFrame({
        "Hour": hours,
        "Pressure": 3.2 - (pipe_age * 0.01) + 0.3 * np.sin(hours/3),
        "Flow": 120 + 40 * np.abs(np.cos(hours/6)),
        "WaterAge": 2 + 0.5 * hours
    })

    # --- MAIN UI ---
    st.title(f"🏙 Цифровой двойник: {city_name}")
    st.info(city_mgr.config.description)

    # Метрики верхнего уровня
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Риск прорыва", f"{15 + (pipe_age * 0.5):.1f}%", delta="High" if pipe_age > 40 else "Normal")
    m2.metric("Потери воды (NRW)", f"{22 + (pipe_age * 0.2):.1f}%")
    m3.metric("Энергоэффективность", "88%", delta="12%")
    m4.metric("Давление в сети", f"{mock_data['Pressure'].mean():.2f} bar")

    st.divider()

    # Сетка: Карта и Графики
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("📡 Гидравлическая диагностика")
        fig = render_hydraulic_charts(mock_data, leak_threshold, smart_pump, dark_mode)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("💰 Экономический анализ")
        
        # График потерь
        loss_val = 22 + (pipe_age * 0.2)
        st.plotly_chart(render_nrw_pie(loss_val, dark_mode), use_container_width=True)
        
        # Окупаемость
        st.markdown(f"""
            <div style="background: rgba(255,255,255,0.05); padding: 20px; border-radius: 15px; border-left: 5px solid {CONFIG.SUCCESS_COLOR};">
                <h4>ROI Прогноз</h4>
                <p>При текущем износе труб ({pipe_age} лет), внедрение Smart Shygyn окупится за <b>14.2 месяца</b>.</p>
                <small>Экономия: ~1.2 млн ₸ / мес</small>
            </div>
        """, unsafe_allow_html=True)

    # Футер
    st.caption(f"Smart Shygyn PRO v3.0 | Система активна | Температура грунта: {city_mgr.config.ground_temp_celsius}°C")

if __name__ == "__main__":
    main()
