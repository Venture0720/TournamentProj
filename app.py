import streamlit as st
import pandas as pd
import numpy as np
import wntr
import requests
import random
import plotly.express as px
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime

# --- 1. ФУНКЦИИ (Backend) ---

def run_epanet_simulation(material_c, degradation, sampling_rate):
    wn = wntr.network.WaterNetworkModel()
    dist = 100
    actual_diameter = 0.2 * (1 - degradation / 100)
    
    for i in range(4):
        for j in range(4):
            name = f"N_{i}_{j}"
            wn.add_junction(name, base_demand=0.001, elevation=10)
            wn.get_node(name).coordinates = (i * dist, j * dist)
            if i > 0:
                wn.add_pipe(f"PH_{i}_{j}", f"N_{i-1}_{j}", name, 
                            length=dist, diameter=actual_diameter, roughness=material_c)
            if j > 0:
                wn.add_pipe(f"PV_{i}_{j}", f"N_{i}_{j-1}", name, 
                            length=dist, diameter=actual_diameter, roughness=material_c)

    wn.add_reservoir('Res', base_head=40)
    wn.get_node('Res').coordinates = (-dist, -dist)
    wn.add_pipe('P_Main', 'Res', 'N_0_0', length=dist, diameter=0.4, roughness=material_c)

    leak_node = "N_2_2"
    st.session_state['leak_node'] = leak_node
    wn.options.time.duration = 24 * 3600
    wn.options.time.report_timestep = 3600 // sampling_rate
    
    node = wn.get_node(leak_node)
    node.add_leak(wn, area=0.08, start_time=12 * 3600)
    
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()
    
    p = results.node['pressure'][leak_node] * 0.1 
    f = results.link['flowrate']['P_Main'] * 1000 
    
    # Генерация "живого" шума
    noise_p = np.random.normal(0, 0.04, len(p))
    noise_f = np.random.normal(0, 0.08, len(f))
    
    df_res = pd.DataFrame({
        'Hour': np.arange(len(p)) / sampling_rate,
        'Pressure (bar)': p.values + noise_p,
        'Flow Rate (L/s)': np.abs(f.values) + noise_f
    }).set_index('Hour')
    
    return df_res, wn

# --- 2. ИНТЕРФЕЙС ---
st.set_page_config(page_title="Smart Shygyn PRO", layout="wide", page_icon="💧")

# Кастомный CSS для красоты
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.sidebar.title("🧪 Инженерная лаборатория")
with st.sidebar.expander("⚙️ Параметры сети", expanded=True):
    m_types = {"Пластик (ПНД)": 150, "Сталь": 140, "Чугун": 100}
    material = st.selectbox("Материал труб", list(m_types.keys()))
    iznos = st.slider("Износ системы (%)", 0, 60, 15)
    freq = st.select_slider("Частота датчиков (Гц)", options=[1, 2, 4])

with st.sidebar.expander("💸 Экономика и ПОИ", expanded=True):
    price = st.number_input("Тариф за литр (тг)", value=0.55)
    limit = st.slider("Порог детекции (Bar)", 1.0, 5.0, 2.7)

if st.sidebar.button("🚀 ОБНОВИТЬ ЦИФРОВОЙ ДВОЙНИК", use_container_width=True):
    data, net = run_epanet_simulation(m_types[material], iznos, freq)
    st.session_state['data'] = data
    st.session_state['network'] = net
    st.session_state['log'] = f"[{datetime.now().strftime('%H:%M:%S')}] Модель пересчитана. Материал: {material}, Износ: {iznos}%"

# --- 3. ГЛАВНЫЙ ЭКРАН ---
st.title("💧 Smart Shygyn: AI Water Management")

if st.session_state.get('data') is not None:
    df = st.session_state['data']
    wn = st.session_state['network']
    df['Leak'] = df['Pressure (bar)'] < limit
    active_leak = df['Leak'].any()

    # СИСТЕМА KPI
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Текущий статус", "🚨 КРИТИЧЕСКИ" if active_leak else "✅ СТАБИЛЬНО")
    c2.metric("Мин. Давление", f"{df['Pressure (bar)'].min():.2f} Bar")
    
    lost_l = df[df['Leak'] == True]['Flow Rate (L/s)'].sum() * (3600 / freq) if active_leak else 0
    c3.metric("Потери (литры)", f"{lost_l:,.0f} L")
    c4.metric("Ущерб (тенге)", f"{lost_l * price:,.0f} ₸")

    t1, t2, t3 = st.tabs(["📊 Аналитический дашборд", "🗺 Карта инцидентов", "🧾 Отчетность"])

    with t1:
        # Продвинутый график Plotly
        fig = px.line(df, y=['Pressure (bar)', 'Flow Rate (L/s)'], 
                     title="Осциллограмма гидравлических параметров",
                     color_discrete_map={"Pressure (bar)": "#3498db", "Flow Rate (L/s)": "#e67e22"})
        fig.add_hline(y=limit, line_dash="dash", line_color="red", annotation_text="Порог детекции")
        st.plotly_chart(fig, use_container_width=True)
        
        if st.session_state.get('log'):
            st.code(st.session_state['log'])

    with t2:
        col_left, col_right = st.columns([2, 1])
        with col_left:
            fig_map, ax = plt.subplots(figsize=(10, 7))
            pos = {n: wn.get_node(n).coordinates for n in wn.node_name_list}
            l_node = st.session_state['leak_node']
            
            n_colors = ['#e74c3c' if (n == l_node and active_leak) else '#3498db' if n == 'Res' else '#2ecc71' for n in wn.node_name_list]
            
            nx.draw_networkx_edges(wn.get_graph(), pos, ax=ax, edge_color='#bdc3c7', width=2)
            nx.draw_networkx_nodes(wn.get_graph(), pos, ax=ax, node_color=n_colors, node_size=400, edgecolors='white')
            nx.draw_networkx_labels(wn.get_graph(), pos, ax=ax, font_size=9, font_color='black')
            ax.set_axis_off()
            st.pyplot(fig_map)
        
        with col_right:
            st.info("💡 **Анализ топологии:**")
            st.write(f"- Резервуар: **Напор стабилен**")
            st.write(f"- Точка утечки: **{l_node if active_leak else 'Не обнаружена'}**")
            st.write(f"- Рекомендация: **{'Срочный выезд бригады' if active_leak else 'Плановый осмотр'}**")

    with t3:
        st.subheader("Экспорт данных для акимата/ЖКХ")
        st.dataframe(df)
        st.download_button("📩 Сформировать отчет (CSV)", df.to_csv(), "smart_shygyn_report.csv", use_container_width=True)

else:
    st.info("👋 Добро пожаловать! Настройте инженерные параметры слева и нажмите 'Запустить расчет' для начала мониторинга.")
