import streamlit as st
import pandas as pd
import numpy as np
import wntr
import requests
import random
import plotly.express as px
import matplotlib.pyplot as plt
import networkx as nx

# --- 1. ФУНКЦИИ (Backend) ---

def run_epanet_simulation(material_c, degradation, sampling_rate):
    wn = wntr.network.WaterNetworkModel()
    dist = 100
    
    # Физический расчет диаметра с учетом износа
    actual_diameter = 0.2 * (1 - degradation / 100)
    
    # Создаем сетку узлов
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
    
    # Включаем динамику времени (24 часа)
    wn.options.time.duration = 24 * 3600
    wn.options.time.report_timestep = 3600 // sampling_rate
    
    # Добавляем утечку в середине дня
    node = wn.get_node(leak_node)
    node.add_leak(wn, area=0.08, start_time=12 * 3600)
    
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()
    
    # Извлекаем давление и поток
    p = results.node['pressure'][leak_node] * 0.1 # Перевод в Bar
    f = results.link['flowrate']['P_Main'] * 1000 # Перевод в л/с
    
    # Добавляем "Живой шум" для датчиков (чтобы графики не были мертвыми)
    noise_p = np.random.normal(0, 0.05, len(p))
    noise_f = np.random.normal(0, 0.1, len(f))
    
    df_res = pd.DataFrame({
        'Time (h)': np.arange(len(p)) / sampling_rate,
        'Pressure (bar)': p.values + noise_p,
        'Flow Rate (L/s)': np.abs(f.values) + noise_f
    }).set_index('Time (h)')
    
    return df_res, wn

# --- 2. ИНТЕРФЕЙС И НАСТРОЙКИ ---
st.set_page_config(page_title="Smart Shygyn PRO", layout="wide")
st.sidebar.title("🛠 Инженерная панель")

materials = {"Пластик (ПНД)": 150, "Сталь": 140, "Чугун (старый)": 100}
selected_material = st.sidebar.selectbox("Материал труб:", list(materials.keys()))
degradation = st.sidebar.slider("Износ сети (%):", 0, 50, 10)
sampling = st.sidebar.select_slider("Частота (опросов/час):", options=[1, 2, 4])
tariff = st.sidebar.number_input("Тариф (тг/литр):", value=0.5)
threshold = st.sidebar.slider("Порог тревоги (Bar):", 1.0, 5.0, 2.8)

if 'data' not in st.session_state:
    st.session_state['data'] = None

if st.sidebar.button("🚀 Запустить расчет"):
    with st.spinner('Симуляция гидравлики...'):
        data, network = run_epanet_simulation(materials[selected_material], degradation, sampling)
        st.session_state['data'] = data
        st.session_state['network'] = network

# --- 3. ГЛАВНЫЙ ЭКРАН ---
st.title("💧 Промышленный мониторинг утечек")

if st.session_state['data'] is not None:
    df = st.session_state['data']
    wn = st.session_state['network']
    
    # ЛОГИКА АНАЛИЗА (Важно для Экономики)
    df['Alert'] = df['Pressure (bar)'] < threshold
    is_leak = df['Alert'].any()
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Аналитика", "📋 Данные", "💰 Экономика", "🗺 Карта"])
    
    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Статус", "🚩 АВАРИЯ" if is_leak else "✅ НОРМА")
        c2.metric("Мин. давление", f"{df['Pressure (bar)'].min():.2f} Bar")
        c3.metric("Материал", selected_material)
        
        st.subheader("Показатели давления и расхода")
        st.line_chart(df[['Pressure (bar)', 'Flow Rate (L/s)']])

    with tab2:
        st.subheader("Сырые данные с датчиков")
        st.dataframe(df.style.highlight_between(left=0, right=threshold, subset=['Pressure (bar)'], color='red'))

    with tab3:
        st.subheader("Финансовый анализ потерь")
        # Считаем объем потерь: разница в потоке до и после аварии
        # Если давление ниже порога, считаем, что вода уходит впустую
        lost_vol = df[df['Alert'] == True]['Flow Rate (L/s)'].sum() * (3600 / sampling)
        total_cost = lost_vol * tariff
        
        col_a, col_b = st.columns(2)
        col_a.metric("Объем утечки", f"{lost_vol:,.1f} литров")
        col_b.metric("Финансовый ущерб", f"{total_cost:,.0f} ₸", delta_color="inverse")
        
        st.info(f"При текущем тарифе {tariff} тг/л система окупится за счет обнаружения подобных аварий.")

    with tab4:
        st.subheader("Визуализация участка сети")
        fig, ax = plt.subplots(figsize=(10, 6))
        pos = {n: wn.get_node(n).coordinates for n in wn.node_name_list}
        
        # Красим аварию на карте
        leak_node = st.session_state.get('leak_node')
        node_colors = []
        for n in wn.node_name_list:
            if n == 'Res': node_colors.append('blue')
            elif n == leak_node and is_leak: node_colors.append('red')
            else: node_colors.append('green')
            
        nx.draw_networkx_edges(wn.get_graph(), pos, ax=ax, edge_color='gray', width=2)
        nx.draw_networkx_nodes(wn.get_graph(), pos, ax=ax, node_color=node_colors, node_size=300)
        nx.draw_networkx_labels(wn.get_graph(), pos, ax=ax, font_size=8)
        st.pyplot(fig)
else:
    st.warning("Нажмите кнопку 'Запустить расчет' в левом меню для получения данных.")
