import streamlit as st
import pandas as pd
import numpy as np
import wntr
import requests
import random
import plotly.express as px
import matplotlib.pyplot as plt

# --- 1. ФУНКЦИИ (Backend с реальной физикой) ---

def run_epanet_simulation(material_c, degradation, sampling_rate):
    wn = wntr.network.WaterNetworkModel()
    dist = 100
    
    # Реальный внутренний диаметр с учетом износа (минус % от номинала)
    base_diameter = 0.2
    actual_diameter = base_diameter * (1 - degradation / 100)
    
    # Создаем сетку
    for i in range(4):
        for j in range(4):
            name = f"N_{i}_{j}"
            wn.add_junction(name, base_demand=0.001, elevation=10)
            wn.get_node(name).coordinates = (i * dist, j * dist)
            if i > 0:
                # Вставляем выбранную шероховатость (material_c)
                wn.add_pipe(f"PH_{i}_{j}", f"N_{i-1}_{j}", name, 
                            length=dist, diameter=actual_diameter, roughness=material_c)
            if j > 0:
                wn.add_pipe(f"PV_{i}_{j}", f"N_{i}_{j-1}", name, 
                            length=dist, diameter=actual_diameter, roughness=material_c)

    wn.add_reservoir('Res', base_head=40)
    wn.get_node('Res').coordinates = (-dist, -dist)
    wn.add_pipe('P_Main', 'Res', 'N_0_0', length=dist, diameter=0.4, roughness=material_c)

    leak_node = "N_2_2" # Для стабильности теста
    st.session_state['leak_node'] = leak_node
    
    # Настройка времени симуляции
    wn.options.time.duration = 24 * 3600
    wn.options.time.report_timestep = 3600 // sampling_rate # Частота данных
    
    node = wn.get_node(leak_node)
    node.add_leak(wn, area=0.08, start_time=12 * 3600)
    
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()
    
    p = results.node['pressure'][leak_node] * 0.1
    f = results.link['flowrate']['P_Main'] * 1000
    
    return pd.DataFrame({
        'Pressure (bar)': p.values,
        'Flow Rate (L/s)': np.abs(f.values)
    }), wn

# --- 2. SIDEBAR (Функциональные настройки) ---
st.sidebar.title("🛠 Инженерные параметры")

# Выбор материала напрямую влияет на формулу трения
materials = {"Пластик (ПНД)": 150, "Новая сталь": 140, "Чугун (старый)": 100, "Бетон": 110}
selected_material = st.sidebar.selectbox("Материал труб (Коэф. шероховатости):", list(materials.keys()))
c_value = materials[selected_material]

# Износ влияет на диаметр труб в модели
degradation = st.sidebar.slider("Степень износа сети (% зарастания):", 0, 50, 10)

# Sampling влияет на плотность точек на графике
sampling = st.sidebar.select_slider("Частота опроса датчиков (раз в час):", options=[1, 2, 4, 6])

tariff = st.sidebar.number_input("Тариф (тг/литр):", value=0.45)
threshold = st.sidebar.slider("Порог детекции утечки (Bar):", 1.0, 5.0, 2.8)

if st.sidebar.button("🚀 Пересчитать Цифровой Двойник"):
    # Теперь функция принимает РЕАЛЬНЫЕ параметры
    data, network = run_epanet_simulation(c_value, degradation, sampling)
    st.session_state['data'] = data
    st.session_state['network'] = network

# --- 3. ИНТЕРФЕЙС ---
st.title("💧 Smart Shygyn: Промышленный мониторинг")

if st.session_state.get('data') is not None:
    df = st.session_state['data']
    wn = st.session_state['network']
    
    # Логика детекции
    df['Alert'] = df['Pressure (bar)'] < threshold
    is_leak = df['Alert'].any()
    
    t1, t2, t3, t4 = st.tabs(["📊 Аналитика", "📋 Данные", "💰 Экономика", "🗺 Карта сети"])
    
    with t1:
        st.subheader(f"Гидравлический режим: {selected_material}")
        st.line_chart(df[['Pressure (bar)', 'Flow Rate (L/s)']])
        if is_leak:
            st.error(f"⚠️ ВНИМАНИЕ: Давление упало ниже {threshold} bar. Зафиксирована утечка.")

    with t4:
        st.subheader("🗺 Состояние узлов городского квартала")
        import networkx as nx
        fig, ax = plt.subplots(figsize=(10, 6))
        pos = {n: wn.get_node(n).coordinates for n in wn.node_name_list}
        
        # Визуально показываем влияние износа на толщину линий
        edge_width = 1 + (1 - degradation/100) * 3
        
        nx.draw_networkx_edges(wn.get_graph(), pos, ax=ax, width=edge_width, edge_color='gray')
        
        # Красим аварию
        leak_n = st.session_state['leak_node']
        node_colors = ['red' if (n == leak_n and is_leak) else 'blue' if n == 'Res' else 'green' for n in wn.node_name_list]
        
        nx.draw_networkx_nodes(wn.get_graph(), pos, ax=ax, node_color=node_colors, node_size=300)
        st.pyplot(fig)
        st.info(f"Расчет выполнен для труб с эквивалентной шероховатостью C={c_value}")
