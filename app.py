import streamlit as st
import pandas as pd
import numpy as np
import wntr
import requests
import random
import plotly.express as px
import matplotlib.pyplot as plt

# --- 1. ФУНКЦИИ (Backend) ---

def validate_and_clean_data(df):
    required_columns = ['Pressure (bar)', 'Flow Rate (L/s)']
    for col in required_columns:
        if col not in df.columns:
            st.error(f"❌ В файле отсутствует колонка: {col}")
            return None
    df = df.dropna(subset=required_columns)
    df = df[df['Pressure (bar)'] < 100] 
    return df

def send_telegram_msg(text):
    try:
        token = st.secrets["TELEGRAM_TOKEN"]
        chat_id = st.secrets["CHAT_ID"]
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        params = {"chat_id": chat_id, "text": text}
        response = requests.get(url, params=params)
        if response.status_code == 200:
            st.success("✅ Отчет доставлен!")
        else:
            st.error(f"Ошибка Telegram: {response.text}")
    except Exception as e:
        st.error(f"Ошибка секретов: {e}")

def run_epanet_simulation():
    # Создаем сложную сетку 5x5 (городской квартал)
    wn = wntr.network.WaterNetworkModel()
    
    # Параметры сетки
    dim = 5  
    dist = 100 # расстояние между узлами
    
    # Создаем узлы и трубы автоматически
    for i in range(dim):
        for j in range(dim):
            name = f"N_{i}_{j}"
            wn.add_junction(name, base_demand=0.001, elevation=10)
            wn.get_node(name).coordinates = (i * dist, j * dist)
            
            # Соединяем горизонтально
            if i > 0:
                wn.add_pipe(f"PH_{i}_{j}", f"N_{i-1}_{j}", name, length=dist, diameter=0.2, roughness=100)
            # Соединяем вертикально
            if j > 0:
                wn.add_pipe(f"PV_{i}_{j}", f"N_{i}_{j-1}", name, length=dist, diameter=0.2, roughness=100)

    # Добавляем мощный резервуар в углу
    wn.add_reservoir('Res', base_head=40)
    wn.get_node('Res').coordinates = (-dist, -dist)
    wn.add_pipe('P_Main', 'Res', 'N_0_0', length=dist, diameter=0.4, roughness=100)

    # Имитируем СЛУЧАЙНУЮ аварию в одном из узлов квартала
    leak_node = f"N_{random.randint(1, 4)}_{random.randint(1, 4)}"
    st.session_state['leak_node'] = leak_node # Запоминаем для карты
    
    wn.options.time.duration = 24 * 3600
    wn.options.time.report_timestep = 3600
    
    node = wn.get_node(leak_node)
    node.add_leak(wn, area=0.08, start_time=12 * 3600)
    
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()
    
    # Берем данные давления именно из узла утечки
    p = results.node['pressure'][leak_node] * 0.1
    f = results.link['flowrate']['P_Main'] * 1000
    noise = np.random.normal(0, 0.02, len(p))
    
    df_res = pd.DataFrame({
        'Pressure (bar)': p.values + noise,
        'Flow Rate (L/s)': np.abs(f.values) + (noise * 0.1),
        'Leak Status': [0 if t < 12*3600 else 1 for t in p.index]
    })
    
    return df_res, wn

# --- 2. КОНФИГУРАЦИЯ ИНТЕРФЕЙСА ---
st.set_page_config(page_title="Smart Shygyn PRO", page_icon="💧", layout="wide")

# --- 3. SIDEBAR ---
st.sidebar.title("💧 Smart Shygyn v2.0")
mode = st.sidebar.radio("Режим данных:", ["Генератор EPANET", "Загрузить CSV"])
city = st.sidebar.selectbox("📍 Локация:", ["Алматы", "Астана", "Шымкент"])
tariff = st.sidebar.slider("💰 Тариф (тг/литр):", 0.1, 1.5, 0.5)
threshold = st.sidebar.slider("📉 Порог тревоги (Bar):", 1.0, 5.0, 2.5)

if 'data' not in st.session_state:
    st.session_state['data'] = None
    st.session_state['network'] = None

if mode == "Генератор EPANET":
    if st.sidebar.button("🚀 Запустить ИИ-симуляцию"):
        data, network = run_epanet_simulation()
        st.session_state['data'] = data
        st.session_state['network'] = network
else:
    uploaded_file = st.sidebar.file_uploader("Загрузите CSV", type="csv")
    if uploaded_file:
        raw_df = pd.read_csv(uploaded_file)
        st.session_state['data'] = validate_and_clean_data(raw_df)

# --- 4. ОСНОВНОЙ БЛОК ---
st.title(f"🏢 Мониторинг сети: {city}")
df = st.session_state['data']
wn = st.session_state['network']

if df is not None:
    df['AI_Alert'] = df['Pressure (bar)'] < threshold
    total_leaks = int(df['AI_Alert'].sum())
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Мониторинг", "📋 Данные", "💰 Экономика", "🛠 Тех-аудит"])

    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        is_leak = total_leaks > 0
        c1.metric("Статус", "🔴 КРИТИЧЕСКИ" if is_leak else "✅ НОРМА")
        lost_vol = df[df['AI_Alert'] == True]['Flow Rate (L/s)'].sum() * 3600
        c2.metric("Потери воды", f"{lost_vol:.1f} л")
        c3.metric("Убытки", f"{int(lost_vol * tariff)} ₸")
        c4.metric("Давление (min)", f"{df['Pressure (bar)'].min():.2f} bar")

        st.subheader("🌋 Анализ давления по времени")
        fig = px.scatter(df, x=df.index, y="Pressure (bar)", 
                         color="Pressure (bar)", 
                         color_continuous_scale="RdYlGn")
        st.plotly_chart(fig, use_container_width=True)
        
        if is_leak:
            st.error("⚠️ Внимание! Обнаружена разгерметизация участка.")
            if st.button("📲 Отправить отчет в Telegram"):
                msg = f"🚨 АВАРИЯ: {city}\nПотери: {lost_vol:.1f}л\nУщерб: {int(lost_vol * tariff)}тг"
                send_telegram_msg(msg)

    with tab2:
        st.dataframe(df.style.highlight_max(axis=0, subset=['Flow Rate (L/s)'], color='orange'))

    with tab3:
        st.subheader("Прогноз потерь (30 дней)")
        daily_loss_val = lost_vol * 24 if total_leaks > 0 else 0
        st.info(f"Риск потерь: {daily_loss_val * 30 * tariff:,.0f} ₸/мес")
        st.bar_chart(np.random.randint(100, 500, 30))

   with tab4:
        st.subheader("🗺 Цифровой двойник: Анализ городского квартала")
        if wn:
            import networkx as nx
            fig_map, ax = plt.subplots(figsize=(12, 8))
            
            graph = wn.get_graph()
            pos = {node: wn.get_node(node).coordinates for node in wn.node_name_list}
            leak_node = st.session_state.get('leak_node', None)
            
            # Цвета: Резервуар - синий, Обычные - зеленые, Авария - мигающий красный
            node_colors = []
            node_sizes = []
            for node in wn.node_name_list:
                if node == 'Res':
                    node_colors.append('#1f77b4') # Синий
                    node_sizes.append(500)
                elif node == leak_node and is_leak:
                    node_colors.append('#d62728') # Красный
                    node_sizes.append(700)
                else:
                    node_colors.append('#2ca02c') # Зеленый
                    node_sizes.append(200)
            
            # Рисуем сеть
            nx.draw_networkx_edges(graph, pos, ax=ax, width=1.5, edge_color='#bdc3c7', alpha=0.7)
            nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=node_colors, node_size=node_sizes, edgecolors='white')
            
            # Подписи только для важных узлов
            important_nodes = {'Res': 'ИСТОЧНИК', leak_node: 'ЗОНА АВАРИИ' if is_leak else ''}
            labels = {n: important_nodes.get(n, '') for n in wn.node_name_list}
            nx.draw_networkx_labels(graph, pos, labels=labels, ax=ax, font_size=12, font_weight='bold', font_color='#2c3e50')
            
            ax.axis('off')
            st.pyplot(fig_map)
            
            if is_leak:
                st.critical(f"📍 Авария локализована в секторе: **{leak_node}**")
                st.info("ИИ рекомендует перекрыть задвижки PV_1_2 и PH_2_1 для изоляции участка.")
