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
    wn = wntr.network.WaterNetworkModel()
    start_p = random.uniform(28, 42)
    leak_hr = random.randint(10, 16)
    
    # 1. Добавляем узлы
    res = wn.add_reservoir('res', base_head=start_p)
    n1 = wn.add_junction('node1', base_demand=0.005, elevation=10)
    n2 = wn.add_junction('node2', base_demand=0.005, elevation=10)
    
    # 2. Устанавливаем координаты ЧЕРЕЗ АТРИБУТЫ (самый надежный способ)
    # Это исправит AttributeError
    wn.get_node('res').coordinates = (0, 5)
    wn.get_node('node1').coordinates = (5, 5)
    wn.get_node('node2').coordinates = (10, 5)
    
    # 3. Добавляем трубы
    wn.add_pipe('p1', 'res', 'node1', length=100, diameter=0.2, roughness=100)
    wn.add_pipe('p2', 'node1', 'node2', length=100, diameter=0.2, roughness=100)
    
    wn.options.time.duration = 24 * 3600
    wn.options.time.report_timestep = 3600
    
    # Моделируем утечку
    node2 = wn.get_node('node2')
    node2.add_leak(wn, area=0.05, start_time=leak_hr * 3600)
    
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()
    
    p = results.node['pressure']['node2'] * 0.1
    f = results.link['flowrate']['p2'] * 1000
    noise = np.random.normal(0, 0.015, len(p))
    
    df_res = pd.DataFrame({
        'Pressure (bar)': p.values + noise,
        'Flow Rate (L/s)': np.abs(f.values) + (noise * 0.1),
        'Leak Status': [0 if t < leak_hr*3600 else 1 for t in p.index]
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
        st.subheader("🗺 Проекция цифрового двойника сети")
        if wn:
            import networkx as nx # WNTR строит графы на базе networkx
            
            fig_map, ax = plt.subplots(figsize=(10, 5))
            
            # Получаем граф и координаты
            graph = wn.get_graph()
            pos = {node: wn.get_node(node).coordinates for node in wn.node_name_list}
            
            # Определяем цвета узлов вручную
            colors = []
            for node in wn.node_name_list:
                if node == 'res':
                    colors.append('blue')
                elif node == 'node2' and is_leak:
                    colors.append('red')
                else:
                    colors.append('green')
            
            # Рисуем трубы (ребра)
            nx.draw_networkx_edges(graph, pos, ax=ax, width=3, edge_color='gray')
            
            # Рисуем узлы
            nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=colors, node_size=300)
            
            # Добавляем подписи
            nx.draw_networkx_labels(graph, pos, ax=ax, font_size=10, font_weight='bold', verticalalignment='bottom')
            
            ax.set_title("Схема мониторинга: Резервуар (Синий) -> Магистраль -> Узел утечки (Красный)")
            ax.axis('off') # Убираем лишние оси координат
            st.pyplot(fig_map)
            
            if is_leak:
                st.warning("📍 Локализация: Авария подтверждена гидравлической моделью в узле Node 2")
        else:
            st.info("Проекция доступна только после запуска EPANET симуляции.")
