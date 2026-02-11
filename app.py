import streamlit as st
import pandas as pd
import requests
import wntr  # Библиотека для работы с EPANET
import numpy as np

# 1. Функция отправки сообщения в Telegram
def send_telegram_msg(text):
    try:
        token = st.secrets["TELEGRAM_TOKEN"]
        chat_id = st.secrets["CHAT_ID"]
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        params = {"chat_id": chat_id, "text": text}
        response = requests.get(url, params=params)
        if response.status_code == 200:
            st.success("✅ Отчет успешно доставлен в Telegram!")
        else:
            st.error(f"Ошибка Telegram: {response.text}")
    except Exception as e:
        st.error(f"Ошибка доступа к секретам: {e}")

# 2. Функция генерации данных через EPANET (WNTR)
def run_epanet_simulation():
    wn = wntr.network.WaterNetworkModel()
    # Создаем базовую сеть
    wn.add_reservoir('res', base_head=30)
    wn.add_junction('node1', base_demand=0.005, elevation=10)
    wn.add_junction('node2', base_demand=0.005, elevation=10)
    wn.add_pipe('p1', 'res', 'node1', length=100, diameter=0.2, roughness=100)
    wn.add_pipe('p2', 'node1', 'node2', length=100, diameter=0.2, roughness=100)
    
    # Настройка времени (24 часа)
    wn.options.time.duration = 24 * 3600
    wn.options.time.report_timestep = 3600 # шаг 1 час
    
    # Моделируем утечку на node2 в середине дня
    node2 = wn.get_node('node2')
    # Добавляем эмиттер (утечку) с 12-го часа
    node2.add_leak(wn, area=0.05, start_time=12 * 3600)
    
    # Симуляция
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()
    
    # Превращаем в таблицу
    p = results.node['pressure']['node2'] * 0.1 # бар
    f = results.link['flowrate']['p2'] * 1000  # л/с
    
    data = pd.DataFrame({
        'Pressure (bar)': p.values,
        'Flow Rate (L/s)': np.abs(f.values),
        'Leak Status': [0 if t < 12*3600 else 1 for t in p.index]
    })
    return data

# --- ИНТЕРФЕЙС STREAMLIT ---
st.set_page_config(page_title="Smart Shygyn", page_icon="💧")
st.title("💧 Smart Shygyn: Цифровой двойник сети")
st.markdown("Система на базе гидравлического движка **EPANET**")
st.markdown("---")

# Выбор режима данных
mode = st.radio("Выберите источник данных:", ["Загрузить свой CSV", "Сгенерировать через EPANET (Live Simulation)"])

df = None

if mode == "Загрузить свой CSV":
    uploaded_file = st.file_uploader("Загрузите CSV-файл с датчиков", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
else:
    if st.button("🚀 Запустить симуляцию гидравлики"):
        with st.spinner('Движок EPANET рассчитывает давление...'):
            df = run_epanet_simulation()
            st.session_state['data'] = df

if 'data' in st.session_state and mode == "Сгенерировать через EPANET (Live Simulation)":
    df = st.session_state['data']

# Обработка данных, если они есть
if df is not None:
    # Расчеты
    total_leaks = int(df['Leak Status'].sum())
    lost_litres = df[df['Leak Status'] == 1]['Flow Rate (L/s)'].sum() * 3600 # за час
    money_lost = int(lost_litres * 0.5)
    
    # Метрики
    col1, col2, col3 = st.columns(3)
    col1.metric("Статус", "🔴 АВАРИЯ" if total_leaks > 0 else "🟢 НОРМА")
    col2.metric("Потери за период", f"{lost_litres:.1f} л")
    col3.metric("Убытки (прогноз)", f"{money_lost} ₸")

    # Уведомление
    if total_leaks > 0:
        st.warning(f"⚠️ Обнаружено отклонение давления в узле симуляции!")
        if st.button("📲 Отправить данные в Telegram"):
            report = (f"🚨 Smart Shygyn (EPANET Model)\n"
                      f"Утечка зафиксирована на 12-м часе.\n"
                      f"Потери: {lost_litres:.1f} л.\n"
                      f"Давление упало до: {df['Pressure (bar)'].min():.2f} bar")
            send_telegram_msg(report)

    # Графики
    st.subheader("📊 Аналитика Digital Twin")
    st.line_chart(df[['Flow Rate (L/s)', 'Pressure (bar)']])
    
    st.info("ℹ️ Данные рассчитаны на основе модели водопроводной сети с использованием библиотеки WNTR (EPANET engine).")
