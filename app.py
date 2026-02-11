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
    import random # Добавь это в импорты в самом верху
    wn = wntr.network.WaterNetworkModel()
    
    # Рандомизируем параметры сети
    start_pressure = random.uniform(25, 45) # Случайное давление от 2.5 до 4.5 бар
    leak_start_hour = random.randint(8, 18) # Утечка начнется в случайный час
    leak_size = random.uniform(0.03, 0.08)  # Случайный размер дырки
    
    wn.add_reservoir('res', base_head=start_pressure)
    wn.add_junction('node1', base_demand=0.005, elevation=10)
    wn.add_junction('node2', base_demand=0.005, elevation=10)
    wn.add_pipe('p1', 'res', 'node1', length=100, diameter=0.2, roughness=100)
    wn.add_pipe('p2', 'node1', 'node2', length=100, diameter=0.2, roughness=100)
    
    wn.options.time.duration = 24 * 3600
    wn.options.time.report_timestep = 3600
    
    node2 = wn.get_node('node2')
    # Утечка теперь начинается в случайное время и имеет разный масштаб
    node2.add_leak(wn, area=leak_size, start_time=leak_start_hour * 3600)
    
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()
    
    p = results.node['pressure']['node2'] * 0.1
    f = results.link['flowrate']['p2'] * 1000
    
    # Добавляем небольшой сенсорный шум (jitter), чтобы графики не были идеально гладкими
    noise = np.random.normal(0, 0.02, len(p)) 
    
    data = pd.DataFrame({
        'Pressure (bar)': p.values + noise,
        'Flow Rate (L/s)': np.abs(f.values) + (noise * 0.1),
        'Leak Status': [0 if t < leak_start_hour*3600 else 1 for t in p.index]
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
# --- УМНАЯ АНАЛИТИКА (ВСТАВИТЬ ВМЕСТО СТАРЫХ РАСЧЕТОВ) ---

if df is not None:
    # 1. Очистка и сглаживание (убираем шум датчиков)
    # Считаем скользящее среднее по 3 точкам
    df['Smooth_P'] = df['Pressure (bar)'].rolling(window=3).mean()
    df['Smooth_F'] = df['Flow Rate (L/s)'].rolling(window=3).mean()
    
    # 2. ЛОГИКА ДЕТЕКЦИИ (Собственный ИИ-алгоритм)
    # Условие: Давление ниже 2.5 бар И поток выше среднего на 20%
    mean_flow = df['Smooth_F'].mean()
    
    # Создаем маску аномалий (программа сама решает, где авария)
    df['AI_Leak_Detected'] = (df['Smooth_P'] < 2.5) & (df['Smooth_F'] > mean_flow * 1.2)
    
    # Считаем итоги по нашему AI, а не по меткам в файле
    total_leaks = int(df['AI_Leak_Detected'].sum())
    
    # Считаем объем потерь только там, где наш AI увидел аварию
    lost_litres = df[df['AI_Leak_Detected'] == True]['Flow Rate (L/s)'].sum() * 3600 # в час
    money_lost = int(lost_litres * 0.5)

    # --- ВИЗУАЛИЗАЦИЯ МЕТРИК ---
    col1, col2, col3 = st.columns(3)
    
    # Статус теперь зависит от нашего анализа
    if total_leaks > 0:
        col1.error("🔴 АВАРИЯ НАЙДЕНА")
        st.sidebar.warning(f"AI обнаружил {total_leaks} аномальных сегментов")
    else:
        col1.success("🟢 СИСТЕМА В НОРМЕ")

    col2.metric("Потери воды (AI)", f"{lost_litres:.1f} л")
    col3.metric("Убытки", f"{money_lost} ₸")

    # График с зонами аномалий
    st.subheader("📊 Анализ гидравлических отклонений")
    # Показываем точки, где AI зафиксировал проблему
    st.line_chart(df[['Smooth_P', 'Smooth_F']])
    
    if total_leaks > 0:
        st.info("🤖 **Анализ ИИ:** Зафиксировано резкое падение давления при аномальном росте расхода. Это не похоже на обычное потребление. Рекомендуется проверка задвижек.")
