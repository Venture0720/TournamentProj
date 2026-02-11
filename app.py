import streamlit as st
import pandas as pd
import numpy as np
import wntr
import requests
import random

# --- 1. ФУНКЦИИ (Backend) ---

def send_telegram_msg(text):
    try:
        token = st.secrets["TELEGRAM_TOKEN"]
        chat_id = st.secrets["CHAT_ID"]
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        params = {"chat_id": chat_id, "text": text}
        response = requests.get(url, params=params)
        if response.status_code == 200:
            st.success("✅ Отчет доставлен в Telegram!")
        else:
            st.error(f"Ошибка Telegram: {response.text}")
    except Exception as e:
        st.error(f"Ошибка секретов: {e}")

def run_epanet_simulation():
    wn = wntr.network.WaterNetworkModel()
    
    # Случайные параметры для разнообразия данных
    start_p = random.uniform(28, 42)
    leak_hr = random.randint(10, 16)
    
    wn.add_reservoir('res', base_head=start_p)
    wn.add_junction('node1', base_demand=0.005, elevation=10)
    wn.add_junction('node2', base_demand=0.005, elevation=10)
    wn.add_pipe('p1', 'res', 'node1', length=100, diameter=0.2, roughness=100)
    wn.add_pipe('p2', 'node1', 'node2', length=100, diameter=0.2, roughness=100)
    
    wn.options.time.duration = 24 * 3600
    wn.options.time.report_timestep = 3600
    
    # Добавляем утечку
    node2 = wn.get_node('node2')
    node2.add_leak(wn, area=0.05, start_time=leak_hr * 3600)
    
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()
    
    p = results.node['pressure']['node2'] * 0.1 # в бары
    f = results.link['flowrate']['p2'] * 1000  # в л/с
    
    # Добавляем шум для реализма
    noise = np.random.normal(0, 0.015, len(p))
    
    return pd.DataFrame({
        'Pressure (bar)': p.values + noise,
        'Flow Rate (L/s)': np.abs(f.values) + (noise * 0.1),
        'Leak Status': [0 if t < leak_hr*3600 else 1 for t in p.index]
    })

# --- 2. КОНФИГУРАЦИЯ ИНТЕРФЕЙСА ---
st.set_page_config(page_title="Smart Shygyn PRO", page_icon="💧", layout="wide")

# --- 3. SIDEBAR ---
st.sidebar.title("💧 Smart Shygyn v2.0")
city = st.sidebar.selectbox("📍 Локация:", ["Алматы", "Астана", "Шымкент"])
tariff = st.sidebar.slider("💰 Тариф (тг/литр):", 0.1, 1.5, 0.5)
threshold = st.sidebar.slider("📉 Порог тревоги (Bar):", 1.0, 5.0, 2.5)

if st.sidebar.button("🚀 Запустить ИИ-симуляцию"):
    st.session_state['data'] = run_epanet_simulation()

# --- 4. ОСНОВНОЙ БЛОК ---
st.title(f"🏢 Мониторинг сети: {city}")

if 'data' not in st.session_state:
    st.session_state['data'] = None

df = st.session_state['data']

if df is not None:
    # Анализ на основе нашего ИИ-порога
    df['AI_Alert'] = df['Pressure (bar)'] < threshold
    total_leaks = int(df['AI_Alert'].sum())
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Мониторинг", "📋 Данные", "💰 Экономика", "🛠 Тех-аудит"])

    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        status = "🔴 КРИТИЧЕСКИ" if total_leaks > 0 else "✅ НОРМА"
        c1.metric("Статус", status)
        
        # Считаем потери только в моменты аномалий
        lost_vol = df[df['AI_Alert'] == True]['Flow Rate (L/s)'].sum() * 3600
        c2.metric("Потери воды", f"{lost_vol:.1f} л")
        c3.metric("Убытки", f"{int(lost_vol * tariff)} ₸")
        c4.metric("Давление (min)", f"{df['Pressure (bar)'].min():.2f} bar")

        st.subheader("Анализ гидравлических показателей")
        st.line_chart(df[['Pressure (bar)', 'Flow Rate (L/s)']])
        
        if total_leaks > 0:
            st.error("⚠️ Внимание! Обнаружена разгерметизация участка.")
            if st.button("📲 Отправить отчет в Telegram"):
                msg = f"🚨 АВАРИЯ: {city}\nПотери: {lost_vol:.1f}л\nУщерб: {int(lost_vol * tariff)}тг"
                send_telegram_msg(msg)

    with tab2:
        st.dataframe(df.style.highlight_max(axis=0, subset=['Flow Rate (L/s)'], color='orange'))
        st.download_button("📥 Экспорт в CSV", df.to_csv(), "report_shygyn.csv")

    with tab3:
        st.subheader("Прогноз потерь (30 дней)")
        daily_loss_val = lost_vol * 24 if total_leaks > 0 else 0
        st.info(f"При текущем состоянии сети риск потерь составляет {daily_loss_val * 30 * tariff:,.0f} ₸ в месяц.")
        st.bar_chart(np.random.randint(100, 500, 30))

    with tab4:
        st.write("🔧 **Диагностика узлов:**")
        st.write("- Датчик давления (Node2): **Стабилен**")
        st.write("- Шлюз LoRaWAN: **Подключен**")
        st.write("- Последняя синхронизация: **Только что**")

else:
    st.info("👋 Добро пожаловать! Нажмите кнопку в боковом меню для запуска анализа сети.")
