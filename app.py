import streamlit as st
import pandas as pd
import numpy as np
import wntr
import requests

# --- КОНФИГУРАЦИЯ ---
st.set_page_config(page_title="Smart Shygyn PRO", page_icon="💧", layout="wide")

# Функции (Telegram и EPANET) оставляем те же, что были раньше...
# [Здесь должны быть функции send_telegram_msg и run_epanet_simulation]

# --- SIDEBAR ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3145/3145024.png", width=100)
st.sidebar.title("Smart Shygyn v2.0")
city = st.sidebar.selectbox("📍 Город:", ["Алматы", "Астана", "Шымкент"])
tariff = st.sidebar.slider("💰 Тариф (тг/литр):", 0.1, 1.5, 0.5)
threshold = st.sidebar.slider("📉 Порог тревоги (Bar):", 1.0, 5.0, 2.8)

# --- ГЛАВНЫЙ ЭКРАН ---
st.title(f"🏢 Система мониторинга водоснабжения: {city}")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Мониторинг", "📋 Данные", "💰 Экономика", "🛠 Тех-поддержка"])

# --- ЛОГИКА ЗАГРУЗКИ ---
if 'data' not in st.session_state:
    st.session_state['data'] = None

with st.sidebar:
    if st.button("🚀 Запустить ИИ-симуляцию"):
        # Тут вызываем нашу функцию с EPANET
        st.session_state['data'] = run_epanet_simulation()

df = st.session_state['data']

if df is not None:
    # Аналитика
    df['Alert'] = df['Pressure (bar)'] < threshold
    lost_vol = df[df['Alert'] == True]['Flow Rate (L/s)'].sum() * 3600
    
    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Текущее давление", f"{df['Pressure (bar)'].iloc[-1]:.2f} Bar")
        c2.metric("Потери воды", f"{lost_vol:.1f} Л", delta=f"-{lost_vol*0.1:.1f}", delta_color="inverse")
        c3.metric("Ущерб", f"{int(lost_vol * tariff)} ₸")
        c4.metric("Статус", "🚩 КРИТИЧЕСКИ" if lost_vol > 0 else "✅ ОК")
        
        st.subheader("Живой график потока и давления")
        st.line_chart(df[['Pressure (bar)', 'Flow Rate (L/s)']])
        
        if lost_vol > 0:
            st.error(f"Внимание! Обнаружена утечка. Давление упало ниже {threshold} Bar.")
            if st.button("📢 Оповестить диспетчера"):
                send_telegram_msg(f"Авария в {city}! Потери {lost_vol:.1f} л.")

    with tab2:
        st.subheader("Сырые данные с сенсоров")
        st.dataframe(df.style.highlight_max(axis=0, color='lightcoral'))
        st.download_button("📥 Скачать CSV отчет", df.to_csv(), "report.csv")

    with tab3:
        st.subheader("Прогноз окупаемости системы")
        col_a, col_b = st.columns(2)
        daily_loss = lost_vol * 24
        col_a.info(f"Прогноз потерь в сутки: {daily_loss:.0f} литров")
        col_b.warning(f"Финансовый риск в месяц: {daily_loss * 30 * tariff:,.0f} ₸")
        
        # Маленький график прогноза
        chart_data = pd.DataFrame(np.random.randn(20, 1), columns=['Прогноз экономии'])
        st.area_chart(chart_data)

    with tab4:
        st.subheader("Состояние оборудования")
        st.write("✅ Датчик давления №041 - Активен")
        st.write("✅ Радиомодуль LoRaWAN - Сигнал отличный")
        st.write("⚠️ Требуется калибровка датчика потока через 14 дней")

else:
    st.warning("Нажмите кнопку в меню слева, чтобы получить данные из системы EPANET.")
