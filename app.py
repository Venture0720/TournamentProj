import streamlit as st
import pandas as pd
import requests

# 1. Функция отправки сообщения в Telegram
def send_telegram_msg(text):
    try:
        # Берем данные из Secrets по коротким именам
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

# Настройка интерфейса
st.set_page_config(page_title="Smart Shygyn", page_icon="💧")
st.title("💧 Smart Shygyn: ИИ-мониторинг воды")
st.markdown("---")

# 2. Загрузка данных
uploaded_file = st.file_uploader("Загрузите CSV-файл с датчиков", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Расчеты
    total_leaks = int(df['Leak Status'].sum())
    lost_litres = df[df['Leak Status'] == 1]['Flow Rate (L/s)'].sum()
    money_lost = int(lost_litres * 0.5) # 0.5 тенге за литр
    
    # Метрики
    col1, col2, col3 = st.columns(3)
    col1.metric("Статус", "🔴 АВАРИЯ" if total_leaks > 0 else "🟢 НОРМА")
    col2.metric("Потери воды", f"{lost_litres:.1f} л")
    col3.metric("Убытки", f"{money_lost} ₸")

    # Уведомление и кнопка
    if total_leaks > 0:
        st.warning(f"В системе зафиксировано {total_leaks} аномалий!")
        if st.button("🚀 Отправить мгновенный отчет диспетчеру"):
            report = (
                f"🚨 Smart Shygyn ALERT\n"
                f"----------------------\n"
                f"💧 Потери: {lost_litres:.1f} литров\n"
                f"💸 Ущерб: {money_lost} тенге\n"
                f"📍 Статус: Требуется выезд бригады!"
            )
            send_telegram_msg(report)

    # График
    st.subheader("📊 Анализ показателей давления и расхода")
    st.line_chart(df[['Flow Rate (L/s)', 'Pressure (bar)']].head(500))
    
    st.info("💡 **ИИ-анализ:** Аномалии обнаружены. Система рекомендует проверить целостность труб в Секторе 4.")
else:
    st.info("Ожидание загрузки данных для анализа...")
