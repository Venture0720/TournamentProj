import streamlit as st
import pandas as pd
import requests

# Функция отправки уведомления
def send_telegram_msg(text):
    token = st.secrets["8374801663:AAHmqjjDbFs2F54FZqxXjYLpuRK1uTSlqp0"]
    chat_id = st.secrets["smartshygyn_bot"]
    url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text={text}"
    requests.get(url)

# Твой основной код дальше...
# В месте, где выявляется утечка, добавь:
if total_leaks > 0:
    st.error("Обнаружена утечка! Отправляем уведомление...")
    send_telegram_msg(f"⚠️ ТРЕВОГА! В секторе найдена утечка. Потери: {lost_litres} литров.")
import streamlit as st
import pandas as pd

# Заголовок приложения
st.title("💧 Smart Shygyn: ИИ-мониторинг воды")
st.markdown("Система раннего обнаружения утечек для ЖК и предприятий Казахстана")

# 1. Загрузка данных (пользователь может сам загрузить свой CSV)
uploaded_file = st.file_uploader("Загрузите данные со счетчиков (.csv)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Считаем показатели
    total_leaks = df['Leak Status'].sum()
    lost_litres = df[df['Leak Status'] == 1]['Flow Rate (L/s)'].sum()
    money_lost = lost_litres * 0.5 # Тариф 0.5 тенге за литр
    
    # 2. Главные карточки (Индикаторы)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status = "🔴 АВАРИЯ" if total_leaks > 0 else "🟢 НОРМА"
        st.metric("Статус системы", status)
        
    with col2:
        st.metric("Потеряно воды", f"{lost_litres:.1f} л")
        
    with col3:
        st.metric("Убытки (тенге)", f"{int(money_lost)} ₸", delta=f"-{int(money_lost)}", delta_color="inverse")

    # 3. Визуализация для профи (график)
    st.subheader("Анализ потока и давления")
    st.line_chart(df[['Flow Rate (L/s)', 'Pressure (bar)']].head(500))

    # 4. Рекомендация для обычного человека
    st.info("💡 **Рекомендация ИИ:** Обнаружена аномалия в Секторе 4. Вероятная причина: износ прокладки. Рекомендуется осмотр в течение 24 часов.")
else:
    st.warning("Пожалуйста, загрузите файл с данными для начала анализа.")
