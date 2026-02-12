import streamlit as st
import pandas as pd
import numpy as np
import wntr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Smart Shygyn PRO", 
    layout="wide", 
    page_icon="💧",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: 700;
    }
    
    /* Make headers stand out */
    h1 {
        color: #1f77b4;
        text-align: center;
        padding: 20px 0;
    }
    
    h3 {
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 10px;
        margin-top: 20px;
    }
    
    /* Improve dataframe appearance */
    .dataframe {
        font-size: 12px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- BACKEND FUNCTIONS ---

def run_epanet_simulation(material_c, degradation, sampling_rate):
    """Run water network simulation with specified parameters"""
    wn = wntr.network.WaterNetworkModel()
    dist = 100
    actual_diameter = 0.2 * (1 - degradation / 100)
    
    # Create grid network
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

    # Add reservoir
    wn.add_reservoir('Res', base_head=40)
    wn.get_node('Res').coordinates = (-dist, -dist)
    wn.add_pipe('P_Main', 'Res', 'N_0_0', length=dist, diameter=0.4, roughness=material_c)

    # Add leak
    leak_node = "N_2_2"
    st.session_state['leak_node'] = leak_node
    
    wn.options.time.duration = 24 * 3600
    wn.options.time.report_timestep = 3600 // sampling_rate
    
    node = wn.get_node(leak_node)
    node.add_leak(wn, area=0.08, start_time=12 * 3600)
    
    # Run simulation
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()
    
    # Extract results
    p = results.node['pressure'][leak_node] * 0.1 
    f = results.link['flowrate']['P_Main'] * 1000 
    
    # Add realistic noise
    noise_p = np.random.normal(0, 0.04, len(p))
    noise_f = np.random.normal(0, 0.08, len(f))
    
    df_res = pd.DataFrame({
        'Hour': np.arange(len(p)) / sampling_rate,
        'Pressure (bar)': p.values + noise_p,
        'Flow Rate (L/s)': np.abs(f.values) + noise_f
    })
    
    return df_res, wn

def create_advanced_plot(df, threshold):
    """Create professional dual-axis plot with Plotly"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Давление в системе', 'Расход воды'),
        vertical_spacing=0.12,
        row_heights=[0.5, 0.5]
    )
    
    # Pressure plot
    fig.add_trace(
        go.Scatter(
            x=df['Hour'],
            y=df['Pressure (bar)'],
            name='Давление',
            line=dict(color='#3498db', width=3),
            fill='tonexty',
            fillcolor='rgba(52, 152, 219, 0.1)',
            hovertemplate='<b>Час:</b> %{x:.1f}<br><b>Давление:</b> %{y:.2f} bar<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Threshold line
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text="⚠️ Порог детекции",
        annotation_position="right",
        row=1, col=1
    )
    
    # Flow rate plot
    fig.add_trace(
        go.Scatter(
            x=df['Hour'],
            y=df['Flow Rate (L/s)'],
            name='Расход',
            line=dict(color='#e67e22', width=3),
            fill='tonexty',
            fillcolor='rgba(230, 126, 34, 0.1)',
            hovertemplate='<b>Час:</b> %{x:.1f}<br><b>Расход:</b> %{y:.2f} L/s<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_xaxes(title_text="Время (часы)", row=2, col=1, gridcolor='lightgray')
    fig.update_xaxes(title_text="Время (часы)", row=1, col=1, gridcolor='lightgray')
    fig.update_yaxes(title_text="Давление (bar)", row=1, col=1, gridcolor='lightgray')
    fig.update_yaxes(title_text="Расход (L/s)", row=2, col=1, gridcolor='lightgray')
    
    fig.update_layout(
        height=700,
        showlegend=True,
        hovermode='x unified',
        plot_bgcolor='white',
        font=dict(size=12),
        margin=dict(l=60, r=60, t=80, b=60)
    )
    
    return fig

# --- INITIALIZE SESSION STATE ---
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'network' not in st.session_state:
    st.session_state['network'] = None
if 'log' not in st.session_state:
    st.session_state['log'] = []

# --- SIDEBAR ---
st.sidebar.title("🧪 Панель управления")

with st.sidebar.expander("⚙️ Параметры сети", expanded=True):
    m_types = {
        "Пластик (ПНД)": 150, 
        "Сталь": 140, 
        "Чугун": 100
    }
    material = st.selectbox("Материал труб", list(m_types.keys()))
    iznos = st.slider("Износ системы (%)", 0, 60, 15, help="Процент деградации трубопровода")
    freq = st.select_slider("Частота опроса датчиков", options=[1, 2, 4], format_func=lambda x: f"{x} Гц")

with st.sidebar.expander("💰 Экономические параметры", expanded=True):
    price = st.number_input("Тариф за литр (₸)", value=0.55, step=0.05, format="%.2f")
    limit = st.slider("Порог детекции утечки (bar)", 1.0, 5.0, 2.7, step=0.1)

st.sidebar.markdown("---")

if st.sidebar.button("🚀 ЗАПУСТИТЬ СИМУЛЯЦИЮ", use_container_width=True, type="primary"):
    with st.spinner("⏳ Расчет модели..."):
        try:
            data, net = run_epanet_simulation(m_types[material], iznos, freq)
            st.session_state['data'] = data
            st.session_state['network'] = net
            log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Симуляция завершена | {material}, Износ: {iznos}%"
            st.session_state['log'].append(log_entry)
            st.sidebar.success("✅ Готово!")
        except Exception as e:
            st.sidebar.error(f"❌ Ошибка: {str(e)}")

# --- MAIN CONTENT ---
st.title("💧 Smart Shygyn: AI Water Management System")
st.markdown("##### Интеллектуальная система мониторинга водоснабжения")

if st.session_state['data'] is not None:
    df = st.session_state['data']
    wn = st.session_state['network']
    
    # Detect leaks
    df['Leak'] = df['Pressure (bar)'] < limit
    active_leak = df['Leak'].any()
    
    # --- KPI DASHBOARD ---
    st.markdown("### 📊 Ключевые показатели")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if active_leak:
            st.metric(label="Статус системы", value="🚨 УТЕЧКА", delta="Критично", delta_color="inverse")
        else:
            st.metric(label="Статус системы", value="✅ НОРМА", delta="Стабильно", delta_color="normal")
    
    with col2:
        min_pressure = df['Pressure (bar)'].min()
        st.metric(
            label="Минимальное давление",
            value=f"{min_pressure:.2f} bar",
            delta=f"{min_pressure - limit:.2f}" if active_leak else None,
            delta_color="inverse"
        )
    
    with col3:
        lost_l = df[df['Leak']]['Flow Rate (L/s)'].sum() * (3600 / freq) if active_leak else 0
        st.metric(
            label="Потери воды",
            value=f"{lost_l:,.0f} L",
            delta="Критично" if lost_l > 10000 else None,
            delta_color="inverse"
        )
    
    with col4:
        damage = lost_l * price
        st.metric(
            label="Финансовый ущерб",
            value=f"{damage:,.0f} ₸",
            delta=f"-{damage:.0f} ₸" if damage > 0 else None,
            delta_color="inverse"
        )
    
    st.markdown("---")
    
    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["📈 Аналитика", "🗺️ Топология сети", "📋 Отчеты"])
    
    with tab1:
        st.markdown("### Гидравлические параметры системы")
        
        # Advanced plot
        fig = create_advanced_plot(df, limit)
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### 📉 Статистика давления")
            stats_p = df['Pressure (bar)'].describe()
            st.dataframe(stats_p.to_frame().style.format("{:.3f}"), use_container_width=True)
        
        with col_b:
            st.markdown("#### 📊 Статистика расхода")
            stats_f = df['Flow Rate (L/s)'].describe()
            st.dataframe(stats_f.to_frame().style.format("{:.3f}"), use_container_width=True)
        
        # Log
        if st.session_state['log']:
            with st.expander("📜 История операций"):
                for log in reversed(st.session_state['log'][-10:]):
                    st.code(log, language=None)
    
    with tab2:
        st.markdown("### Схема водопроводной сети")
        
        col_map, col_info = st.columns([2, 1])
        
        with col_map:
            fig_map, ax = plt.subplots(figsize=(10, 8), facecolor='white')
            pos = {n: wn.get_node(n).coordinates for n in wn.node_name_list}
            l_node = st.session_state['leak_node']
            
            # Node colors
            n_colors = []
            for n in wn.node_name_list:
                if n == l_node and active_leak:
                    n_colors.append('#e74c3c')  # Red for leak
                elif n == 'Res':
                    n_colors.append('#3498db')  # Blue for reservoir
                else:
                    n_colors.append('#2ecc71')  # Green for normal
            
            # Draw network
            nx.draw_networkx_edges(wn.get_graph(), pos, ax=ax, 
                                 edge_color='#95a5a6', width=3, alpha=0.6)
            nx.draw_networkx_nodes(wn.get_graph(), pos, ax=ax, 
                                 node_color=n_colors, node_size=500, 
                                 edgecolors='white', linewidths=2)
            nx.draw_networkx_labels(wn.get_graph(), pos, ax=ax, 
                                  font_size=8, font_weight='bold')
            
            ax.set_axis_off()
            ax.set_aspect('equal')
            plt.tight_layout()
            st.pyplot(fig_map)
        
        with col_info:
            st.markdown("#### 🔍 Анализ сети")
            
            st.info("**Резервуар (Res)**\n\n✅ Напор: Стабильный\n\n✅ Подача: Нормальная")
            
            if active_leak:
                st.error(f"**⚠️ УТЕЧКА ОБНАРУЖЕНА**\n\n"
                        f"📍 Узел: {l_node}\n\n"
                        f"⏰ Время: ~12:00\n\n"
                        f"🚨 Действие: Срочный выезд!")
            else:
                st.success(f"**✅ Утечек не обнаружено**\n\n"
                          f"🔍 Мониторинг: Активен\n\n"
                          f"📅 Плановый осмотр")
            
            st.markdown("#### 📊 Параметры сети")
            st.write(f"**Узлов:** {len(wn.node_name_list)}")
            st.write(f"**Труб:** {len(wn.link_name_list)}")
            st.write(f"**Материал:** {material}")
            st.write(f"**Износ:** {iznos}%")
    
    with tab3:
        st.markdown("### Экспорт данных и отчетность")
        
        col_r1, col_r2 = st.columns(2)
        
        with col_r1:
            st.markdown("#### 📊 Сводная таблица")
            st.dataframe(
                df.style.format({
                    'Hour': '{:.1f}',
                    'Pressure (bar)': '{:.3f}',
                    'Flow Rate (L/s)': '{:.3f}'
                }).background_gradient(cmap='RdYlGn', subset=['Pressure (bar)']),
                height=400,
                use_container_width=True
            )
        
        with col_r2:
            st.markdown("#### 📥 Формирование отчетов")
            
            report_data = df.copy()
            report_data['Status'] = report_data['Leak'].apply(lambda x: 'Утечка' if x else 'Норма')
            
            csv = report_data.to_csv(index=False, encoding='utf-8-sig')
            
            st.download_button(
                label="📄 Скачать CSV",
                data=csv,
                file_name=f"smart_shygyn_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            st.markdown("---")
            
            st.markdown("**Включить в отчет:**")
            inc_stats = st.checkbox("Статистические данные", value=True)
            inc_map = st.checkbox("Схему сети", value=False)
            inc_rec = st.checkbox("Рекомендации", value=True)
            
            if st.button("📧 Отправить в ЖКХ", use_container_width=True):
                st.success("✅ Отчет отправлен!")

else:
    # Welcome screen
    st.markdown("### 👋 Добро пожаловать в Smart Shygyn!")
    st.markdown("Настройте параметры системы в боковой панели и нажмите **'ЗАПУСТИТЬ СИМУЛЯЦИЮ'** для начала мониторинга.")
    
    st.markdown("---")
    
    col_w1, col_w2, col_w3 = st.columns(3)
    
    with col_w1:
        st.markdown("#### ⚙️ Настройка")
        st.markdown("- Выберите материал труб")
        st.markdown("- Укажите износ системы")
        st.markdown("- Установите частоту датчиков")
    
    with col_w2:
        st.markdown("#### 🚀 Симуляция")
        st.markdown("- Цифровая модель сети")
        st.markdown("- Реалистичная физика")
        st.markdown("- Детекция утечек")
    
    with col_w3:
        st.markdown("#### 📊 Анализ")
        st.markdown("- Визуализация данных")
        st.markdown("- Экономический расчет")
        st.markdown("- Формирование отчетов")
