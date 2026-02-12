import streamlit as st
import pandas as pd
import numpy as np
import wntr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from datetime import datetime
import io

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Smart Shygyn PRO - Expert Edition", 
    layout="wide", 
    page_icon="💧",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    [data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: 700;
    }
    
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
    
    .dataframe {
        font-size: 12px;
    }
    
    .stAlert {
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- BACKEND FUNCTIONS ---

def create_demand_pattern():
    """Создание суточного паттерна потребления (MNF учет)"""
    # Реалистичный паттерн потребления воды по часам
    hours = np.arange(24)
    pattern = []
    
    for h in hours:
        if 0 <= h < 6:  # Ночь (02:00-05:00 - минимум)
            pattern.append(0.3 + 0.1 * np.sin(h * np.pi / 6))
        elif 6 <= h < 9:  # Утренний пик
            pattern.append(1.2 + 0.3 * np.sin((h - 6) * np.pi / 3))
        elif 9 <= h < 18:  # День
            pattern.append(0.8 + 0.2 * np.sin((h - 9) * np.pi / 9))
        elif 18 <= h < 22:  # Вечерний пик
            pattern.append(1.4 + 0.2 * np.sin((h - 18) * np.pi / 4))
        else:  # Поздний вечер
            pattern.append(0.5 + 0.2 * np.sin((h - 22) * np.pi / 2))
    
    return pattern

def calculate_mnf_anomaly(df, expected_mnf=0.4):
    """Анализ ночного минимума (02:00-05:00)"""
    night_hours = df[(df['Hour'] >= 2) & (df['Hour'] <= 5)]
    if len(night_hours) == 0:
        return False, 0
    
    avg_night_flow = night_hours['Flow Rate (L/s)'].mean()
    anomaly = (avg_night_flow - expected_mnf) / expected_mnf * 100
    
    return anomaly > 15, anomaly

def calculate_failure_probability(pressure, degradation):
    """Вероятность отказа трубы (Predictive Analytics)"""
    # P_fail = α × (1 - P/P_max)^β × (D/100)^γ
    alpha = 0.5
    beta = 2.0
    gamma = 1.5
    p_max = 5.0
    
    p_fail = alpha * ((1 - pressure / p_max) ** beta) * ((degradation / 100) ** gamma)
    return min(p_fail * 100, 100)  # Процент

def find_isolation_valves(network, leak_node):
    """Поиск задвижек для изоляции участка"""
    graph = network.get_graph()
    
    # Находим соседние узлы
    neighbors = list(graph.neighbors(leak_node))
    
    # Находим трубы для изоляции
    pipes_to_close = []
    for neighbor in neighbors:
        for link_name in network.link_name_list:
            link = network.get_link(link_name)
            if hasattr(link, 'start_node_name') and hasattr(link, 'end_node_name'):
                if (link.start_node_name == leak_node and link.end_node_name == neighbor) or \
                   (link.end_node_name == leak_node and link.start_node_name == neighbor):
                    pipes_to_close.append(link_name)
    
    return pipes_to_close, neighbors

def run_epanet_simulation(material_c, degradation, sampling_rate, pump_pressure=40, add_valves=False):
    """Запуск симуляции с расширенным функционалом"""
    wn = wntr.network.WaterNetworkModel()
    dist = 100
    actual_diameter = 0.2 * (1 - degradation / 100)
    
    # Создание паттерна потребления
    demand_pattern = create_demand_pattern()
    pattern_name = 'daily_pattern'
    wn.add_pattern(pattern_name, demand_pattern)
    
    # Создание сети
    for i in range(4):
        for j in range(4):
            name = f"N_{i}_{j}"
            # Применяем паттерн потребления к узлам
            wn.add_junction(name, base_demand=0.001, elevation=10, demand_pattern=pattern_name)
            wn.get_node(name).coordinates = (i * dist, j * dist)
            
            if i > 0:
                pipe_name = f"PH_{i}_{j}"
                wn.add_pipe(pipe_name, f"N_{i-1}_{j}", name, 
                            length=dist, diameter=actual_diameter, roughness=material_c)
            if j > 0:
                pipe_name = f"PV_{i}_{j}"
                wn.add_pipe(pipe_name, f"N_{i}_{j-1}", name, 
                            length=dist, diameter=actual_diameter, roughness=material_c)
    
    # Резервуар с настраиваемым напором
    wn.add_reservoir('Res', base_head=pump_pressure)
    wn.get_node('Res').coordinates = (-dist, -dist)
    wn.add_pipe('P_Main', 'Res', 'N_0_0', length=dist, diameter=0.4, roughness=material_c)
    
    # Добавление задвижек (если требуется)
    if add_valves:
        valve_positions = [('N_1_1', 'N_2_1'), ('N_2_1', 'N_2_2'), ('N_2_2', 'N_2_3')]
        for i, (start, end) in enumerate(valve_positions):
            valve_name = f"Valve_{i+1}"
            # Находим трубу между узлами
            for link_name in wn.link_name_list:
                link = wn.get_link(link_name)
                if hasattr(link, 'start_node_name') and hasattr(link, 'end_node_name'):
                    if (link.start_node_name == start and link.end_node_name == end) or \
                       (link.end_node_name == start and link.start_node_name == end):
                        st.session_state[f'valve_{valve_name}'] = link_name
    
    # Утечка
    leak_node = "N_2_2"
    st.session_state['leak_node'] = leak_node
    
    # Настройка времени и качества воды
    wn.options.time.duration = 24 * 3600
    wn.options.time.report_timestep = 3600 // sampling_rate
    wn.options.quality.parameter = 'AGE'  # Отслеживание возраста воды
    
    # Добавление утечки
    node = wn.get_node(leak_node)
    node.add_leak(wn, area=0.08, start_time=12 * 3600)
    
    # Симуляция
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()
    
    # Извлечение результатов
    p = results.node['pressure'][leak_node] * 0.1 
    f = results.link['flowrate']['P_Main'] * 1000 
    
    # Возраст воды (качество)
    water_age = results.node['quality'][leak_node] / 3600  # В часах
    
    # Шум
    noise_p = np.random.normal(0, 0.04, len(p))
    noise_f = np.random.normal(0, 0.08, len(f))
    
    df_res = pd.DataFrame({
        'Hour': np.arange(len(p)) / sampling_rate,
        'Pressure (bar)': p.values + noise_p,
        'Flow Rate (L/s)': np.abs(f.values) + noise_f,
        'Water Age (h)': water_age.values,
        'Demand Pattern': np.tile(demand_pattern, len(p) // 24 + 1)[:len(p)]
    })
    
    return df_res, wn

def create_advanced_plot(df, threshold):
    """Профессиональный график с 3 подграфиками"""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('💧 Давление в системе', '🌊 Расход воды', '⏱️ Возраст воды (качество)'),
        vertical_spacing=0.1,
        row_heights=[0.35, 0.35, 0.3]
    )
    
    # График давления
    fig.add_trace(
        go.Scatter(
            x=df['Hour'],
            y=df['Pressure (bar)'],
            name='Давление',
            line=dict(color='#3498db', width=2.5),
            fill='tonexty',
            fillcolor='rgba(52, 152, 219, 0.15)',
            hovertemplate='<b>Час:</b> %{x:.1f}<br><b>Давление:</b> %{y:.2f} bar<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text="⚠️ Порог",
        row=1, col=1
    )
    
    # Критическая зона (риск заражения при P < 1.5 bar)
    fig.add_hrect(
        y0=0, y1=1.5,
        fillcolor="red", opacity=0.1,
        layer="below", line_width=0,
        annotation_text="Зона риска заражения",
        annotation_position="top left",
        row=1, col=1
    )
    
    # График расхода с паттерном потребления
    fig.add_trace(
        go.Scatter(
            x=df['Hour'],
            y=df['Flow Rate (L/s)'],
            name='Расход (реальный)',
            line=dict(color='#e67e22', width=2.5),
            hovertemplate='<b>Час:</b> %{x:.1f}<br><b>Расход:</b> %{y:.2f} L/s<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Ожидаемый расход (паттерн)
    expected_flow = df['Demand Pattern'] * df['Flow Rate (L/s)'].mean()
    fig.add_trace(
        go.Scatter(
            x=df['Hour'],
            y=expected_flow,
            name='Расход (ожидаемый)',
            line=dict(color='#27ae60', width=2, dash='dot'),
            hovertemplate='<b>Час:</b> %{x:.1f}<br><b>Ожидаемый:</b> %{y:.2f} L/s<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Выделение ночного минимума (MNF: 02:00-05:00)
    fig.add_vrect(
        x0=2, x1=5,
        fillcolor="blue", opacity=0.1,
        layer="below", line_width=0,
        annotation_text="MNF зона",
        annotation_position="top left",
        row=2, col=1
    )
    
    # График качества воды
    fig.add_trace(
        go.Scatter(
            x=df['Hour'],
            y=df['Water Age (h)'],
            name='Возраст воды',
            line=dict(color='#9b59b6', width=2.5),
            fill='tonexty',
            fillcolor='rgba(155, 89, 182, 0.15)',
            hovertemplate='<b>Час:</b> %{x:.1f}<br><b>Возраст:</b> %{y:.1f} ч<extra></extra>'
        ),
        row=3, col=1
    )
    
    # Оформление
    fig.update_xaxes(title_text="Время (часы)", row=3, col=1, gridcolor='lightgray')
    fig.update_xaxes(gridcolor='lightgray', row=1, col=1)
    fig.update_xaxes(gridcolor='lightgray', row=2, col=1)
    
    fig.update_yaxes(title_text="Давление (bar)", row=1, col=1, gridcolor='lightgray')
    fig.update_yaxes(title_text="Расход (L/s)", row=2, col=1, gridcolor='lightgray')
    fig.update_yaxes(title_text="Возраст (часы)", row=3, col=1, gridcolor='lightgray')
    
    fig.update_layout(
        height=900,
        showlegend=True,
        hovermode='x unified',
        plot_bgcolor='white',
        font=dict(size=11),
        margin=dict(l=60, r=60, t=80, b=60)
    )
    
    return fig

def create_heatmap_network(wn, df, degradation):
    """Тепловая карта вероятности отказа"""
    fig, ax = plt.subplots(figsize=(12, 10), facecolor='white')
    pos = {n: wn.get_node(n).coordinates for n in wn.node_name_list}
    
    # Расчет вероятности отказа для каждого узла
    failure_probs = {}
    node_colors = []
    
    avg_pressure = df['Pressure (bar)'].mean()
    
    for node in wn.node_name_list:
        if node != 'Res':
            prob = calculate_failure_probability(avg_pressure, degradation)
            failure_probs[node] = prob
            
            # Цветовая шкала по вероятности отказа
            if prob > 40:
                node_colors.append('#e74c3c')  # Красный - высокий риск
            elif prob > 25:
                node_colors.append('#f39c12')  # Оранжевый - средний риск
            elif prob > 15:
                node_colors.append('#f1c40f')  # Желтый - умеренный риск
            else:
                node_colors.append('#2ecc71')  # Зеленый - низкий риск
        else:
            node_colors.append('#3498db')  # Синий для резервуара
            failure_probs[node] = 0
    
    # Отрисовка
    nx.draw_networkx_edges(wn.get_graph(), pos, ax=ax, 
                         edge_color='#95a5a6', width=3, alpha=0.5)
    
    node_list = list(wn.node_name_list)
    
    # Рисуем узлы по отдельности
    for i, node in enumerate(node_list):
        x, y = pos[node]
        circle = plt.Circle((x, y), 18, color=node_colors[i], 
                          ec='white', linewidth=2.5, zorder=2)
        ax.add_patch(circle)
        ax.text(x, y, node, fontsize=8, fontweight='bold',
               ha='center', va='center', zorder=3)
    
    # Легенда
    legend_elements = [
        mpatches.Patch(color='#e74c3c', label='Высокий риск (>40%)'),
        mpatches.Patch(color='#f39c12', label='Средний риск (25-40%)'),
        mpatches.Patch(color='#f1c40f', label='Умеренный риск (15-25%)'),
        mpatches.Patch(color='#2ecc71', label='Низкий риск (<15%)'),
        mpatches.Patch(color='#3498db', label='Резервуар')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    ax.set_title('Тепловая карта вероятности отказа трубопроводов', fontsize=14, fontweight='bold')
    ax.set_axis_off()
    ax.set_aspect('equal')
    
    return fig, failure_probs

# --- SESSION STATE ---
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'network' not in st.session_state:
    st.session_state['network'] = None
if 'log' not in st.session_state:
    st.session_state['log'] = []
if 'isolated_pipes' not in st.session_state:
    st.session_state['isolated_pipes'] = []
if 'csv_data' not in st.session_state:
    st.session_state['csv_data'] = None

# --- SIDEBAR ---
st.sidebar.title("🧪 Экспертная панель")

with st.sidebar.expander("⚙️ Параметры сети", expanded=True):
    m_types = {
        "Пластик (ПНД)": 150, 
        "Сталь": 140, 
        "Чугун": 100
    }
    material = st.selectbox("Материал труб", list(m_types.keys()))
    iznos = st.slider("Износ системы (%)", 0, 60, 15, help="Процент деградации трубопровода")
    freq = st.select_slider("Частота датчиков", options=[1, 2, 4], format_func=lambda x: f"{x} Гц")

with st.sidebar.expander("🔧 Стресс-тест насоса", expanded=True):
    pump_pressure = st.slider("Напор насоса (м)", 30, 60, 40, step=5, 
                              help="Проверка устойчивости системы при изменении давления")
    st.info(f"💡 Текущий напор: **{pump_pressure} м** = **{pump_pressure * 0.098:.1f} bar**")

with st.sidebar.expander("💰 Экономика", expanded=True):
    price = st.number_input("Тариф за литр (₸)", value=0.55, step=0.05, format="%.2f")
    limit = st.slider("Порог детекции (bar)", 1.0, 5.0, 2.7, step=0.1)

with st.sidebar.expander("🔄 IoT интеграция", expanded=False):
    st.markdown("**Загрузка данных с реальных датчиков**")
    uploaded_file = st.file_uploader("Загрузить CSV", type=['csv'], help="Формат: Hour, Pressure, Flow Rate")
    
    if uploaded_file is not None:
        try:
            csv_df = pd.read_csv(uploaded_file)
            st.session_state['csv_data'] = csv_df
            st.success(f"✅ Загружено {len(csv_df)} записей")
        except Exception as e:
            st.error(f"❌ Ошибка: {str(e)}")

with st.sidebar.expander("🛡️ Управление задвижками", expanded=False):
    enable_valves = st.checkbox("Включить систему задвижек", value=False)
    st.info("При обнаружении утечки система предложит перекрыть участок")

st.sidebar.markdown("---")

if st.sidebar.button("🚀 ЗАПУСТИТЬ СИМУЛЯЦИЮ", use_container_width=True, type="primary"):
    with st.spinner("⏳ Расчет цифрового двойника..."):
        try:
            data, net = run_epanet_simulation(
                m_types[material], 
                iznos, 
                freq, 
                pump_pressure,
                enable_valves
            )
            st.session_state['data'] = data
            st.session_state['network'] = net
            st.session_state['isolated_pipes'] = []
            
            log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Симуляция | {material}, Износ: {iznos}%, Напор: {pump_pressure}м"
            st.session_state['log'].append(log_entry)
            st.sidebar.success("✅ Готово!")
        except Exception as e:
            st.sidebar.error(f"❌ Ошибка: {str(e)}")

# --- MAIN CONTENT ---
st.title("💧 Smart Shygyn PRO: Expert Water Management System")
st.markdown("##### Профессиональная система мониторинга с MNF, изоляцией участков и прогнозной аналитикой")

if st.session_state['data'] is not None:
    df = st.session_state['data']
    wn = st.session_state['network']
    
    # Детекция утечек
    df['Leak'] = df['Pressure (bar)'] < limit
    active_leak = df['Leak'].any()
    
    # MNF анализ
    mnf_detected, mnf_anomaly = calculate_mnf_anomaly(df)
    
    # Зона риска заражения
    contamination_risk = (df['Pressure (bar)'] < 1.5).any()
    
    # --- KPI DASHBOARD ---
    st.markdown("### 📊 Панель состояния системы")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if active_leak:
            st.metric(label="🚨 Статус", value="УТЕЧКА", delta="Критично", delta_color="inverse")
        else:
            st.metric(label="✅ Статус", value="НОРМА", delta="Стабильно", delta_color="normal")
    
    with col2:
        min_pressure = df['Pressure (bar)'].min()
        st.metric(
            label="Давление min",
            value=f"{min_pressure:.2f} bar",
            delta=f"{min_pressure - limit:.2f}",
            delta_color="inverse" if min_pressure < limit else "normal"
        )
    
    with col3:
        lost_l = df[df['Leak']]['Flow Rate (L/s)'].sum() * (3600 / freq) if active_leak else 0
        st.metric(
            label="Потери воды",
            value=f"{lost_l:,.0f} L",
            delta="⚠️" if lost_l > 5000 else None
        )
    
    with col4:
        damage = lost_l * price
        st.metric(
            label="Ущерб",
            value=f"{damage:,.0f} ₸",
            delta=f"-{damage:.0f}" if damage > 0 else None,
            delta_color="inverse"
        )
    
    with col5:
        if mnf_detected:
            st.metric(label="MNF аномалия", value=f"+{mnf_anomaly:.1f}%", delta="Скрытая утечка", delta_color="inverse")
        else:
            st.metric(label="MNF статус", value="Норма", delta=f"{mnf_anomaly:.1f}%", delta_color="normal")
    
    # Предупреждения
    if contamination_risk:
        st.error("⚠️ **ОПАСНОСТЬ ИНФИЛЬТРАЦИИ!** Давление упало ниже 1.5 bar. Риск загрязнения грунтовыми водами!")
    
    if mnf_detected:
        st.warning(f"🔍 **MNF АНОМАЛИЯ:** Ночной расход превышает норму на {mnf_anomaly:.1f}%. Возможна скрытая утечка.")
    
    st.markdown("---")
    
    # --- TABS ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Гидравлика", 
        "🗺️ Топология", 
        "🔥 Риск-карта", 
        "🔄 IoT данные",
        "📋 Отчеты"
    ])
    
    with tab1:
        st.markdown("### Расширенный анализ гидравлических параметров")
        
        # Продвинутый график
        fig = create_advanced_plot(df, limit)
        st.plotly_chart(fig, use_container_width=True)
        
        # Статистика
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.markdown("#### 💧 Давление")
            stats_p = df['Pressure (bar)'].describe()
            st.dataframe(stats_p.to_frame().style.format("{:.3f}"), use_container_width=True)
        
        with col_b:
            st.markdown("#### 🌊 Расход")
            stats_f = df['Flow Rate (L/s)'].describe()
            st.dataframe(stats_f.to_frame().style.format("{:.3f}"), use_container_width=True)
        
        with col_c:
            st.markdown("#### ⏱️ Качество")
            stats_age = df['Water Age (h)'].describe()
            st.dataframe(stats_age.to_frame().style.format("{:.2f}"), use_container_width=True)
        
        # Лог событий
        if st.session_state['log']:
            with st.expander("📜 История операций"):
                for log in reversed(st.session_state['log'][-15:]):
                    st.code(log, language=None)
    
    with tab2:
        st.markdown("### Схема сети с системой изоляции")
        
        col_map, col_control = st.columns([2, 1])
        
        with col_map:
            fig_map, ax = plt.subplots(figsize=(11, 9), facecolor='white')
            pos = {n: wn.get_node(n).coordinates for n in wn.node_name_list}
            l_node = st.session_state['leak_node']
            
            # Цвета узлов
            n_colors = []
            for n in wn.node_name_list:
                if n == l_node and active_leak:
                    n_colors.append('#e74c3c')
                elif n == 'Res':
                    n_colors.append('#3498db')
                else:
                    n_colors.append('#2ecc71')
            
            # Рисуем трубы
            edges = wn.get_graph().edges()
            for edge in edges:
                start_pos = pos[edge[0]]
                end_pos = pos[edge[1]]
                
                # Проверяем, изолирована ли труба
                is_isolated = any(
                    (edge[0] in pipe or edge[1] in pipe) 
                    for pipe in st.session_state['isolated_pipes']
                )
                
                color = '#2c3e50' if is_isolated else '#95a5a6'
                width = 4 if is_isolated else 3
                alpha = 1.0 if is_isolated else 0.5
                
                ax.plot([start_pos[0], end_pos[0]], 
                       [start_pos[1], end_pos[1]], 
                       color=color, linewidth=width, alpha=alpha, zorder=1)
            
            # Рисуем узлы
            node_list = list(wn.node_name_list)
            
            # Рисуем узлы по отдельности для большей совместимости
            for i, node in enumerate(node_list):
                x, y = pos[node]
                circle = plt.Circle((x, y), 15, color=n_colors[i], 
                                  ec='white', linewidth=2.5, zorder=2)
                ax.add_patch(circle)
                ax.text(x, y, node, fontsize=8, fontweight='bold',
                       ha='center', va='center', zorder=3)
            
            # Легенда
            legend_elements = [
                mpatches.Patch(color='#e74c3c', label='Утечка'),
                mpatches.Patch(color='#3498db', label='Резервуар'),
                mpatches.Patch(color='#2ecc71', label='Норма'),
                mpatches.Patch(color='#2c3e50', label='Изолировано')
            ]
            ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
            
            ax.set_title('Топология сети с задвижками', fontsize=13, fontweight='bold')
            ax.set_axis_off()
            ax.set_aspect('equal')
            plt.tight_layout()
            st.pyplot(fig_map)
        
        with col_control:
            st.markdown("#### 🛡️ Система изоляции")
            
            if active_leak:
                st.error(f"**⚠️ УТЕЧКА В УЗЛЕ {st.session_state['leak_node']}**")
                
                if st.button("🔒 ПЕРЕКРЫТЬ УЧАСТОК", use_container_width=True, type="primary"):
                    pipes_to_close, affected_nodes = find_isolation_valves(wn, st.session_state['leak_node'])
                    st.session_state['isolated_pipes'] = pipes_to_close
                    
                    log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] 🔒 Изолировано труб: {len(pipes_to_close)}"
                    st.session_state['log'].append(log_entry)
                    st.rerun()
                
                if st.session_state['isolated_pipes']:
                    st.success(f"✅ **Участок изолирован**")
                    st.write(f"Перекрыто труб: **{len(st.session_state['isolated_pipes'])}**")
                    
                    # Расчет жителей без воды
                    affected = len(affected_nodes) * 250  # Примерно 250 человек на узел
                    st.write(f"Затронуто жителей: **~{affected}**")
                    
                    if st.button("🔓 Восстановить подачу"):
                        st.session_state['isolated_pipes'] = []
                        st.rerun()
            else:
                st.success("✅ **Система в норме**")
                st.info("Система задвижек в режиме ожидания")
            
            st.markdown("---")
            st.markdown("#### 📊 Параметры")
            st.write(f"**Узлов:** {len(wn.node_name_list)}")
            st.write(f"**Труб:** {len(wn.link_name_list)}")
            st.write(f"**Материал:** {material}")
            st.write(f"**Износ:** {iznos}%")
            st.write(f"**Напор:** {pump_pressure} м")
    
    with tab3:
        st.markdown("### Прогнозная аналитика отказов (Predictive Maintenance)")
        
        fig_heat, fail_probs = create_heatmap_network(wn, df, iznos)
        st.pyplot(fig_heat)
        
        st.markdown("#### 📊 Вероятность отказа по узлам")
        
        # Топ-5 узлов риска
        sorted_probs = sorted(
            [(k, v) for k, v in fail_probs.items() if k != 'Res'], 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        col_r1, col_r2 = st.columns(2)
        
        with col_r1:
            st.markdown("**🔴 Топ-5 узлов высокого риска:**")
            for i, (node, prob) in enumerate(sorted_probs, 1):
                color = "🔴" if prob > 40 else "🟠" if prob > 25 else "🟡"
                st.write(f"{i}. {color} **{node}** — {prob:.1f}% риска")
        
        with col_r2:
            st.markdown("**💡 Рекомендации:**")
            if sorted_probs and sorted_probs[0][1] > 40:
                st.error("⚠️ Срочная замена труб в узлах высокого риска!")
            elif sorted_probs and sorted_probs[0][1] > 25:
                st.warning("📋 Плановая замена в течение 6 месяцев")
            else:
                st.success("✅ Система в удовлетворительном состоянии")
            
            st.info(f"**Стресс-тест:** При напоре {pump_pressure}м система {'выдерживает' if pump_pressure <= 50 else 'перегружена'}")
    
    with tab4:
        st.markdown("### IoT интеграция и сравнение с моделью")
        
        if st.session_state['csv_data'] is not None:
            csv_df = st.session_state['csv_data']
            
            # Сравнение модели и реальных данных
            if 'Pressure (bar)' in csv_df.columns and 'Flow Rate (L/s)' in csv_df.columns:
                fig_compare = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Сравнение давления', 'Сравнение расхода'),
                    vertical_spacing=0.12
                )
                
                # Давление
                fig_compare.add_trace(
                    go.Scatter(x=df['Hour'], y=df['Pressure (bar)'], 
                             name='Модель', line=dict(color='blue', dash='dot')),
                    row=1, col=1
                )
                fig_compare.add_trace(
                    go.Scatter(x=csv_df['Hour'], y=csv_df['Pressure (bar)'], 
                             name='Датчики', line=dict(color='red')),
                    row=1, col=1
                )
                
                # Расход
                fig_compare.add_trace(
                    go.Scatter(x=df['Hour'], y=df['Flow Rate (L/s)'], 
                             name='Модель', line=dict(color='blue', dash='dot')),
                    row=2, col=1
                )
                fig_compare.add_trace(
                    go.Scatter(x=csv_df['Hour'], y=csv_df['Flow Rate (L/s)'], 
                             name='Датчики', line=dict(color='red')),
                    row=2, col=1
                )
                
                fig_compare.update_xaxes(title_text="Время (часы)", row=2, col=1)
                fig_compare.update_yaxes(title_text="Давление (bar)", row=1, col=1)
                fig_compare.update_yaxes(title_text="Расход (L/s)", row=2, col=1)
                fig_compare.update_layout(height=700, showlegend=True)
                
                st.plotly_chart(fig_compare, use_container_width=True)
                
                # Residual анализ
                st.markdown("#### 📉 Анализ отклонений (Residuals)")
                
                # Ресемплинг для совпадения длины
                if len(csv_df) == len(df):
                    residual_p = csv_df['Pressure (bar)'].values - df['Pressure (bar)'].values
                    residual_f = csv_df['Flow Rate (L/s)'].values - df['Flow Rate (L/s)'].values
                    
                    col_res1, col_res2 = st.columns(2)
                    
                    with col_res1:
                        st.metric("Макс. отклонение давления", f"{np.max(np.abs(residual_p)):.3f} bar")
                        st.metric("Средн. отклонение давления", f"{np.mean(np.abs(residual_p)):.3f} bar")
                    
                    with col_res2:
                        st.metric("Макс. отклонение расхода", f"{np.max(np.abs(residual_f)):.3f} L/s")
                        st.metric("Средн. отклонение расхода", f"{np.mean(np.abs(residual_f)):.3f} L/s")
                    
                    if np.max(np.abs(residual_p)) > 0.5:
                        st.error("⚠️ Значительное расхождение с моделью! Возможна аномалия в сети.")
                else:
                    st.warning("⚠️ Длина данных не совпадает. Загрузите CSV с тем же временным диапазоном.")
            else:
                st.error("❌ CSV должен содержать колонки: Hour, Pressure (bar), Flow Rate (L/s)")
        else:
            st.info("📁 Загрузите CSV файл с данными датчиков в боковой панели для сравнения с моделью")
            
            st.markdown("**Пример формата CSV:**")
            example_csv = pd.DataFrame({
                'Hour': [0, 1, 2, 3, 4],
                'Pressure (bar)': [3.2, 3.1, 2.9, 2.8, 2.7],
                'Flow Rate (L/s)': [1.2, 1.1, 0.9, 0.8, 0.85]
            })
            st.dataframe(example_csv)
    
    with tab5:
        st.markdown("### Экспорт и отчетность")
        
        col_r1, col_r2 = st.columns([3, 2])
        
        with col_r1:
            st.markdown("#### 📊 Полная таблица данных")
            
            display_df = df.copy()
            display_df['Status'] = display_df['Leak'].apply(lambda x: '🚨 Утечка' if x else '✅ Норма')
            display_df['Risk'] = display_df['Pressure (bar)'].apply(
                lambda x: '⚠️ Риск' if x < 1.5 else '✅ Норма'
            )
            
            st.dataframe(
                display_df.style.format({
                    'Hour': '{:.1f}',
                    'Pressure (bar)': '{:.3f}',
                    'Flow Rate (L/s)': '{:.3f}',
                    'Water Age (h)': '{:.2f}',
                    'Demand Pattern': '{:.3f}'
                }).background_gradient(cmap='RdYlGn', subset=['Pressure (bar)']),
                height=450,
                use_container_width=True
            )
        
        with col_r2:
            st.markdown("#### 📥 Генерация отчетов")
            
            # Опции отчета
            inc_mnf = st.checkbox("MNF анализ", value=True)
            inc_risk = st.checkbox("Карта рисков", value=True)
            inc_quality = st.checkbox("Качество воды", value=True)
            inc_isolation = st.checkbox("План изоляции", value=st.session_state['isolated_pipes'] != [])
            
            # CSV экспорт
            report_data = display_df.to_csv(index=False, encoding='utf-8-sig')
            
            st.download_button(
                label="📄 Скачать полный отчет CSV",
                data=report_data,
                file_name=f"smart_shygyn_expert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            st.markdown("---")
            
            # Краткая сводка
            st.markdown("**📋 Краткая сводка:**")
            st.write(f"• Статус: {'🚨 Утечка' if active_leak else '✅ Норма'}")
            st.write(f"• MNF: {'⚠️ Аномалия' if mnf_detected else '✅ Норма'}")
            st.write(f"• Риск заражения: {'⚠️ Да' if contamination_risk else '✅ Нет'}")
            st.write(f"• Потери: {lost_l:,.0f} L")
            st.write(f"• Ущерб: {damage:,.0f} ₸")
            
            if st.button("📧 Отправить в ЖКХ", use_container_width=True, type="primary"):
                st.success("✅ Отчет отправлен на систему управления!")
                log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] 📧 Отчет отправлен в ЖКХ"
                st.session_state['log'].append(log_entry)

else:
    # Welcome screen
    st.markdown("### 👋 Добро пожаловать в Smart Shygyn Expert Edition!")
    st.markdown("Профессиональная система с модулями: **MNF анализ** • **Зональная изоляция** • **Качество воды** • **Прогнозная аналитика** • **IoT интеграция**")
    
    st.markdown("---")
    
    col_w1, col_w2, col_w3, col_w4 = st.columns(4)
    
    with col_w1:
        st.markdown("#### 🌙 MNF анализ")
        st.markdown("- Ночной минимум")
        st.markdown("- Скрытые утечки")
        st.markdown("- Паттерн потребления")
    
    with col_w2:
        st.markdown("#### 🛡️ Изоляция")
        st.markdown("- Автопоиск задвижек")
        st.markdown("- Минимизация ущерба")
        st.markdown("- Контроль участков")
    
    with col_w3:
        st.markdown("#### 💧 Качество")
        st.markdown("- Возраст воды")
        st.markdown("- Риск заражения")
        st.markdown("- Санитарный контроль")
    
    with col_w4:
        st.markdown("#### 🔮 Прогноз")
        st.markdown("- Вероятность отказа")
        st.markdown("- Тепловая карта")
        st.markdown("- План замены труб")
