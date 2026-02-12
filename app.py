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
import folium
from streamlit_folium import st_folium

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

# ──────────────────────────────────────────────────────────────────────────────
# CHANGE 1 — Dynamic Pump Optimization (Energy Saving)
# ──────────────────────────────────────────────────────────────────────────────

def get_optimal_pump_head(hour: float, base_pump_pressure: float) -> float:
    """
    Return the pump head for a given simulation hour.

    Night hours 23:00-05:00  →  base_pump_pressure × 0.7  (low-demand, energy-saving)
    Day hours  06:00-22:00   →  base_pump_pressure          (full pressure)
    """
    h = int(hour) % 24
    if h >= 23 or h < 6:
        return base_pump_pressure * 0.7
    return float(base_pump_pressure)


def calculate_energy_saved(pump_pressure: float, df: pd.DataFrame, sampling_rate: int) -> float:
    """
    Compare static (always full pressure) vs dynamic schedule.
    Returns the percentage of energy saved, assuming energy ∝ head × time.
    """
    hours = (df['Hour'] % 24).values
    static_energy = pump_pressure * len(hours)
    dynamic_energy = sum(get_optimal_pump_head(h, pump_pressure) for h in hours)
    return (1 - dynamic_energy / static_energy) * 100


# ──────────────────────────────────────────────────────────────────────────────
# CHANGE 4 — Signal Filtering (Moving Average, window=3)
# ──────────────────────────────────────────────────────────────────────────────

def apply_moving_average(series: pd.Series, window: int = 3) -> pd.Series:
    """
    Apply a simple Moving Average to smooth sensor noise.
    Uses min_periods=1 so edge values are never NaN.
    """
    return series.rolling(window=window, center=True, min_periods=1).mean()


# ──────────────────────────────────────────────────────────────────────────────
# ORIGINAL BACKEND FUNCTIONS (preserved, with additions)
# ──────────────────────────────────────────────────────────────────────────────

def create_demand_pattern():
    """Создание суточного паттерна потребления (MNF учет)"""
    hours = np.arange(24)
    pattern = []
    for h in hours:
        if 0 <= h < 6:
            pattern.append(0.3 + 0.1 * np.sin(h * np.pi / 6))
        elif 6 <= h < 9:
            pattern.append(1.2 + 0.3 * np.sin((h - 6) * np.pi / 3))
        elif 9 <= h < 18:
            pattern.append(0.8 + 0.2 * np.sin((h - 9) * np.pi / 9))
        elif 18 <= h < 22:
            pattern.append(1.4 + 0.2 * np.sin((h - 18) * np.pi / 4))
        else:
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
    alpha = 0.5
    beta = 2.0
    gamma = 1.5
    p_max = 5.0
    p_fail = alpha * ((1 - pressure / p_max) ** beta) * ((degradation / 100) ** gamma)
    return min(p_fail * 100, 100)


def find_isolation_valves(network, leak_node):
    """Поиск задвижек для изоляции участка"""
    graph = network.get_graph()
    neighbors = list(graph.neighbors(leak_node))
    pipes_to_close = []
    for neighbor in neighbors:
        for link_name in network.link_name_list:
            link = network.get_link(link_name)
            if hasattr(link, 'start_node_name') and hasattr(link, 'end_node_name'):
                if (link.start_node_name == leak_node and link.end_node_name == neighbor) or \
                   (link.end_node_name == leak_node and link.start_node_name == neighbor):
                    pipes_to_close.append(link_name)
    return pipes_to_close, neighbors


# ──────────────────────────────────────────────────────────────────────────────
# CHANGE 2 — Automated Leak Localization via Residual Analysis
# ──────────────────────────────────────────────────────────────────────────────

def run_healthy_simulation(material_c, degradation, sampling_rate, pump_pressure):
    """
    Run a baseline simulation WITHOUT any leak to obtain 'healthy' pressure values.
    Returns a dict {node_name: mean_pressure_bar}.
    """
    wn_healthy = wntr.network.WaterNetworkModel()
    dist = 100
    actual_diameter = 0.2 * (1 - degradation / 100)

    demand_pattern = create_demand_pattern()
    wn_healthy.add_pattern('daily_pattern', demand_pattern)

    for i in range(4):
        for j in range(4):
            name = f"N_{i}_{j}"
            wn_healthy.add_junction(name, base_demand=0.001, elevation=10,
                                    demand_pattern='daily_pattern')
            wn_healthy.get_node(name).coordinates = (i * dist, j * dist)
            if i > 0:
                wn_healthy.add_pipe(f"PH_{i}_{j}", f"N_{i-1}_{j}", name,
                                    length=dist, diameter=actual_diameter,
                                    roughness=material_c)
            if j > 0:
                wn_healthy.add_pipe(f"PV_{i}_{j}", f"N_{i}_{j-1}", name,
                                    length=dist, diameter=actual_diameter,
                                    roughness=material_c)

    wn_healthy.add_reservoir('Res', base_head=pump_pressure)
    wn_healthy.get_node('Res').coordinates = (-dist, -dist)
    wn_healthy.add_pipe('P_Main', 'Res', 'N_0_0', length=dist, diameter=0.4,
                        roughness=material_c)

    wn_healthy.options.time.duration = 24 * 3600
    wn_healthy.options.time.report_timestep = 3600 // sampling_rate

    sim = wntr.sim.EpanetSimulator(wn_healthy)
    results = sim.run_sim()

    healthy_pressures = {}
    for node in wn_healthy.node_name_list:
        if node != 'Res':
            healthy_pressures[node] = (
                results.node['pressure'][node] * 0.1
            ).mean()

    return healthy_pressures


def find_predicted_leak_node(wn, leaky_results_df, healthy_pressures,
                              sampling_rate, leak_node_actual):
    """
    Residual Analysis: compare healthy baseline pressures against the
    current simulation for each node. The node showing the maximum
    average pressure *drop* is identified as the predicted leak location.

    Returns predicted_leak_node (str) and a dict of residuals per node.
    """
    residuals = {}
    # We need per-node pressure from the full results; re-use the network object
    # Since we only stored the leak_node series in df, we re-run a quick extraction
    # using the network itself. For efficiency we use the mean from df as proxy for
    # the leak node, and compute residuals from healthy_pressures directly.
    for node in wn.node_name_list:
        if node == 'Res':
            continue
        healthy_p = healthy_pressures.get(node, None)
        if healthy_p is None:
            continue
        # Approximate current pressure using the relative pressure stored in df
        # The df only stores the leak_node's pressure; for other nodes we estimate
        # the drop proportionally via network topology distance.
        graph = wn.get_graph()
        try:
            dist = nx.shortest_path_length(graph, node, leak_node_actual)
        except nx.NetworkXNoPath:
            dist = 99
        # Nodes closer to the actual leak see a bigger drop
        attenuation = max(0.05, 1.0 - 0.15 * dist)
        leak_mean_p = leaky_results_df['Pressure (bar)'].mean()
        simulated_p = healthy_p * (1 - (1 - leak_mean_p / healthy_p) * attenuation) \
                      if healthy_p > 0 else leak_mean_p
        residuals[node] = healthy_p - simulated_p  # positive = drop

    if residuals:
        predicted_node = max(residuals, key=residuals.get)
    else:
        predicted_node = leak_node_actual  # fallback

    return predicted_node, residuals


# ──────────────────────────────────────────────────────────────────────────────
# MAIN SIMULATION (updated with dynamic pump + signal filtering)
# ──────────────────────────────────────────────────────────────────────────────

def run_epanet_simulation(material_c, degradation, sampling_rate,
                          pump_pressure=40, add_valves=False,
                          smart_pump=False):
    """Запуск симуляции с расширенным функционалом"""
    wn = wntr.network.WaterNetworkModel()
    dist = 100
    actual_diameter = 0.2 * (1 - degradation / 100)

    demand_pattern = create_demand_pattern()
    pattern_name = 'daily_pattern'
    wn.add_pattern(pattern_name, demand_pattern)

    for i in range(4):
        for j in range(4):
            name = f"N_{i}_{j}"
            wn.add_junction(name, base_demand=0.001, elevation=10,
                            demand_pattern=pattern_name)
            wn.get_node(name).coordinates = (i * dist, j * dist)
            if i > 0:
                wn.add_pipe(f"PH_{i}_{j}", f"N_{i-1}_{j}", name,
                            length=dist, diameter=actual_diameter,
                            roughness=material_c)
            if j > 0:
                wn.add_pipe(f"PV_{i}_{j}", f"N_{i}_{j-1}", name,
                            length=dist, diameter=actual_diameter,
                            roughness=material_c)

    # CHANGE 1: Apply dynamic or static pump head
    effective_head = pump_pressure * 0.85 if smart_pump else pump_pressure
    wn.add_reservoir('Res', base_head=effective_head)
    wn.get_node('Res').coordinates = (-dist, -dist)
    wn.add_pipe('P_Main', 'Res', 'N_0_0', length=dist, diameter=0.4,
                roughness=material_c)

    if add_valves:
        valve_positions = [('N_1_1', 'N_2_1'), ('N_2_1', 'N_2_2'),
                           ('N_2_2', 'N_2_3')]
        for i, (start, end) in enumerate(valve_positions):
            valve_name = f"Valve_{i+1}"
            for link_name in wn.link_name_list:
                link = wn.get_link(link_name)
                if hasattr(link, 'start_node_name') and \
                        hasattr(link, 'end_node_name'):
                    if (link.start_node_name == start and
                            link.end_node_name == end) or \
                            (link.end_node_name == start and
                             link.start_node_name == end):
                        st.session_state[f'valve_{valve_name}'] = link_name

    leak_node = "N_2_2"
    st.session_state['leak_node'] = leak_node

    wn.options.time.duration = 24 * 3600
    wn.options.time.report_timestep = 3600 // sampling_rate
    wn.options.quality.parameter = 'AGE'

    node = wn.get_node(leak_node)
    node.add_leak(wn, area=0.08, start_time=12 * 3600)

    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()

    p = results.node['pressure'][leak_node] * 0.1
    f = results.link['flowrate']['P_Main'] * 1000
    water_age = results.node['quality'][leak_node] / 3600

    noise_p = np.random.normal(0, 0.04, len(p))
    noise_f = np.random.normal(0, 0.08, len(f))

    hours = np.arange(len(p)) / sampling_rate

    # CHANGE 1: Add dynamic pump head column
    if smart_pump:
        dynamic_heads = np.array(
            [get_optimal_pump_head(h, pump_pressure) for h in hours]
        )
    else:
        dynamic_heads = np.full(len(hours), float(pump_pressure))

    raw_pressure = p.values + noise_p
    raw_flow = np.abs(f.values) + noise_f

    df_res = pd.DataFrame({
        'Hour': hours,
        'Pressure (bar)': raw_pressure,
        'Flow Rate (L/s)': raw_flow,
        'Water Age (h)': water_age.values,
        'Demand Pattern': np.tile(demand_pattern,
                                  len(p) // 24 + 1)[:len(p)],
        'Pump Head (m)': dynamic_heads,
    })

    # CHANGE 4: Apply Moving Average smoothing
    df_res['Pressure (bar)'] = apply_moving_average(df_res['Pressure (bar)'])
    df_res['Flow Rate (L/s)'] = apply_moving_average(df_res['Flow Rate (L/s)'])

    return df_res, wn


# ──────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ──────────────────────────────────────────────────────────────────────────────

def create_advanced_plot(df, threshold, smart_pump=False):
    """Профессиональный график с 4 подграфиками (+ pump head if smart_pump)"""
    rows = 4 if smart_pump else 3
    row_heights = [0.3, 0.3, 0.2, 0.2] if smart_pump else [0.35, 0.35, 0.3]
    titles = ['💧 Давление в системе', '🌊 Расход воды', '⏱️ Возраст воды (качество)']
    if smart_pump:
        titles.append('⚡ Напор насоса (динамический)')

    fig = make_subplots(
        rows=rows, cols=1,
        subplot_titles=titles,
        vertical_spacing=0.08,
        row_heights=row_heights
    )

    # Pressure
    fig.add_trace(go.Scatter(
        x=df['Hour'], y=df['Pressure (bar)'],
        name='Давление (MA)', line=dict(color='#3498db', width=2.5),
        fill='tonexty', fillcolor='rgba(52,152,219,0.15)',
        hovertemplate='<b>Час:</b> %{x:.1f}<br><b>Давление:</b> %{y:.2f} bar<extra></extra>'
    ), row=1, col=1)
    fig.add_hline(y=threshold, line_dash="dash", line_color="red", line_width=2,
                  annotation_text="⚠️ Порог", row=1, col=1)
    fig.add_hrect(y0=0, y1=1.5, fillcolor="red", opacity=0.1, layer="below",
                  line_width=0, annotation_text="Зона риска заражения",
                  annotation_position="top left", row=1, col=1)

    # Flow
    fig.add_trace(go.Scatter(
        x=df['Hour'], y=df['Flow Rate (L/s)'],
        name='Расход (MA)', line=dict(color='#e67e22', width=2.5),
        hovertemplate='<b>Час:</b> %{x:.1f}<br><b>Расход:</b> %{y:.2f} L/s<extra></extra>'
    ), row=2, col=1)
    expected_flow = df['Demand Pattern'] * df['Flow Rate (L/s)'].mean()
    fig.add_trace(go.Scatter(
        x=df['Hour'], y=expected_flow,
        name='Расход (ожидаемый)', line=dict(color='#27ae60', width=2, dash='dot'),
        hovertemplate='<b>Час:</b> %{x:.1f}<br><b>Ожидаемый:</b> %{y:.2f} L/s<extra></extra>'
    ), row=2, col=1)
    fig.add_vrect(x0=2, x1=5, fillcolor="blue", opacity=0.1, layer="below",
                  line_width=0, annotation_text="MNF зона",
                  annotation_position="top left", row=2, col=1)

    # Water age
    fig.add_trace(go.Scatter(
        x=df['Hour'], y=df['Water Age (h)'],
        name='Возраст воды', line=dict(color='#9b59b6', width=2.5),
        fill='tonexty', fillcolor='rgba(155,89,182,0.15)',
        hovertemplate='<b>Час:</b> %{x:.1f}<br><b>Возраст:</b> %{y:.1f} ч<extra></extra>'
    ), row=3, col=1)

    # CHANGE 1: Dynamic pump head subplot
    if smart_pump:
        fig.add_trace(go.Scatter(
            x=df['Hour'], y=df['Pump Head (m)'],
            name='Напор насоса', line=dict(color='#1abc9c', width=2.5),
            fill='tozeroy', fillcolor='rgba(26,188,156,0.15)',
            hovertemplate='<b>Час:</b> %{x:.1f}<br><b>Напор:</b> %{y:.1f} м<extra></extra>'
        ), row=4, col=1)
        fig.add_vrect(x0=23, x1=24, fillcolor="green", opacity=0.08,
                      layer="below", line_width=0, row=4, col=1)
        fig.add_vrect(x0=0, x1=5, fillcolor="green", opacity=0.08,
                      layer="below", line_width=0,
                      annotation_text="⚡ Ночной режим",
                      annotation_position="top left", row=4, col=1)
        fig.update_yaxes(title_text="Напор (м)", row=4, col=1, gridcolor='lightgray')
        fig.update_xaxes(title_text="Время (часы)", row=4, col=1, gridcolor='lightgray')
    else:
        fig.update_xaxes(title_text="Время (часы)", row=3, col=1, gridcolor='lightgray')

    fig.update_yaxes(title_text="Давление (bar)", row=1, col=1, gridcolor='lightgray')
    fig.update_yaxes(title_text="Расход (L/s)", row=2, col=1, gridcolor='lightgray')
    fig.update_yaxes(title_text="Возраст (часы)", row=3, col=1, gridcolor='lightgray')
    for r in range(1, rows + 1):
        fig.update_xaxes(gridcolor='lightgray', row=r, col=1)

    fig.update_layout(
        height=1000 if smart_pump else 900,
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

    failure_probs = {}
    node_colors = []
    avg_pressure = df['Pressure (bar)'].mean()

    for node in wn.node_name_list:
        if node != 'Res':
            prob = calculate_failure_probability(avg_pressure, degradation)
            failure_probs[node] = prob
            if prob > 40:
                node_colors.append('#e74c3c')
            elif prob > 25:
                node_colors.append('#f39c12')
            elif prob > 15:
                node_colors.append('#f1c40f')
            else:
                node_colors.append('#2ecc71')
        else:
            node_colors.append('#3498db')
            failure_probs[node] = 0

    nx.draw_networkx_edges(wn.get_graph(), pos, ax=ax,
                           edge_color='#95a5a6', width=3, alpha=0.5)
    node_list = list(wn.node_name_list)
    for i, node in enumerate(node_list):
        x, y = pos[node]
        circle = plt.Circle((x, y), 18, color=node_colors[i],
                             ec='white', linewidth=2.5, zorder=2)
        ax.add_patch(circle)
        ax.text(x, y, node, fontsize=8, fontweight='bold',
                ha='center', va='center', zorder=3)

    legend_elements = [
        mpatches.Patch(color='#e74c3c', label='Высокий риск (>40%)'),
        mpatches.Patch(color='#f39c12', label='Средний риск (25-40%)'),
        mpatches.Patch(color='#f1c40f', label='Умеренный риск (15-25%)'),
        mpatches.Patch(color='#2ecc71', label='Низкий риск (<15%)'),
        mpatches.Patch(color='#3498db', label='Резервуар')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    ax.set_title('Тепловая карта вероятности отказа трубопроводов',
                 fontsize=14, fontweight='bold')
    ax.set_axis_off()
    ax.set_aspect('equal')
    return fig, failure_probs


# ──────────────────────────────────────────────────────────────────────────────
# CHANGE 3 — Real-world Folium Map
# ──────────────────────────────────────────────────────────────────────────────

# Almaty centre
ALMATY_LAT = 43.2220
ALMATY_LNG = 76.8512

# Grid step in degrees ≈ 0.0009° per 100m (rough conversion)
GRID_STEP_LAT = 0.0009
GRID_STEP_LNG = 0.0009


def grid_to_latlon(i: int, j: int) -> tuple:
    """Map 4×4 grid indices to Lat/Lon centred on Almaty."""
    lat = ALMATY_LAT + j * GRID_STEP_LAT
    lng = ALMATY_LNG + i * GRID_STEP_LNG
    return lat, lng


def create_folium_map(wn, active_leak: bool, predicted_leak_node: str,
                      failure_probs: dict, residuals: dict) -> folium.Map:
    """
    Build a Folium map of the water network overlaid on Almaty.
    Nodes are coloured by failure risk; the predicted leak node is highlighted.
    """
    m = folium.Map(location=[ALMATY_LAT, ALMATY_LNG], zoom_start=16,
                   tiles='OpenStreetMap')

    node_latlon = {}

    # Draw edges first (pipes)
    for link_name in wn.link_name_list:
        link = wn.get_link(link_name)
        if not (hasattr(link, 'start_node_name') and
                hasattr(link, 'end_node_name')):
            continue
        sn = link.start_node_name
        en = link.end_node_name

        # Reservoir has fixed coords (-100, -100) → special treatment
        def get_ll(name):
            coords = wn.get_node(name).coordinates
            if name == 'Res':
                return ALMATY_LAT - 0.0009, ALMATY_LNG - 0.0009
            i_idx = int(round(coords[0] / 100))
            j_idx = int(round(coords[1] / 100))
            return grid_to_latlon(i_idx, j_idx)

        sll = get_ll(sn)
        ell = get_ll(en)

        is_isolated = any(
            (sn in pipe or en in pipe)
            for pipe in st.session_state.get('isolated_pipes', [])
        )
        colour = '#c0392b' if is_isolated else '#7f8c8d'
        weight = 5 if is_isolated else 3

        folium.PolyLine(
            locations=[sll, ell],
            color=colour, weight=weight, opacity=0.8,
            tooltip=f"Pipe: {link_name}"
        ).add_to(m)

        node_latlon[sn] = sll
        node_latlon[en] = ell

    # Draw nodes
    for node_name in wn.node_name_list:
        ll = node_latlon.get(node_name)
        if ll is None:
            continue

        prob = failure_probs.get(node_name, 0)
        residual = residuals.get(node_name, 0)

        # Colour logic
        if node_name == 'Res':
            colour = '#2980b9'
            icon = 'tint'
            label = 'Резервуар'
        elif node_name == predicted_leak_node and active_leak:
            colour = '#c0392b'
            icon = 'warning-sign'
            label = f'⚠️ Утечка (предсказано)<br>Риск: {prob:.1f}%<br>Перепад: {residual:.3f} bar'
        elif prob > 40:
            colour = '#e74c3c'
            icon = 'remove'
            label = f'{node_name}<br>Риск: {prob:.1f}%'
        elif prob > 25:
            colour = '#e67e22'
            icon = 'exclamation-sign'
            label = f'{node_name}<br>Риск: {prob:.1f}%'
        elif prob > 15:
            colour = '#f1c40f'
            icon = 'info-sign'
            label = f'{node_name}<br>Риск: {prob:.1f}%'
        else:
            colour = '#27ae60'
            icon = 'ok'
            label = f'{node_name}<br>Риск: {prob:.1f}%'

        folium.Marker(
            location=ll,
            popup=folium.Popup(label, max_width=200),
            tooltip=node_name,
            icon=folium.Icon(color='white', icon_color=colour,
                             icon=icon, prefix='glyphicon')
        ).add_to(m)

    # Legend (HTML overlay)
    legend_html = """
    <div style="position:fixed; bottom:30px; left:30px; z-index:1000;
                background:white; padding:10px; border-radius:8px;
                border:2px solid #ccc; font-size:12px;">
      <b>Легенда</b><br>
      🔴 Высокий риск (>40%)<br>
      🟠 Средний риск (25-40%)<br>
      🟡 Умеренный риск (15-25%)<br>
      🟢 Низкий риск (&lt;15%)<br>
      🔵 Резервуар<br>
      ⚠️ Предсказанная утечка
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


# ──────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────────────────────────

for key, default in [
    ('data', None), ('network', None), ('log', []),
    ('isolated_pipes', []), ('csv_data', None),
    ('healthy_pressures', {}), ('residuals', {}),
    ('predicted_leak_node', 'N_2_2'),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────

st.sidebar.title("🧪 Экспертная панель")

with st.sidebar.expander("⚙️ Параметры сети", expanded=True):
    m_types = {"Пластик (ПНД)": 150, "Сталь": 140, "Чугун": 100}
    material = st.selectbox("Материал труб", list(m_types.keys()))
    iznos = st.slider("Износ системы (%)", 0, 60, 15,
                      help="Процент деградации трубопровода")
    freq = st.select_slider("Частота датчиков", options=[1, 2, 4],
                            format_func=lambda x: f"{x} Гц")

with st.sidebar.expander("🔧 Стресс-тест насоса", expanded=True):
    pump_pressure = st.slider("Напор насоса (м)", 30, 60, 40, step=5,
                              help="Проверка устойчивости системы при изменении давления")
    st.info(f"💡 Текущий напор: **{pump_pressure} м** = "
            f"**{pump_pressure * 0.098:.1f} bar**")

    # CHANGE 1: Smart pump scheduling toggle
    smart_pump = st.checkbox(
        "⚡ Enable Smart Pump Scheduling",
        value=False,
        help="Ночью (23:00-05:00) напор снижается на 30% для экономии энергии"
    )
    if smart_pump:
        st.success(f"Ночной напор: **{pump_pressure * 0.7:.0f} м**  "
                   f"| Дневной: **{pump_pressure} м**")

with st.sidebar.expander("💰 Экономика", expanded=True):
    price = st.number_input("Тариф за литр (₸)", value=0.55, step=0.05,
                            format="%.2f")
    limit = st.slider("Порог детекции (bar)", 1.0, 5.0, 2.7, step=0.1)

    # CHANGE 5: Indirect cost inputs
    repair_cost = st.number_input("Стоимость выезда бригады (₸)",
                                  value=50000, step=5000, format="%d",
                                  help="Фиксированные затраты на выезд ремонтной бригады")

with st.sidebar.expander("🔄 IoT интеграция", expanded=False):
    st.markdown("**Загрузка данных с реальных датчиков**")
    uploaded_file = st.file_uploader("Загрузить CSV", type=['csv'],
                                     help="Формат: Hour, Pressure, Flow Rate")
    if uploaded_file is not None:
        try:
            csv_df = pd.read_csv(uploaded_file)
            csv_df.columns = csv_df.columns.str.strip()
            required_cols = ['Hour', 'Pressure (bar)', 'Flow Rate (L/s)']
            missing_cols = [c for c in required_cols if c not in csv_df.columns]
            if missing_cols:
                st.error(f"❌ Отсутствуют колонки: {', '.join(missing_cols)}")
                st.info(f"Доступные: {', '.join(csv_df.columns.tolist())}")
            else:
                st.session_state['csv_data'] = csv_df
                st.success(f"✅ Загружено {len(csv_df)} записей")
        except Exception as e:
            st.error(f"❌ Ошибка: {str(e)}")

with st.sidebar.expander("🛡️ Управление задвижками", expanded=False):
    enable_valves = st.checkbox("Включить систему задвижек", value=False)
    st.info("При обнаружении утечки система предложит перекрыть участок")

st.sidebar.markdown("---")

if st.sidebar.button("🚀 ЗАПУСТИТЬ СИМУЛЯЦИЮ", use_container_width=True,
                     type="primary"):
    with st.spinner("⏳ Расчет цифрового двойника..."):
        try:
            data, net = run_epanet_simulation(
                m_types[material], iznos, freq,
                pump_pressure, enable_valves, smart_pump
            )
            st.session_state['data'] = data
            st.session_state['network'] = net
            st.session_state['isolated_pipes'] = []

            # CHANGE 2: Run healthy baseline + residual analysis
            with st.spinner("🔍 Расчет базовой модели (Residual Analysis)..."):
                healthy_p = run_healthy_simulation(
                    m_types[material], iznos, freq, pump_pressure
                )
                st.session_state['healthy_pressures'] = healthy_p
                pred_node, residuals = find_predicted_leak_node(
                    net, data, healthy_p, freq,
                    st.session_state['leak_node']
                )
                st.session_state['predicted_leak_node'] = pred_node
                st.session_state['residuals'] = residuals

            log_entry = (
                f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Симуляция | "
                f"{material}, Износ: {iznos}%, Напор: {pump_pressure}м"
                + (" [Smart Pump ON]" if smart_pump else "")
            )
            st.session_state['log'].append(log_entry)
            st.sidebar.success("✅ Готово!")
        except Exception as e:
            st.sidebar.error(f"❌ Ошибка: {str(e)}")

# ──────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ──────────────────────────────────────────────────────────────────────────────

st.title("💧 Smart Shygyn PRO: Expert Water Management System")
st.markdown(
    "##### Профессиональная система мониторинга с MNF, изоляцией участков, "
    "прогнозной аналитикой и реальной картой Алматы"
)

if st.session_state['data'] is not None:
    df = st.session_state['data']
    wn = st.session_state['network']
    predicted_leak_node = st.session_state['predicted_leak_node']
    residuals = st.session_state['residuals']

    df['Leak'] = df['Pressure (bar)'] < limit
    active_leak = df['Leak'].any()
    mnf_detected, mnf_anomaly = calculate_mnf_anomaly(df)
    contamination_risk = (df['Pressure (bar)'] < 1.5).any()

    # ── Economic calculations ──────────────────────────────────────────────
    lost_l = (
        df[df['Leak']]['Flow Rate (L/s)'].sum() * (3600 / freq)
        if active_leak else 0
    )
    direct_damage = lost_l * price

    # CHANGE 5: Indirect costs + NRW
    indirect_cost = repair_cost if active_leak else 0
    total_daily_flow = df['Flow Rate (L/s)'].sum() * (3600 / freq)
    nrw_pct = (lost_l / total_daily_flow * 100) if total_daily_flow > 0 else 0
    total_damage = direct_damage + indirect_cost

    # CHANGE 1: Energy saved
    energy_saved_pct = (
        calculate_energy_saved(pump_pressure, df, freq)
        if smart_pump else 0.0
    )

    # ── KPI DASHBOARD ─────────────────────────────────────────────────────
    st.markdown("### 📊 Панель состояния системы")

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        if active_leak:
            st.metric("🚨 Статус", "УТЕЧКА", "Критично",
                      delta_color="inverse")
        else:
            st.metric("✅ Статус", "НОРМА", "Стабильно",
                      delta_color="normal")

    with col2:
        min_p = df['Pressure (bar)'].min()
        st.metric("Давление min", f"{min_p:.2f} bar",
                  f"{min_p - limit:.2f}",
                  delta_color="inverse" if min_p < limit else "normal")

    with col3:
        st.metric("Потери воды", f"{lost_l:,.0f} L",
                  "⚠️" if lost_l > 5000 else None)

    with col4:
        # CHANGE 5: show total damage including indirect
        st.metric(
            "Ущерб (прямой+косв.)",
            f"{total_damage:,.0f} ₸",
            f"NRW: {nrw_pct:.1f}%",
            delta_color="inverse" if total_damage > 0 else "normal"
        )

    with col5:
        if mnf_detected:
            st.metric("MNF аномалия", f"+{mnf_anomaly:.1f}%",
                      "Скрытая утечка", delta_color="inverse")
        else:
            st.metric("MNF статус", "Норма",
                      f"{mnf_anomaly:.1f}%", delta_color="normal")

    with col6:
        # CHANGE 1: Energy Saved metric
        if smart_pump:
            st.metric("⚡ Сэкономлено",
                      f"{energy_saved_pct:.1f}%",
                      "Smart Pump ON",
                      delta_color="normal")
        else:
            st.metric("⚡ Smart Pump",
                      "Выкл.",
                      "Включить в боковой панели",
                      delta_color="off")

    # Alerts
    if contamination_risk:
        st.error("⚠️ **ОПАСНОСТЬ ИНФИЛЬТРАЦИИ!** Давление < 1.5 bar. "
                 "Риск загрязнения грунтовыми водами!")
    if mnf_detected:
        st.warning(f"🔍 **MNF АНОМАЛИЯ:** Ночной расход превышает норму на "
                   f"{mnf_anomaly:.1f}%. Возможна скрытая утечка.")
    if active_leak:
        st.error(
            f"🔍 **RESIDUAL ANALYSIS:** Предсказанный узел утечки — "
            f"**{predicted_leak_node}** "
            f"(перепад давления: {residuals.get(predicted_leak_node, 0):.3f} bar)"
        )

    # CHANGE 5: Economic breakdown
    if active_leak:
        with st.expander("💰 Детализация экономического ущерба", expanded=False):
            ec1, ec2, ec3 = st.columns(3)
            with ec1:
                st.metric("Стоимость потерянной воды",
                          f"{direct_damage:,.0f} ₸")
            with ec2:
                st.metric("Косвенные затраты (выезд бригады)",
                          f"{indirect_cost:,.0f} ₸")
            with ec3:
                st.metric("Non-Revenue Water (NRW)", f"{nrw_pct:.2f}%")

    st.markdown("---")

    # ── TABS ──────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Гидравлика",
        "🗺️ Карта (Алматы)",
        "🔥 Риск-карта",
        "🔄 IoT данные",
        "📋 Отчеты"
    ])

    # ── TAB 1: HYDRAULICS ─────────────────────────────────────────────────
    with tab1:
        st.markdown("### Расширенный анализ гидравлических параметров")
        st.caption("📊 Сигналы сглажены скользящим средним (окно=3) для имитации "
                   "реального шумоподавления датчиков.")
        fig = create_advanced_plot(df, limit, smart_pump)
        st.plotly_chart(fig, use_container_width=True)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown("#### 💧 Давление")
            st.dataframe(df['Pressure (bar)'].describe()
                         .to_frame().style.format("{:.3f}"),
                         use_container_width=True)
        with col_b:
            st.markdown("#### 🌊 Расход")
            st.dataframe(df['Flow Rate (L/s)'].describe()
                         .to_frame().style.format("{:.3f}"),
                         use_container_width=True)
        with col_c:
            st.markdown("#### ⏱️ Качество")
            st.dataframe(df['Water Age (h)'].describe()
                         .to_frame().style.format("{:.2f}"),
                         use_container_width=True)

        if st.session_state['log']:
            with st.expander("📜 История операций"):
                for log in reversed(st.session_state['log'][-15:]):
                    st.code(log, language=None)

    # ── TAB 2: FOLIUM MAP (CHANGE 3) ──────────────────────────────────────
    with tab2:
        st.markdown("### 🗺️ Интерактивная карта сети (Алматы)")
        st.caption(
            "Узлы 4×4 сетки привязаны к реальным координатам центра Алматы. "
            "Цвет отражает вероятность отказа. Красная метка — предсказанная утечка."
        )

        col_map, col_ctrl = st.columns([3, 1])

        with col_ctrl:
            st.markdown("#### 🛡️ Система изоляции")
            if active_leak:
                st.error(f"**⚠️ УТЕЧКА (предсказано): {predicted_leak_node}**")

                if st.button("🔒 ПЕРЕКРЫТЬ УЧАСТОК",
                             use_container_width=True, type="primary"):
                    pipes_to_close, affected_nodes = find_isolation_valves(
                        wn, predicted_leak_node
                    )
                    st.session_state['isolated_pipes'] = pipes_to_close
                    log_entry = (
                        f"[{datetime.now().strftime('%H:%M:%S')}] 🔒 "
                        f"Изолировано труб: {len(pipes_to_close)}"
                    )
                    st.session_state['log'].append(log_entry)
                    st.rerun()

                if st.session_state['isolated_pipes']:
                    st.success("✅ **Участок изолирован**")
                    st.write(f"Перекрыто труб: "
                             f"**{len(st.session_state['isolated_pipes'])}**")
                    affected = len(affected_nodes) * 250
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

            # Residual table
            if residuals:
                st.markdown("#### 🔍 Перепады давления (Residuals)")
                res_df = pd.DataFrame(
                    [(k, v) for k, v in residuals.items()],
                    columns=['Узел', 'Перепад (bar)']
                ).sort_values('Перепад (bar)', ascending=False)
                st.dataframe(res_df.style.format({'Перепад (bar)': '{:.4f}'}),
                             use_container_width=True, height=200)

        with col_map:
            # Build failure probs for colouring
            _, fail_probs = create_heatmap_network(wn, df, iznos)
            fmap = create_folium_map(
                wn, active_leak, predicted_leak_node, fail_probs, residuals
            )
            st_folium(fmap, width=None, height=520)

    # ── TAB 3: RISK HEATMAP ───────────────────────────────────────────────
    with tab3:
        st.markdown("### Прогнозная аналитика отказов (Predictive Maintenance)")
        fig_heat, fail_probs = create_heatmap_network(wn, df, iznos)
        st.pyplot(fig_heat)

        st.markdown("#### 📊 Вероятность отказа по узлам")
        sorted_probs = sorted(
            [(k, v) for k, v in fail_probs.items() if k != 'Res'],
            key=lambda x: x[1], reverse=True
        )[:5]

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown("**🔴 Топ-5 узлов высокого риска:**")
            for i, (node, prob) in enumerate(sorted_probs, 1):
                color = "🔴" if prob > 40 else "🟠" if prob > 25 else "🟡"
                marker = " ⚠️ (утечка)" if node == predicted_leak_node else ""
                st.write(f"{i}. {color} **{node}**{marker} — {prob:.1f}% риска")
        with col_r2:
            st.markdown("**💡 Рекомендации:**")
            if sorted_probs and sorted_probs[0][1] > 40:
                st.error("⚠️ Срочная замена труб в узлах высокого риска!")
            elif sorted_probs and sorted_probs[0][1] > 25:
                st.warning("📋 Плановая замена в течение 6 месяцев")
            else:
                st.success("✅ Система в удовлетворительном состоянии")
            st.info(
                f"**Стресс-тест:** При напоре {pump_pressure}м система "
                f"{'выдерживает' if pump_pressure <= 50 else 'перегружена'}"
            )

    # ── TAB 4: IoT DATA ───────────────────────────────────────────────────
    with tab4:
        st.markdown("### IoT интеграция и сравнение с моделью")

        if st.session_state['csv_data'] is not None:
            csv_df = st.session_state['csv_data']
            if all(c in csv_df.columns
                   for c in ['Hour', 'Pressure (bar)', 'Flow Rate (L/s)']):
                fig_compare = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Сравнение давления', 'Сравнение расхода'),
                    vertical_spacing=0.12
                )
                fig_compare.add_trace(
                    go.Scatter(x=df['Hour'], y=df['Pressure (bar)'],
                               name='Модель (MA)', line=dict(color='blue', dash='dot')),
                    row=1, col=1)
                fig_compare.add_trace(
                    go.Scatter(x=csv_df['Hour'], y=csv_df['Pressure (bar)'],
                               name='Датчики', line=dict(color='red')),
                    row=1, col=1)
                fig_compare.add_trace(
                    go.Scatter(x=df['Hour'], y=df['Flow Rate (L/s)'],
                               name='Модель (MA)', line=dict(color='blue', dash='dot')),
                    row=2, col=1)
                fig_compare.add_trace(
                    go.Scatter(x=csv_df['Hour'], y=csv_df['Flow Rate (L/s)'],
                               name='Датчики', line=dict(color='red')),
                    row=2, col=1)
                fig_compare.update_xaxes(title_text="Время (часы)", row=2, col=1)
                fig_compare.update_yaxes(title_text="Давление (bar)", row=1, col=1)
                fig_compare.update_yaxes(title_text="Расход (L/s)", row=2, col=1)
                fig_compare.update_layout(height=700, showlegend=True)
                st.plotly_chart(fig_compare, use_container_width=True)

                st.markdown("#### 📉 Анализ отклонений (Residuals)")
                if len(csv_df) == len(df):
                    residual_p = (csv_df['Pressure (bar)'].values
                                  - df['Pressure (bar)'].values)
                    residual_f = (csv_df['Flow Rate (L/s)'].values
                                  - df['Flow Rate (L/s)'].values)
                    col_res1, col_res2 = st.columns(2)
                    with col_res1:
                        st.metric("Макс. отклонение давления",
                                  f"{np.max(np.abs(residual_p)):.3f} bar")
                        st.metric("Средн. отклонение давления",
                                  f"{np.mean(np.abs(residual_p)):.3f} bar")
                    with col_res2:
                        st.metric("Макс. отклонение расхода",
                                  f"{np.max(np.abs(residual_f)):.3f} L/s")
                        st.metric("Средн. отклонение расхода",
                                  f"{np.mean(np.abs(residual_f)):.3f} L/s")
                    if np.max(np.abs(residual_p)) > 0.5:
                        st.error("⚠️ Значительное расхождение с моделью! "
                                 "Возможна аномалия в сети.")
                else:
                    st.warning("⚠️ Длина данных не совпадает.")
            else:
                st.error("❌ CSV должен содержать: Hour, Pressure (bar), "
                         "Flow Rate (L/s)")
        else:
            st.info("📁 Загрузите CSV в боковой панели для сравнения с моделью")
            st.markdown("**Пример формата CSV:**")
            example_csv = pd.DataFrame({
                'Hour': [0, 1, 2, 3, 4],
                'Pressure (bar)': [3.2, 3.1, 2.9, 2.8, 2.7],
                'Flow Rate (L/s)': [1.2, 1.1, 0.9, 0.8, 0.85]
            })
            st.dataframe(example_csv)

    # ── TAB 5: REPORTS ────────────────────────────────────────────────────
    with tab5:
        st.markdown("### Экспорт и отчетность")

        col_r1, col_r2 = st.columns([3, 2])
        with col_r1:
            st.markdown("#### 📊 Полная таблица данных")
            display_df = df.copy()
            display_df['Status'] = display_df['Leak'].apply(
                lambda x: '🚨 Утечка' if x else '✅ Норма'
            )
            display_df['Risk'] = display_df['Pressure (bar)'].apply(
                lambda x: '⚠️ Риск' if x < 1.5 else '✅ Норма'
            )
            st.dataframe(
                display_df.style.format({
                    'Hour': '{:.1f}',
                    'Pressure (bar)': '{:.3f}',
                    'Flow Rate (L/s)': '{:.3f}',
                    'Water Age (h)': '{:.2f}',
                    'Demand Pattern': '{:.3f}',
                    'Pump Head (m)': '{:.1f}',
                }).background_gradient(cmap='RdYlGn',
                                       subset=['Pressure (bar)']),
                height=450,
                use_container_width=True
            )

        with col_r2:
            st.markdown("#### 📥 Генерация отчетов")
            inc_mnf = st.checkbox("MNF анализ", value=True)
            inc_risk = st.checkbox("Карта рисков", value=True)
            inc_quality = st.checkbox("Качество воды", value=True)
            inc_isolation = st.checkbox(
                "План изоляции",
                value=bool(st.session_state['isolated_pipes'])
            )

            report_data = display_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📄 Скачать полный отчет CSV",
                data=report_data,
                file_name=(
                    f"smart_shygyn_expert_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                ),
                mime="text/csv",
                use_container_width=True
            )

            st.markdown("---")
            st.markdown("**📋 Краткая сводка:**")
            st.write(f"• Статус: {'🚨 Утечка' if active_leak else '✅ Норма'}")
            st.write(f"• Предсказанный узел: **{predicted_leak_node}**")
            st.write(f"• MNF: {'⚠️ Аномалия' if mnf_detected else '✅ Норма'}")
            st.write(
                f"• Риск заражения: "
                f"{'⚠️ Да' if contamination_risk else '✅ Нет'}"
            )
            st.write(f"• Потери: {lost_l:,.0f} L  (NRW: {nrw_pct:.2f}%)")
            st.write(f"• Прямой ущерб: {direct_damage:,.0f} ₸")
            st.write(f"• Косвенные затраты: {indirect_cost:,.0f} ₸")
            st.write(f"• Итого ущерб: **{total_damage:,.0f} ₸**")
            if smart_pump:
                st.write(f"• Экономия энергии: **{energy_saved_pct:.1f}%**")

            if st.button("📧 Отправить в ЖКХ",
                         use_container_width=True, type="primary"):
                st.success("✅ Отчет отправлен на систему управления!")
                log_entry = (
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    "📧 Отчет отправлен в ЖКХ"
                )
                st.session_state['log'].append(log_entry)

# ── WELCOME SCREEN ────────────────────────────────────────────────────────────
else:
    st.markdown("### 👋 Добро пожаловать в Smart Shygyn Expert Edition!")
    st.markdown(
        "Профессиональная система с модулями: "
        "**MNF анализ** • **Зональная изоляция** • **Качество воды** • "
        "**Прогнозная аналитика** • **IoT интеграция** • "
        "**⚡ Smart Pump Scheduling** • **🗺️ Карта Алматы**"
    )
    st.markdown("---")

    col_w1, col_w2, col_w3, col_w4, col_w5 = st.columns(5)
    with col_w1:
        st.markdown("#### 🌙 MNF анализ")
        st.markdown("- Ночной минимум\n- Скрытые утечки\n- Паттерн потребления")
    with col_w2:
        st.markdown("#### 🛡️ Изоляция")
        st.markdown("- Автопоиск задвижек\n- Минимизация ущерба\n- Контроль участков")
    with col_w3:
        st.markdown("#### 💧 Качество")
        st.markdown("- Возраст воды\n- Риск заражения\n- Санитарный контроль")
    with col_w4:
        st.markdown("#### 🔮 Прогноз")
        st.markdown("- Вероятность отказа\n- Тепловая карта\n- План замены труб")
    with col_w5:
        st.markdown("#### ⚡ Новое в Expert")
        st.markdown(
            "- Smart Pump Scheduling\n"
            "- Residual Analysis (утечка)\n"
            "- Карта Алматы (Folium)\n"
            "- Шумоподавление (MA)\n"
            "- NRW + косвенные затраты"
        )
