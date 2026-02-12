"""
Advanced visualization functions with performance optimization.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import MarkerCluster, Fullscreen, LocateControl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from typing import Dict, List
from config import CONFIG


@st.cache_data(ttl=CONFIG.CACHE_TTL)
def create_hydraulic_plot(
    df: pd.DataFrame,
    threshold_bar: float,
    smart_pump: bool,
    dark_mode: bool
) -> go.Figure:
    """
    Create interactive hydraulic diagnostics plot with range selectors.
    
    Features:
    - Range sliders for time navigation
    - Hover templates with detailed info
    - Responsive layout
    """
    bg = "#0F172A" if dark_mode else "#F8FAFC"
    fg = "#F1F5F9" if dark_mode else "#0F172A"
    grid_c = "#334155" if dark_mode else "#E2E8F0"
    
    rows = 4 if smart_pump else 3
    row_heights = [0.28, 0.28, 0.22, 0.22] if smart_pump else [0.35, 0.35, 0.30]
    
    titles = [
        "💧 Pressure Monitoring (bar)",
        "🌊 Flow Rate Analysis (L/s)",
        "⏱ Water Age Tracking (hours)"
    ]
    
    if smart_pump:
        titles.append("⚡ Smart Pump Schedule (m)")
    
    fig = make_subplots(
        rows=rows,
        cols=1,
        subplot_titles=titles,
        vertical_spacing=0.08,
        row_heights=row_heights,
        shared_xaxes=True
    )
    
    # ========== SUBPLOT 1: PRESSURE ==========
    fig.add_trace(
        go.Scatter(
            x=df["Hour"],
            y=df["Pressure (bar)"],
            name="Pressure",
            line=dict(color=CONFIG.ACCENT_COLOR, width=3),
            fill="tozeroy",
            fillcolor=f"rgba(59, 130, 246, 0.15)",
            hovertemplate=(
                "<b>Time:</b> %{x:.1f}h<br>"
                "<b>Pressure:</b> %{y:.2f} bar<br>"
                "<extra></extra>"
            )
        ),
        row=1, col=1
    )
    
    # Leak threshold
    fig.add_hline(
        y=threshold_bar,
        line_dash="dash",
        line_color=CONFIG.DANGER_COLOR,
        line_width=2.5,
        annotation_text=f"⚠ Threshold ({threshold_bar} bar)",
        annotation_position="right",
        row=1, col=1
    )
    
    # Critical zone
    fig.add_hrect(
        y0=0, y1=1.5,
        fillcolor="rgba(239, 68, 68, 0.1)",
        layer="below",
        line_width=0,
        annotation_text="Critical Zone",
        annotation_position="top left",
        row=1, col=1
    )
    
    # ========== SUBPLOT 2: FLOW RATE ==========
    fig.add_trace(
        go.Scatter(
            x=df["Hour"],
            y=df["Flow Rate (L/s)"],
            name="Observed Flow",
            line=dict(color=CONFIG.WARNING_COLOR, width=3),
            hovertemplate=(
                "<b>Time:</b> %{x:.1f}h<br>"
                "<b>Flow:</b> %{y:.2f} L/s<br>"
                "<extra></extra>"
            )
        ),
        row=2, col=1
    )
    
    # Expected flow
    expected_flow = df["Demand Pattern"] * df["Flow Rate (L/s)"].mean()
    fig.add_trace(
        go.Scatter(
            x=df["Hour"],
            y=expected_flow,
            name="Expected Flow",
            line=dict(color=CONFIG.SUCCESS_COLOR, width=2, dash="dot"),
            hovertemplate="<b>Expected:</b> %{y:.2f} L/s<extra></extra>"
        ),
        row=2, col=1
    )
    
    # MNF window
    fig.add_vrect(
        x0=2, x1=5,
        fillcolor="rgba(6, 182, 212, 0.1)",
        layer="below",
        line_width=0,
        annotation_text="MNF Window",
        annotation_position="top left",
        annotation_font_size=11,
        row=2, col=1
    )
    
    # ========== SUBPLOT 3: WATER AGE ==========
    fig.add_trace(
        go.Scatter(
            x=df["Hour"],
            y=df["Water Age (h)"],
            name="Water Age",
            line=dict(color="#A855F7", width=3),
            fill="tozeroy",
            fillcolor="rgba(168, 85, 247, 0.15)",
            hovertemplate=(
                "<b>Time:</b> %{x:.1f}h<br>"
                "<b>Age:</b> %{y:.1f} hours<br>"
                "<extra></extra>"
            )
        ),
        row=3, col=1
    )
    
    # ========== SUBPLOT 4: PUMP HEAD (if smart pump) ==========
    if smart_pump:
        fig.add_trace(
            go.Scatter(
                x=df["Hour"],
                y=df["Pump Head (m)"],
                name="Pump Head",
                line=dict(color=CONFIG.SUCCESS_COLOR, width=3),
                fill="tozeroy",
                fillcolor="rgba(16, 185, 129, 0.15)",
                hovertemplate=(
                    "<b>Time:</b> %{x:.1f}h<br>"
                    "<b>Head:</b> %{y:.0f} m<br>"
                    "<extra></extra>"
                )
            ),
            row=4, col=1
        )
        
        # Night mode windows
        for x0, x1 in [(0, 6), (23, 24)]:
            fig.add_vrect(
                x0=x0, x1=x1,
                fillcolor="rgba(16, 185, 129, 0.1)",
                layer="below",
                line_width=0,
                annotation_text="Night Mode" if x0 == 0 else "",
                annotation_position="top left",
                row=4, col=1
            )
    
    # ========== LAYOUT CONFIGURATION ==========
    # X-axis with range slider
    fig.update_xaxes(
        title_text="Hour of Day",
        gridcolor=grid_c,
        color=fg,
        showgrid=True,
        rangeslider=dict(visible=True, thickness=0.05),
        row=rows, col=1
    )
    
    # Other X-axes
    for r in range(1, rows):
        fig.update_xaxes(
            gridcolor=grid_c,
            color=fg,
            showgrid=True,
            row=r, col=1
        )
    
    # Y-axes
    y_titles = ["Pressure (bar)", "Flow (L/s)", "Age (h)"]
    if smart_pump:
        y_titles.append("Head (m)")
    
    for r, title in enumerate(y_titles, 1):
        fig.update_yaxes(
            title_text=title,
            gridcolor=grid_c,
            color=fg,
            showgrid=True,
            row=r, col=1
        )
    
    # Global layout
    fig.update_layout(
        height=1000 if smart_pump else 850,
        showlegend=True,
        hovermode="x unified",
        plot_bgcolor=bg,
        paper_bgcolor=bg,
        font=dict(color=fg, size=13, family="Inter, sans-serif"),
        margin=dict(l=70, r=50, t=80, b=80),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor=grid_c,
            borderwidth=1,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


@st.cache_data(ttl=CONFIG.CACHE_TTL)
def create_folium_map(
    results: Dict,
    isolated_pipes: List[str],
    dark_mode: bool,
    use_clustering: bool = False
) -> folium.Map:
    """
    Create advanced Folium map with plugins.
    
    Features:
    - MarkerCluster for large networks
    - Fullscreen control
    - Locate control
    - Custom legend
    """
    from backend import CityManager
    
    city_cfg = results["city_config"]
    wn = results["network"]
    predicted_leak = results["predicted_leak"]
    failure_probs = results["failure_probabilities"]
    residuals = results["residuals"]
    sensors = results["sensors"]
    
    # Initialize map
    tiles = CONFIG.MAP_TILE_OPTIONS["dark" if dark_mode else "light"]
    m = folium.Map(
        location=[city_cfg["lat"], city_cfg["lng"]],
        zoom_start=city_cfg["zoom"],
        tiles=tiles,
        prefer_canvas=True
    )
    
    # Add plugins
    Fullscreen(position="topright").add_to(m)
    LocateControl(auto_start=False).add_to(m)
    
    # Marker cluster (optional)
    if use_clustering:
        marker_cluster = MarkerCluster().add_to(m)
        marker_target = marker_cluster
    else:
        marker_target = m
    
    # Create city manager for coordinates
    city_manager = CityManager(city_cfg["name"])
    
    # Store coordinates
    node_coords = {}
    
    # Draw pipes
    for link_name in wn.link_name_list:
        link = wn.get_link(link_name)
        
        if not (hasattr(link, "start_node_name") and hasattr(link, "end_node_name")):
            continue
        
        start_node = link.start_node_name
        end_node = link.end_node_name
        
        # Get coordinates
        def get_coords(node_name):
            if node_name == "Res":
                return city_cfg["lat"] - 0.0009, city_cfg["lng"] - 0.0009
            
            node = wn.get_node(node_name)
            x, y = node.coordinates
            i, j = int(round(x / 100)), int(round(y / 100))
            return city_manager.grid_to_latlon(i, j)
        
        start_coords = get_coords(start_node)
        end_coords = get_coords(end_node)
        
        node_coords[start_node] = start_coords
        node_coords[end_node] = end_coords
        
        # Pipe styling
        is_isolated = link_name in isolated_pipes
        
        folium.PolyLine(
            [start_coords, end_coords],
            color="#DC2626" if is_isolated else "#475569",
            weight=7 if is_isolated else 4,
            opacity=0.95 if is_isolated else 0.65,
            tooltip=f"{'⛔ ISOLATED: ' if is_isolated else ''}{link_name}",
            popup=folium.Popup(
                f"<b>Pipe:</b> {link_name}<br>"
                f"<b>Status:</b> {'ISOLATED' if is_isolated else 'OPERATIONAL'}",
                max_width=200
            )
        ).add_to(m)
    
    # Draw nodes
    leak_detected = results["dataframe"]["Pressure (bar)"].min() < 2.7
    
    for node_name in wn.node_name_list:
        coords = node_coords.get(node_name)
        if coords is None:
            continue
        
        prob = failure_probs.get(node_name, 0)
        residual = residuals.get(node_name, 0)
        is_sensor = node_name in sensors
        
        # Determine marker style
        if node_name == "Res":
            color, icon = "blue", "tint"
            popup_html = "<b>🔵 Reservoir</b><br>Water Source"
        elif node_name == predicted_leak and leak_detected:
            color, icon = "red", "warning-sign"
            popup_html = (
                f"<b style='color:#EF4444;'>⚠️ PREDICTED LEAK</b><br>"
                f"<b>Node:</b> {node_name}<br>"
                f"<b>Risk:</b> {prob:.1f}%<br>"
                f"<b>Residual:</b> {residual:.3f} bar<br>"
                f"<b>Confidence:</b> {results['confidence']:.0f}%"
            )
        elif prob > 40:
            color, icon = "red", "remove"
            popup_html = (
                f"<b style='color:#EF4444;'>🔴 CRITICAL RISK</b><br>"
                f"<b>Node:</b> {node_name}<br>"
                f"<b>Failure Risk:</b> {prob:.1f}%"
            )
        elif prob > 25:
            color, icon = "orange", "exclamation-sign"
            popup_html = f"<b>🟠 ELEVATED</b><br>{node_name}<br>Risk: {prob:.1f}%"
        elif prob > 15:
            color, icon = "beige", "info-sign"
            popup_html = f"<b>🟡 MODERATE</b><br>{node_name}<br>Risk: {prob:.1f}%"
        else:
            color, icon = "green", "ok"
            popup_html = f"<b>🟢 NORMAL</b><br>{node_name}<br>Risk: {prob:.1f}%"
        
        # Sensor ring
        if is_sensor:
            folium.CircleMarker(
                coords,
                radius=16,
                color=CONFIG.WARNING_COLOR,
                weight=3,
                fill=False,
                tooltip=f"📡 Sensor: {node_name}",
                popup=f"<b>📡 Active Sensor</b><br>Node: {node_name}"
            ).add_to(m)
        
        # Main marker
        folium.Marker(
            coords,
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=node_name,
            icon=folium.Icon(color=color, icon=icon, prefix="glyphicon")
        ).add_to(marker_target)
    
    # Enhanced legend
    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 40px;
        left: 40px;
        width: 300px;
        z-index: 9999;
        background: {'rgba(15, 23, 42, 0.95)' if dark_mode else 'rgba(255, 255, 255, 0.95)'};
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 16px;
        border: 1px solid {'#334155' if dark_mode else '#E2E8F0'};
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        font-family: 'Inter', sans-serif;
        color: {'#F1F5F9' if dark_mode else '#0F172A'};
    ">
        <div style="font-size: 16px; font-weight: 700; color: #06B6D4; margin-bottom: 16px;">
            🗺️ Network Legend
        </div>
        <hr style="margin: 12px 0; border-color: {'#334155' if dark_mode else '#E2E8F0'}; opacity: 0.5;">
        <div style="margin: 8px 0; display: flex; align-items: center;">
            <span style="margin-right: 10px;">🔴</span>
            <span style="font-size: 13px;"><b>Critical Risk</b> (&gt;40%)</span>
        </div>
        <div style="margin: 8px 0; display: flex; align-items: center;">
            <span style="margin-right: 10px;">🟠</span>
            <span style="font-size: 13px;"><b>Elevated Risk</b> (25-40%)</span>
        </div>
        <div style="margin: 8px 0; display: flex; align-items: center;">
            <span style="margin-right: 10px;">🟡</span>
            <span style="font-size: 13px;"><b>Moderate Risk</b> (15-25%)</span>
        </div>
        <div style="margin: 8px 0; display: flex; align-items: center;">
            <span style="margin-right: 10px;">🟢</span>
            <span style="font-size: 13px;"><b>Normal</b> (&lt;15%)</span>
        </div>
        <div style="margin: 8px 0; display: flex; align-items: center;">
            <span style="margin-right: 10px;">⚠️</span>
            <span style="font-size: 13px;"><b>Predicted Leak</b></span>
        </div>
        <div style="margin: 8px 0; display: flex; align-items: center;">
            <span style="margin-right: 10px;">🔵</span>
            <span style="font-size: 13px;"><b>Reservoir</b></span>
        </div>
        <hr style="margin: 12px 0; border-color: {'#334155' if dark_mode else '#E2E8F0'}; opacity: 0.5;">
        <div style="margin: 8px 0; display: flex; align-items: center;">
            <span style="margin-right: 10px; color: #F59E0B; font-size: 18px;">⭕</span>
            <span style="font-size: 13px;"><b>Sensor Node</b> ({len(sensors)}/16)</span>
        </div>
        <div style="margin: 8px 0; display: flex; align-items: center;">
            <span style="margin-right: 10px; color: #DC2626; font-size: 18px;">━━</span>
            <span style="font-size: 13px;"><b>Isolated Pipe</b></span>
        </div>
    </div>
    """
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m


@st.cache_data(ttl=CONFIG.CACHE_TTL)
def create_failure_heatmap(results: Dict, dark_mode: bool) -> plt.Figure:
    """Create matplotlib heatmap with enhanced visuals."""
    wn = results["network"]
    failure_probs = results["failure_probabilities"]
    sensors = results["sensors"]
    predicted_leak = results["predicted_leak"]
    
    fig, ax = plt.subplots(
        figsize=(13, 11),
        facecolor="#0F172A" if dark_mode else "#F8FAFC"
    )
    ax.set_facecolor("#0F172A" if dark_mode else "#F8FAFC")
    txt_color = "#F1F5F9" if dark_mode else "#0F172A"
    
    pos = {node: wn.get_node(node).coordinates for node in wn.node_name_list}
    
    # Draw edges
    graph = wn.get_graph()
    nx.draw_networkx_edges(
        graph, pos, ax=ax,
        edge_color="#475569",
        width=4,
        alpha=0.7
    )
    
    # Draw nodes
    for node in wn.node_name_list:
        x, y = pos[node]
        prob = failure_probs.get(node, 0)
        
        # Color based on probability
        if node == "Res":
            color = CONFIG.ACCENT_COLOR
        elif prob > 40:
            color = CONFIG.DANGER_COLOR
        elif prob > 25:
            color = CONFIG.WARNING_COLOR
        elif prob > 15:
            color = "#EAB308"
        else:
            color = CONFIG.SUCCESS_COLOR
        
        # Main circle
        circle = plt.Circle(
            (x, y), radius=22,
            color=color, ec=txt_color,
            linewidth=3, zorder=3
        )
        ax.add_patch(circle)
        
        # Sensor ring
        if node in sensors:
            ring = plt.Circle(
                (x, y), radius=30,
                color=CONFIG.WARNING_COLOR,
                fill=False, linewidth=3,
                linestyle="--", zorder=4
            )
            ax.add_patch(ring)
        
        # Leak prediction ring
        if node == predicted_leak:
            alert_ring = plt.Circle(
                (x, y), radius=38,
                color=CONFIG.DANGER_COLOR,
                fill=False, linewidth=3.5,
                linestyle="-", zorder=5
            )
            ax.add_patch(alert_ring)
        
        # Label
        ax.text(
            x, y, node,
            fontsize=9, fontweight="bold",
            ha="center", va="center",
            color=txt_color, zorder=6
        )
    
    # Legend
    legend_elements = [
        mpatches.Patch(color=CONFIG.DANGER_COLOR, label="Critical (>40%)"),
        mpatches.Patch(color=CONFIG.WARNING_COLOR, label="Elevated (25-40%)"),
        mpatches.Patch(color="#EAB308", label="Moderate (15-25%)"),
        mpatches.Patch(color=CONFIG.SUCCESS_COLOR, label="Normal (<15%)"),
        mpatches.Patch(color=CONFIG.ACCENT_COLOR, label="Reservoir"),
        mpatches.Patch(color="none", label="─────────"),
        mpatches.Patch(color=CONFIG.WARNING_COLOR, label="📡 Sensor"),
        mpatches.Patch(color=CONFIG.DANGER_COLOR, label="⚠️ Predicted Leak"),
    ]
    
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        fontsize=11,
        facecolor="#1E293B" if dark_mode else "white",
        edgecolor="#334155" if dark_mode else "#E2E8F0",
        labelcolor=txt_color,
        framealpha=0.95
    )
    
    # Title
    city_name = results["city_config"]["name"]
    material = results["material"]
    age = results["pipe_age"]
    roughness = results["roughness"]
    
    ax.set_title(
        f"Pipe Failure Probability Heatmap — {city_name}\n"
        f"Material: {material} | Age: {age:.0f} years | H-W C: {roughness:.0f}",
        fontsize=15,
        fontweight="bold",
        color=txt_color,
        pad=25
    )
    
    ax.set_aspect("equal")
    ax.set_axis_off()
    plt.tight_layout()
    
    return fig


@st.cache_data(ttl=CONFIG.CACHE_TTL)
def create_payback_chart(economics: Dict, dark_mode: bool) -> go.Figure:
    """Create enhanced payback timeline chart."""
    bg = "#0F172A" if dark_mode else "#F8FAFC"
    fg = "#F1F5F9" if dark_mode else "#0F172A"
    grid_c = "#334155" if dark_mode else "#E2E8F0"
    
    payback_months = economics["payback_months"]
    max_months = min(int(payback_months * 2.5), 72)
    months = np.arange(0, max_months + 1)
    
    monthly_savings = economics["monthly_total_savings_kzt"]
    cumulative_savings = months * monthly_savings
    capex_line = np.full_like(months, economics["capex_kzt"], dtype=float)
    
    fig = go.Figure()
    
    # Cumulative savings area
    fig.add_trace(
        go.Scatter(
            x=months,
            y=cumulative_savings,
            name="Cumulative Savings",
            line=dict(color=CONFIG.SUCCESS_COLOR, width=4),
            fill="tozeroy",
            fillcolor="rgba(16, 185, 129, 0.2)",
            hovertemplate="<b>Month %{x}</b><br>Savings: ₸%{y:,.0f}<extra></extra>"
        )
    )
    
    # CAPEX line
    fig.add_trace(
        go.Scatter(
            x=months,
            y=capex_line,
            name="Initial Investment",
            line=dict(color=CONFIG.WARNING_COLOR, width=3, dash="dash"),
            hovertemplate="<b>CAPEX:</b> ₸%{y:,.0f}<extra></extra>"
        )
    )
    
    # Break-even point
    if payback_months < max_months:
        fig.add_vline(
            x=payback_months,
            line_dash="dot",
            line_color=CONFIG.ACCENT_COLOR,
            line_width=3,
            annotation_text=f"Break-Even: {payback_months:.1f} months",
            annotation_position="top",
            annotation_font_size=14,
            annotation_font_color=CONFIG.ACCENT_COLOR,
            annotation_bgcolor=bg
        )
    
    # Layout
    fig.update_layout(
        title="Investment Payback Timeline",
        xaxis_title="Months",
        yaxis_title="Tenge (KZT)",
        height=400,
        hovermode="x unified",
        plot_bgcolor=bg,
        paper_bgcolor=bg,
        font=dict(color=fg, size=13, family="Inter"),
        xaxis=dict(gridcolor=grid_c, color=fg),
        yaxis=dict(gridcolor=grid_c, color=fg),
        margin=dict(l=70, r=50, t=60, b=60),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor=grid_c,
            borderwidth=1
        )
    )
    
    return fig


@st.cache_data(ttl=CONFIG.CACHE_TTL)
def create_nrw_pie(economics: Dict, dark_mode: bool) -> go.Figure:
    """Create NRW distribution pie chart."""
    bg = "#0F172A" if dark_mode else "#F8FAFC"
    fg = "#F1F5F9" if dark_mode else "#0F172A"
    
    nrw_pct = economics["nrw_percentage"]
    revenue_pct = 100 - nrw_pct
    
    fig = go.Figure(
        go.Pie(
            labels=["Revenue Water", "Non-Revenue Water"],
            values=[max(0, revenue_pct), nrw_pct],
            hole=0.6,
            marker=dict(
                colors=[CONFIG.SUCCESS_COLOR, CONFIG.DANGER_COLOR],
                line=dict(color=bg, width=3)
            ),
            textinfo="label+percent",
            textfont=dict(size=14, color=fg, family="Inter"),
            hovertemplate="<b>%{label}</b><br>%{value:.1f}%<extra></extra>"
        )
    )
    
    # Center annotation
    fig.add_annotation(
        text=f"<b>NRW</b><br>{nrw_pct:.1f}%",
        x=0.5, y=0.5,
        font=dict(size=22, color=fg, family="Inter"),
        showarrow=False
    )
    
    fig.update_layout(
        title="Water Accountability",
        height=400,
        paper_bgcolor=bg,
        font=dict(color=fg, size=13, family="Inter"),
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig
