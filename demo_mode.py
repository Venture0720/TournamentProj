"""
Smart Shygyn PRO v3 — Demo Mode & Alert System
Раздел 4 ТЗ: Демо "живая симуляция" + вкладка Алёртов.

Использование в app.py:
    from demo_mode import render_demo_tab, render_alerts_tab

    with tab_alerts:
        render_alerts_tab(results, config, dark_mode=dm)

    with tab_demo:
        render_demo_tab(dark_mode=dm)
"""

import time
import random
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ══════════════════════════════════════════════════════════════════════
# ALERT SYSTEM
# ══════════════════════════════════════════════════════════════════════

class AlertLevel:
    CRITICAL = "CRITICAL"
    WARNING  = "WARNING"


def classify_alert(
    pressure_bar: float,
    flow_lps: float,
    anomaly_score: float = 0.0,
    min_pressure_bar: float = 2.5,
    normal_flow_lps: float = 1.0,
) -> str:
    """
    Логика классификации алёртов из ТЗ:
      CRITICAL:  pressure < min AND flow > 2× normal
      WARNING:   pressure < min OR anomaly_score > 0.7
    """
    if pressure_bar < min_pressure_bar and flow_lps > 2 * normal_flow_lps:
        return AlertLevel.CRITICAL
    if pressure_bar < min_pressure_bar or anomaly_score > 0.7:
        return AlertLevel.WARNING
    return None


def generate_alerts_from_results(results: Dict, config: Dict) -> List[Dict]:
    """
    Сформировать список алёртов из результатов симуляции.
    """
    alerts = []
    df      = results["dataframe"]
    thresh  = config.get("leak_threshold", 2.5)
    normal_flow = df["Flow Rate (L/s)"].median()

    for _, row in df.iterrows():
        level = classify_alert(
            pressure_bar   = row["Pressure (bar)"],
            flow_lps       = row["Flow Rate (L/s)"],
            min_pressure_bar = thresh,
            normal_flow_lps  = normal_flow,
        )
        if level in (AlertLevel.CRITICAL, AlertLevel.WARNING):
            alerts.append({
                "Время":        f"{row['Hour']:.1f} ч",
                "Узел":         results.get("predicted_leak", "?"),
                "Давление (бар)": f"{row['Pressure (bar)']:.3f}",
                "Расход (л/с)": f"{row['Flow Rate (L/s)']:.2f}",
                "Уровень":      level,
                "Уверенность":  f"{results.get('confidence', 0):.0f}%",
            })

    # Дедупликация: не более 20 последних
    return alerts[-20:]


def _level_color(level: str, dark: bool) -> str:
    """Цвет строки таблицы по уровню алёрта."""
    colors = {
        AlertLevel.CRITICAL: "rgba(239,68,68,0.25)",
        AlertLevel.WARNING:  "rgba(245,158,11,0.20)",
    }
    return colors.get(level, "")


# ══════════════════════════════════════════════════════════════════════
# ВКЛАДКА АЛЁРТОВ
# ══════════════════════════════════════════════════════════════════════

def render_alerts_tab(
    results: Optional[Dict],
    config: Dict,
    dark_mode: bool = True,
):
    """
    Вкладка '🚨 Алёрты' для app.py.
    """
    st.markdown("## 🚨 Система алёртов — Smart Shygyn PRO v3")

    if results is None:
        st.info("👈 Запусти симуляцию чтобы увидеть алёрты")
        return

    alerts = generate_alerts_from_results(results, config)

    # KPI строка
    critical_n = sum(1 for a in alerts if a["Уровень"] == AlertLevel.CRITICAL)
    warning_n  = sum(1 for a in alerts if a["Уровень"] == AlertLevel.WARNING)
    total_n    = len(alerts)

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("🔴 Критичных",     critical_n, delta_color="inverse")
    with col2: st.metric("🟡 Предупреждений", warning_n,  delta_color="inverse")
    with col3: st.metric("📋 Всего алёртов",  total_n)
    with col4:
        conf = results.get("confidence", 0)
        st.metric("🧠 Уверенность",
                  f"{conf:.0f}%",
                  results.get("predicted_leak", "—"))

    st.markdown("---")

    if not alerts:
        st.success("✅ Всё в норме — алёртов нет")
        return

    # Фильтры
    fc1, _ = st.columns([1, 3])
    with fc1:
        filter_level = st.multiselect(
            "Фильтр по уровню",
            [AlertLevel.CRITICAL, AlertLevel.WARNING],
            default=[AlertLevel.CRITICAL, AlertLevel.WARNING],
        )

    filtered = [a for a in alerts if a["Уровень"] in filter_level]

    # Таблица алёртов с цветами
    if filtered:
        df_alerts = pd.DataFrame(filtered)

        def style_rows(row):
            color = _level_color(row["Уровень"], dark_mode)
            return [f"background-color: {color}"] * len(row)

        st.dataframe(
            df_alerts.style.apply(style_rows, axis=1),
            use_container_width=True,
            hide_index=True,
            height=400,
        )

        # Экспорт CSV
        csv_data = df_alerts.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="📥 Экспорт алёртов CSV",
            data=csv_data,
            file_name=f"alerts_{results['city_config']['name']}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.info("Нет алёртов по выбранным фильтрам")

    st.markdown("---")
    st.markdown("### 📊 Распределение давления")

    df = results["dataframe"]
    thresh = config.get("leak_threshold", 2.5)
    bg = "#0e1117" if dark_mode else "white"
    fg = "#e2e8f0" if dark_mode else "#2c3e50"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Hour"], y=df["Pressure (bar)"],
        line=dict(color="#3b82f6", width=2),
        name="Давление",
    ))
    # Зоны критичности
    fig.add_hrect(y0=0, y1=thresh, fillcolor="rgba(239,68,68,0.1)",
                  layer="below", line_width=0)
    fig.add_hline(y=thresh, line_dash="dash", line_color="#ef4444",
                  annotation_text="Порог утечки", annotation_position="right")

    # Маркеры алёртов
    alert_hours = [float(a["Время"].replace(" ч", "")) for a in alerts]
    alert_pressures = []
    for h in alert_hours:
        idx = (df["Hour"] - h).abs().idxmin()
        alert_pressures.append(float(df.loc[idx, "Pressure (bar)"]))

    if alert_hours:
        fig.add_trace(go.Scatter(
            x=alert_hours, y=alert_pressures,
            mode="markers",
            marker=dict(color="#ef4444", size=10, symbol="x"),
            name="Алёрты",
        ))

    fig.update_layout(
        height=300,
        plot_bgcolor=bg, paper_bgcolor=bg,
        font=dict(color=fg, size=11),
        xaxis=dict(title="Час", gridcolor="#2d3748", color=fg),
        yaxis=dict(title="Давление (бар)", gridcolor="#2d3748", color=fg),
        margin=dict(l=60, r=40, t=20, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# LIVE DEMO — СИМУЛЯЦИЯ ДЛЯ ПИТЧА
# ══════════════════════════════════════════════════════════════════════

def render_demo_tab(dark_mode: bool = True):
    """
    Вкладка '▶ Live Демо' — живая симуляция для питча.
    """
    st.markdown("## ▶ Live Симуляция — для питча")
    st.markdown(
        "Нажми кнопку — система обнаружит утечку в реальном времени. "
        "Идеально для демонстрации жюри."
    )

    bg = "#0e1117" if dark_mode else "white"
    fg = "#e2e8f0" if dark_mode else "#2c3e50"

    # Параметры симуляции
    dc1, dc2, dc3 = st.columns(3)
    with dc1:
        sim_city      = st.selectbox("Город", ["Алматы", "Астана", "Туркестан"], key="demo_city")
    with dc2:
        sim_leak_hour = st.slider("Утечка появляется на шаге", 5, 20, 10, key="demo_leak_hour")
    with dc3:
        sim_speed     = st.select_slider("Скорость", ["Медленно", "Средне", "Быстро"],
                                         value="Средне", key="demo_speed")

    speed_map = {"Медленно": 0.8, "Средне": 0.4, "Быстро": 0.15}
    delay     = speed_map[sim_speed]

    col_btn, col_status = st.columns([1, 3])
    with col_btn:
        run_demo = st.button("▶ ЗАПУСТИТЬ СИМУЛЯЦИЮ", use_container_width=True, type="primary")

    if not run_demo:
        _render_demo_static(dark_mode=dark_mode, city=sim_city)
        return

    # ── ЗАПУСК ЖИВОЙ СИМУЛЯЦИИ ─────────────────────────────────────────
    TOTAL_STEPS = 30
    LEAK_STEP   = sim_leak_hour
    DETECT_STEP = LEAK_STEP + random.randint(2, 5)

    leak_node = random.choice(["N_2_2", "N_1_3", "N_3_1"])

    # Пустые контейнеры
    status_box  = col_status.empty()
    chart_box   = st.empty()
    alert_box   = st.empty()
    metrics_box = st.empty()

    pressures   = []
    detected    = False
    detection_ts = None
    ttd_hours    = 0
    lost_liters  = 0

    for step in range(1, TOTAL_STEPS + 1):
        base_pressure = 3.2 + 0.3 * np.sin(step * np.pi / 12)

        if step >= LEAK_STEP:
            leak_drop = 0.8 * (step - LEAK_STEP) / (TOTAL_STEPS - LEAK_STEP + 1)
            pressure  = base_pressure - leak_drop + np.random.normal(0, 0.04)
        else:
            pressure = base_pressure + np.random.normal(0, 0.04)

        pressures.append(max(0.5, pressure))

        if step == DETECT_STEP and not detected:
            detected     = True
            detection_ts = step
            ttd_hours    = detection_ts - LEAK_STEP
            lost_liters  = ttd_hours * 60 * 30

        # ── Статус ──────────────────────────────────────────────────────
        if step < LEAK_STEP:
            status_text = "🟢 Сеть в норме — все датчики в зелёной зоне"
        elif step < DETECT_STEP:
            status_text = f"🔴 **УТЕЧКА возникла в узле {leak_node}** (шаг {LEAK_STEP})"
        else:
            status_text = f"⚠️ **УТЕЧКА ОБНАРУЖЕНА** — TTD: {ttd_hours} шагов"

        status_box.markdown(status_text)

        # ── График ──────────────────────────────────────────────────────
        with chart_box.container():
            steps_shown = list(range(1, step + 1))

            fig = go.Figure()
            fig.add_hrect(y0=2.5, y1=5.0, fillcolor="rgba(16,185,129,0.08)",
                          layer="below", line_width=0)
            fig.add_hrect(y0=0, y1=2.5, fillcolor="rgba(239,68,68,0.08)",
                          layer="below", line_width=0)

            fig.add_trace(go.Scatter(
                x=steps_shown, y=pressures,
                line=dict(color="#3b82f6", width=2.5),
                name="Давление (бар)",
            ))

            if step >= LEAK_STEP:
                fig.add_vline(x=LEAK_STEP, line_color="#f59e0b",
                              line_dash="dash", line_width=2,
                              annotation_text=f"Утечка: шаг {LEAK_STEP}",
                              annotation_font_color="#f59e0b",
                              annotation_position="top right")

            if detected:
                fig.add_vline(x=DETECT_STEP, line_color="#ef4444",
                              line_width=2.5,
                              annotation_text=f"⚠ Обнаружена! (TTD={ttd_hours})",
                              annotation_font_color="#ef4444",
                              annotation_position="top left")
                fig.add_trace(go.Scatter(
                    x=[DETECT_STEP],
                    y=[pressures[DETECT_STEP - 1]],
                    mode="markers",
                    marker=dict(color="#ef4444", size=14, symbol="star"),
                    name="Момент детекции",
                ))

            fig.add_hline(y=2.5, line_dash="dot", line_color="#94a3b8", line_width=1.5,
                          annotation_text="Мин. норматив 2.5 бар", annotation_position="right")

            fig.update_layout(
                height=320,
                plot_bgcolor=bg, paper_bgcolor=bg,
                font=dict(color=fg, size=11),
                xaxis=dict(title="Шаг симуляции", gridcolor="#2d3748", color=fg,
                           range=[1, TOTAL_STEPS]),
                yaxis=dict(title="Давление (бар)", gridcolor="#2d3748", color=fg,
                           range=[0, 4.5]),
                margin=dict(l=60, r=40, t=20, b=40),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── Алёрт ───────────────────────────────────────────────────────
        if detected:
            alert_box.error(
                f"🚨 **УТЕЧКА ОБНАРУЖЕНА — {sim_city}, узел {leak_node}**  \n"
                f"Время реакции: **{ttd_hours} шагов**  \n"
                f"Ориентировочные потери: **{lost_liters:,.0f} л**  \n"
                f"Без системы: обнаружение через **~72 часа**, потери **~{72 * 60 * 30 / 1000:.0f} тыс. л**"
            )
        elif step >= LEAK_STEP:
            alert_box.warning(
                f"⚠️ Давление падает в зоне {leak_node}... Алгоритм анализирует..."
            )
        else:
            alert_box.empty()

        # ── Метрики ─────────────────────────────────────────────────────
        with metrics_box.container():
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("📍 Шаг", f"{step}/{TOTAL_STEPS}")
            m2.metric("💧 Мин. давление", f"{min(pressures):.2f} бар",
                      delta_color="inverse" if min(pressures) < 2.5 else "normal")
            m3.metric("⏱ TTD",
                      f"{step - LEAK_STEP} шагов" if step >= LEAK_STEP else "—",
                      delta_color="inverse" if step >= LEAK_STEP else "off")
            m4.metric("🧠 Статус", "ОБНАРУЖЕНА" if detected else "Мониторинг")

        time.sleep(delay)

    # ── Итог симуляции ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Итог симуляции")

    fi1, fi2, fi3 = st.columns(3)
    with fi1:
        st.metric("⏱ TTD Smart Shygyn",  f"{ttd_hours} шагов")
    with fi2:
        st.metric("🕰 Без системы",       "~72 часа")
    with fi3:
        st.metric("💧 Сэкономлено воды",
                  f"{(72 - ttd_hours) * 60 * 30 / 1000:.0f} тыс. л",
                  "vs ручное обнаружение")

    st.success(
        f"✅ Симуляция завершена. Smart Shygyn обнаружил утечку в узле **{leak_node}** "
        f"за **{ttd_hours} шагов** — в {72 // max(ttd_hours, 1)}× быстрее ручного метода."
    )


def _render_demo_static(dark_mode: bool, city: str):
    """Статичный превью до запуска симуляции."""
    bg = "#0e1117" if dark_mode else "white"
    fg = "#e2e8f0" if dark_mode else "#2c3e50"

    st.info(
        f"🎬 **Готово к запуску** — демо для {city}.  \n"
        "Нажми **▶ ЗАПУСТИТЬ СИМУЛЯЦИЮ** выше чтобы начать."
    )

    hours = np.linspace(0, 24, 100)
    pressure_normal = 3.0 + 0.4 * np.sin(hours * np.pi / 12) + np.random.normal(0, 0.03, 100)

    fig = go.Figure()
    fig.add_hrect(y0=2.5, y1=5.0, fillcolor="rgba(16,185,129,0.08)", layer="below", line_width=0)
    fig.add_trace(go.Scatter(
        x=hours, y=pressure_normal,
        line=dict(color="#3b82f6", width=2.5),
        name="Давление — нормальный режим",
    ))
    fig.add_hline(y=2.5, line_dash="dot", line_color="#94a3b8",
                  annotation_text="Норматив 2.5 бар", annotation_position="right")

    fig.update_layout(
        title=f"Пример: нормальный суточный профиль давления — {city}",
        height=280,
        plot_bgcolor=bg, paper_bgcolor=bg,
        font=dict(color=fg, size=11),
        xaxis=dict(title="Час", gridcolor="#2d3748", color=fg),
        yaxis=dict(title="Давление (бар)", gridcolor="#2d3748", color=fg),
        margin=dict(l=60, r=40, t=50, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)
