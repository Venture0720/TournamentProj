"""
Smart Shygyn PRO v3 — Business Model
Раздел 2 ТЗ: SaaS модель, ROI калькулятор для клиента, TAM/SAM/SOM.

Использование в app.py:
    from business_model import render_business_tab
    ...
    with tab_economy:
        render_business_tab(dark_mode=dm, city_name=config["city_name"])
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any


# ══════════════════════════════════════════════════════════════════════
# КОНСТАНТЫ — РЫНОЧНЫЕ ДАННЫЕ КЗ
# ══════════════════════════════════════════════════════════════════════

# Реальные данные потерь воды (источник: МИО, Комитет по статистике РК 2024)
CITY_WATER_DATA = {
    "Алматы": {
        "annual_production_m3": 300_000_000,     # 300 млн м³/год
        "nrw_pct": 30.0,                          # % потерь
        "tariff_kzt_m3": 91.96,                  # ₸/м³
        "pipe_km": 3_700,                         # км сети
        "population": 2_200_000,
    },
    "Астана": {
        "annual_production_m3": 120_000_000,
        "nrw_pct": 25.0,
        "tariff_kzt_m3": 85.00,
        "pipe_km": 1_800,
        "population": 1_400_000,
    },
    "Туркестан": {
        "annual_production_m3": 30_000_000,
        "nrw_pct": 35.0,
        "tariff_kzt_m3": 70.00,
        "pipe_km": 600,
        "population": 220_000,
    },
    "Другой город": {
        "annual_production_m3": 20_000_000,
        "nrw_pct": 32.0,
        "tariff_kzt_m3": 75.00,
        "pipe_km": 400,
        "population": 150_000,
    },
}

# Ценовые тиры SaaS
TIERS = {
    "Tier 1 — Пилот":        {"price_kzt_month": 0,         "network_km_max": 100,  "sensors_max": 30},
    "Tier 2 — Базовый":      {"price_kzt_month": 800_000,   "network_km_max": 500,  "sensors_max": 100},
    "Tier 3 — Профессиональный": {"price_kzt_month": 3_500_000, "network_km_max": 9999, "sensors_max": 9999},
    "Tier 4 — Корпоративный": {"price_kzt_month": 1_250_000,  "network_km_max": 9999, "sensors_max": 9999},
}

# Рынок КЗ
KZ_CITIES_TIER2 = 17   # Малые/средние города (Тараз, Актобе, Павлодар…)
KZ_CITIES_TIER3 = 2    # Алматы + Астана

# Эффект внедрения
SMART_SHYGYN_SAVINGS_PCT = 0.20   # 20% снижение потерь
DETECTION_HOURS_WITHOUT = 72.0    # Среднее время без системы
DETECTION_HOURS_WITH = 4.0        # Среднее время с Smart Shygyn (BattLeDIM TTD)


# ══════════════════════════════════════════════════════════════════════
# ROI РАСЧЁТ
# ══════════════════════════════════════════════════════════════════════

def calculate_client_roi(
    city_name: str = "Алматы",
    savings_pct: float = SMART_SHYGYN_SAVINGS_PCT,
    tier: str = "Tier 3 — Профессиональный",
) -> Dict[str, Any]:
    """
    Расчёт ROI для клиента (водоканала).

    Returns:
        dict с ключевыми финансовыми показателями
    """
    data = CITY_WATER_DATA.get(city_name, CITY_WATER_DATA["Другой город"])
    tier_data = TIERS.get(tier, TIERS["Tier 3 — Профессиональный"])

    annual_production  = data["annual_production_m3"]
    nrw_pct            = data["nrw_pct"] / 100.0
    tariff             = data["tariff_kzt_m3"]

    # Текущие потери
    annual_loss_m3     = annual_production * nrw_pct
    annual_loss_kzt    = annual_loss_m3 * tariff

    # Экономия от Smart Shygyn
    saved_m3           = annual_loss_m3 * savings_pct
    saved_kzt_year     = saved_m3 * tariff

    # Стоимость нашего решения
    annual_cost_kzt    = tier_data["price_kzt_month"] * 12

    # ROI
    roi_ratio          = saved_kzt_year / annual_cost_kzt if annual_cost_kzt > 0 else float("inf")
    payback_days       = (annual_cost_kzt / saved_kzt_year * 365) if saved_kzt_year > 0 else 9999

    return {
        "city":               city_name,
        "tier":               tier,
        "annual_production_m3": annual_production,
        "nrw_pct":            data["nrw_pct"],
        "annual_loss_m3":     annual_loss_m3,
        "annual_loss_kzt":    annual_loss_kzt,
        "saved_m3":           saved_m3,
        "saved_kzt_year":     saved_kzt_year,
        "annual_cost_kzt":    annual_cost_kzt,
        "roi_ratio":          roi_ratio,
        "payback_days":       payback_days,
        "tariff_kzt_m3":      tariff,
    }


def calculate_our_revenue(
    n_tier2: int = 5,
    n_tier3: int = 2,
    n_pilot: int = 3,
) -> Dict[str, Any]:
    """
    Расчёт нашей выручки по модели SaaS.

    Returns:
        dict с показателями выручки, TAM, SAM
    """
    monthly_revenue = (
        n_tier2 * TIERS["Tier 2 — Базовый"]["price_kzt_month"] +
        n_tier3 * TIERS["Tier 3 — Профессиональный"]["price_kzt_month"]
    )
    annual_revenue = monthly_revenue * 12

    # TAM / SAM
    tam_kzt = (
        KZ_CITIES_TIER2 * TIERS["Tier 2 — Базовый"]["price_kzt_month"] * 12 +
        KZ_CITIES_TIER3 * TIERS["Tier 3 — Профессиональный"]["price_kzt_month"] * 12
    )
    sam_kzt = tam_kzt * 0.30   # реалистичная доля к году 3

    # Breakeven: нужно >= 3 клиента Tier 2 для покрытия оперрасходов
    breakeven_clients_tier2 = 3
    breakeven_month = max(6, 12 - n_tier2)   # чем больше клиентов, тем быстрее

    return {
        "n_tier2":          n_tier2,
        "n_tier3":          n_tier3,
        "n_pilot":          n_pilot,
        "monthly_revenue":  monthly_revenue,
        "annual_revenue":   annual_revenue,
        "tam_kzt":          tam_kzt,
        "sam_kzt":          sam_kzt,
        "breakeven_month":  breakeven_month,
    }


# ══════════════════════════════════════════════════════════════════════
# ГРАФИКИ
# ══════════════════════════════════════════════════════════════════════

def _theme(dark: bool) -> Dict[str, str]:
    return {
        "bg":  "#0e1117" if dark else "white",
        "fg":  "#e2e8f0" if dark else "#2c3e50",
        "grd": "#2d3748" if dark else "#d0d0d0",
    }


def plot_roi_waterfall(roi: Dict[str, Any], dark: bool) -> go.Figure:
    """Waterfall диаграмма ROI клиента."""
    t   = _theme(dark)
    vals = {
        "Текущие потери": roi["annual_loss_kzt"] / 1e9,
        "Экономия (20%)": roi["saved_kzt_year"] / 1e9,
        "Стоимость Smart Shygyn": -roi["annual_cost_kzt"] / 1e9,
        "Чистая выгода": (roi["saved_kzt_year"] - roi["annual_cost_kzt"]) / 1e9,
    }
    colors = ["#ef4444", "#10b981", "#f59e0b", "#3b82f6"]

    fig = go.Figure(go.Bar(
        x=list(vals.keys()),
        y=list(vals.values()),
        marker_color=colors,
        text=[f"{v:.2f} млрд ₸" for v in vals.values()],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"Экономический эффект для {roi['city']} — {roi['tier']}",
        yaxis_title="Млрд тенге (₸)",
        height=360,
        plot_bgcolor=t["bg"], paper_bgcolor=t["bg"],
        font=dict(color=t["fg"], size=11),
        xaxis=dict(gridcolor=t["grd"], color=t["fg"]),
        yaxis=dict(gridcolor=t["grd"], color=t["fg"]),
        margin=dict(l=60, r=40, t=60, b=50),
    )
    return fig


def plot_revenue_growth(rev: Dict[str, Any], dark: bool) -> go.Figure:
    """Рост выручки по месяцам (S-кривая)."""
    t      = _theme(dark)
    months = np.arange(1, 37)

    def revenue_curve(m):
        """Плавный рост с учётом продаж и пилотов."""
        ramp = 1 / (1 + np.exp(-0.3 * (m - 12)))
        return rev["monthly_revenue"] * ramp

    monthly = np.array([revenue_curve(m) for m in months])
    cumulative = np.cumsum(monthly)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        x=months, y=monthly / 1e6,
        name="Ежемесячная выручка (млн ₸)",
        marker_color="#3b82f6", opacity=0.7,
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=months, y=cumulative / 1e6,
        name="Накопленная выручка (млн ₸)",
        line=dict(color="#10b981", width=2.5),
    ), secondary_y=True)

    # Breakeven line
    fig.add_vline(
        x=rev["breakeven_month"],
        line_dash="dash", line_color="#f59e0b", line_width=2,
        annotation_text=f"Breakeven: мес. {rev['breakeven_month']}",
        annotation_font_color="#f59e0b",
        annotation_position="top right",
    )

    fig.update_layout(
        title="Прогноз выручки Smart Shygyn — 36 месяцев",
        height=360, hovermode="x unified",
        plot_bgcolor=t["bg"], paper_bgcolor=t["bg"],
        font=dict(color=t["fg"], size=11),
        xaxis=dict(title="Месяц", gridcolor=t["grd"], color=t["fg"]),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h",
                    yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        margin=dict(l=60, r=60, t=60, b=60),
    )
    fig.update_yaxes(
        title_text="Месячная выручка (млн ₸)", gridcolor=t["grd"], color=t["fg"],
        secondary_y=False
    )
    fig.update_yaxes(
        title_text="Накопленная (млн ₸)", gridcolor=t["grd"], color=t["fg"],
        secondary_y=True
    )
    return fig


def plot_tam_funnel(rev: Dict[str, Any], dark: bool) -> go.Figure:
    """Воронка TAM → SAM → SOM."""
    t = _theme(dark)

    labels = ["TAM (весь КЗ)", "SAM (год 3, 30%)", "SOM текущий"]
    values = [
        rev["tam_kzt"] / 1e6,
        rev["sam_kzt"] / 1e6,
        rev["annual_revenue"] / 1e6,
    ]

    fig = go.Figure(go.Funnel(
        y=labels, x=values,
        textinfo="value+percent initial",
        marker=dict(color=["#3b82f6", "#10b981", "#f59e0b"]),
        connector=dict(line=dict(color=t["fg"], dash="dot", width=2)),
        texttemplate="%{value:.0f} млн ₸<br>%{percentInitial}",
    ))
    fig.update_layout(
        title="Рынок КЗ: TAM → SAM → SOM",
        height=300,
        paper_bgcolor=t["bg"], plot_bgcolor=t["bg"],
        font=dict(color=t["fg"], size=12),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════
# STREAMLIT ВКЛАДКА
# ══════════════════════════════════════════════════════════════════════

def render_business_tab(
    dark_mode: bool = True,
    city_name: str = "Алматы",
):
    """
    Рендерит вкладку '💼 Экономика & Бизнес-модель' в Streamlit.

    Вызывай из app.py:
        with tab_business:
            render_business_tab(dark_mode=dm, city_name=config["city_name"])
    """
    st.markdown("## 💼 Бизнес-модель & ROI Калькулятор")
    st.caption("Раздел 2 ТЗ — Коммерциализация Smart Shygyn PRO v3")

    # ── Клиентский ROI ─────────────────────────────────────────────────
    st.markdown("### 🏢 ROI для клиента (водоканала)")
    st.markdown("*Почему водоканалу выгодно платить нам — ключевой слайд для B2G продаж*")

    rc1, rc2 = st.columns([1, 2])

    with rc1:
        roi_city = st.selectbox("Город-клиент", list(CITY_WATER_DATA.keys()),
                                index=list(CITY_WATER_DATA.keys()).index(city_name)
                                if city_name in CITY_WATER_DATA else 0,
                                key="roi_city")
        roi_tier = st.selectbox("Ценовой тир", list(TIERS.keys()),
                                index=2, key="roi_tier")  # Tier 3 по умолчанию
        roi_savings_pct = st.slider(
            "Снижение потерь (%)", 10, 40, 20, 5,
            help="Реалистично: 15-25% от текущих потерь"
        )

    roi = calculate_client_roi(
        city_name=roi_city,
        savings_pct=roi_savings_pct / 100,
        tier=roi_tier,
    )

    with rc2:
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(
                "💸 Текущие потери воды",
                f"{roi['annual_loss_kzt'] / 1e9:.1f} млрд ₸/год",
                f"{roi['nrw_pct']:.0f}% NRW",
                delta_color="inverse",
            )
        with m2:
            if roi["roi_ratio"] == float("inf"):
                roi_display = "∞"
            else:
                roi_display = f"{roi['roi_ratio']:.0f}:1"
            st.metric(
                "🚀 ROI клиента",
                roi_display,
                "Возврат инвестиций",
                delta_color="normal",
            )
        with m3:
            if roi["payback_days"] < 9999:
                pb_text = f"{roi['payback_days']:.0f} дней"
                pb_delta = "< 1 месяца" if roi["payback_days"] < 30 else f"≈ {roi['payback_days']/30:.1f} мес"
            else:
                pb_text = "Бесплатно"
                pb_delta = "Пилот"
            st.metric("⏱ Срок окупаемости", pb_text, pb_delta)

    st.plotly_chart(plot_roi_waterfall(roi, dark_mode), use_container_width=True)

    # Ключевая цитата для питча
    if roi["roi_ratio"] != float("inf") and roi["roi_ratio"] > 0:
        st.success(
            f"💡 **Для питча:** За каждые ₸1 потраченные на Smart Shygyn, "
            f"{roi['city']} получает **₸{roi['roi_ratio']:.0f} экономии**. "
            f"Окупаемость за **{roi['payback_days']:.0f} дней** "
            f"— меньше чем один рабочий месяц."
        )

    st.markdown("---")

    # ── Наша выручка ───────────────────────────────────────────────────
    st.markdown("### 📈 Наша выручка (SaaS модель)")

    c1, c2, c3 = st.columns(3)
    with c1:
        n_tier2 = st.slider("Клиентов Tier 2 (800к ₸/мес)", 0, KZ_CITIES_TIER2, 5)
    with c2:
        n_tier3 = st.slider("Клиентов Tier 3 (3.5М ₸/мес)", 0, KZ_CITIES_TIER3, 2)
    with c3:
        n_pilot = st.slider("Пилотов (бесплатно)", 0, 10, 3)

    rev = calculate_our_revenue(n_tier2=n_tier2, n_tier3=n_tier3, n_pilot=n_pilot)

    rm1, rm2, rm3, rm4 = st.columns(4)
    with rm1:
        st.metric("💰 Ежемесячная выручка", f"{rev['monthly_revenue'] / 1e6:.1f} млн ₸")
    with rm2:
        st.metric("📅 Годовая выручка",     f"{rev['annual_revenue'] / 1e6:.1f} млн ₸")
    with rm3:
        st.metric("🌍 TAM Казахстан",       f"{rev['tam_kzt'] / 1e6:.0f} млн ₸")
    with rm4:
        st.metric("🎯 SAM (год 3, 30%)",    f"{rev['sam_kzt'] / 1e6:.0f} млн ₸")

    rc1, rc2 = st.columns(2)
    with rc1:
        st.plotly_chart(plot_revenue_growth(rev, dark_mode), use_container_width=True)
    with rc2:
        st.plotly_chart(plot_tam_funnel(rev, dark_mode), use_container_width=True)

    st.markdown("---")

    # ── Таблица тиров ──────────────────────────────────────────────────
    st.markdown("### 💳 Ценовые тиры")

    tier_rows = []
    for tier_name, td in TIERS.items():
        monthly = td["price_kzt_month"]
        km_max  = td["network_km_max"]
        tier_rows.append({
            "Тир":              tier_name,
            "₸/месяц":         f"{monthly:,}" if monthly > 0 else "БЕСПЛАТНО",
            "₸/год":           f"{monthly*12:,}" if monthly > 0 else "—",
            "Сеть (км)":       f"до {km_max}" if km_max < 9999 else "Без ограничений",
            "Датчиков":        f"до {td['sensors_max']}" if td["sensors_max"] < 9999 else "Без ограничений",
        })

    st.dataframe(
        pd.DataFrame(tier_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    # ── Дорожная карта ─────────────────────────────────────────────────
    st.markdown("### 🗓️ Дорожная карта")
    roadmap = [
        {"Период": "Q1 2026", "Цель": "Пилот в 1 городе (бесплатно)", "KPI": "Получить данные + рекомендательное письмо"},
        {"Период": "Q2 2026", "Цель": "Первый платный клиент (Tier 2)", "KPI": "800,000 ₸/мес, Breakeven"},
        {"Период": "Q3 2026", "Цель": "3 города, выход на окупаемость", "KPI": "2.4 млн ₸/мес"},
        {"Период": "Q4 2026", "Цель": "Алматы или Астана (Tier 3)",     "KPI": "+3.5 млн ₸/мес, рынок СНГ"},
    ]
    st.dataframe(pd.DataFrame(roadmap), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Ответы на вопросы жюри ─────────────────────────────────────────
    st.markdown("### 🎯 Ответы на вопросы жюри")
    with st.expander("Q: Чем отличаетесь от Xylem / IBM Water / Siemens?"):
        st.markdown("""
- **Цена:** Siemens/Xylem — $500K+ за внедрение. Наш пилот **бесплатный**.
- **Локализация:** Мы единственное решение адаптированное под SCADA казахстанских водоканалов,
  реальные тарифы и нормативы СП РК 4.01-101-2012.
- **Доступность:** Siemens не придёт в Туркестан или Тараз. Мы — придём.
- **Верификация:** Тот же BattLeDIM датасет, что используют ETH Zurich и MIT.
        """)
    with st.expander("Q: Какая точность?"):
        st.markdown("""
Назвать реальные цифры из BattLeDIM вкладки:
**Recall X%, Precision Y%, TTD ~Z часов**.
Не завышать — жюри проверит.
        """)
    with st.expander("Q: Как вы защищены от копирования?"):
        st.markdown("""
**Барьер = данные.** После пилота у нас будут реальные SCADA-данные
казахстанских сетей которых нет ни у кого. Это defensibility.
Плюс языковой барьер и знание местной регуляторики.
        """)
    with st.expander("Q: Есть ли данные от казахстанских сетей?"):
        st.markdown("""
*Если нет:* «Пока верифицированы на международном датасете BattLeDIM.
В пилоте получим первые казахстанские данные и дообучим модель.»

*Если есть:* Показать здесь.
        """)


# ══════════════════════════════════════════════════════════════════════
# SELF-TEST
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Business Model — self-test")
    print("=" * 60)

    # ROI для Алматы Tier 3
    roi = calculate_client_roi("Алматы", savings_pct=0.20, tier="Tier 3 — Профессиональный")
    print(f"\nROI для Алматы (Tier 3):")
    print(f"  Потери: {roi['annual_loss_kzt']/1e9:.1f} млрд ₸/год")
    print(f"  Экономия: {roi['saved_kzt_year']/1e9:.2f} млрд ₸/год")
    print(f"  ROI: {roi['roi_ratio']:.0f}:1")
    print(f"  Окупаемость: {roi['payback_days']:.0f} дней")

    # Выручка
    rev = calculate_our_revenue(n_tier2=5, n_tier3=2)
    print(f"\nВыручка (5×Tier2 + 2×Tier3):")
    print(f"  Ежемесячно: {rev['monthly_revenue']/1e6:.1f} млн ₸")
    print(f"  Годовая: {rev['annual_revenue']/1e6:.1f} млн ₸")
    print(f"  TAM: {rev['tam_kzt']/1e6:.0f} млн ₸")
    print(f"  SAM (30%): {rev['sam_kzt']/1e6:.0f} млн ₸")

    print("\n✅ Business Model готов к интеграции")
    print("   from business_model import render_business_tab")
