"""
Smart Shygyn PRO v3 — ML Engine
FIXED v2:
- ML Classifier uses relative flow (% above mean) not absolute L/s values
  so it doesn't flag CATASTROPHIC when pump_head is high (Алматы 1040m)
- classify_flow_rate() now normalises against rolling baseline mean
- All original logic preserved
"""

import logging
import hashlib
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

logger = logging.getLogger("smart_shygyn.ml_engine")

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False
    logger.warning("scikit-learn не установлен. Isolation Forest недоступен.")

try:
    import streamlit as st
    _STREAMLIT_OK = True
except ImportError:
    _STREAMLIT_OK = False

# ══════════════════════════════════════════════════════════════════════
# НАЗВАНИЯ МЕТОДОВ
# ══════════════════════════════════════════════════════════════════════

METHODS = ["Z-score", "Isolation Forest", "Ensemble"]


# ══════════════════════════════════════════════════════════════════════
# FIXED: RELATIVE FLOW CLASSIFIER
# ══════════════════════════════════════════════════════════════════════

def classify_flow_rate(
    flow_lps: float,
    baseline_mean_lps: float,
) -> Dict[str, Any]:
    """
    FIXED: classify flow anomaly using relative deviation from baseline mean.

    Previously used absolute L/s thresholds which broke when pump_head
    was auto-boosted to 1040m for Алматы (normal flow ~700+ L/s flagged
    as CATASTROPHIC).

    Now uses percentage above baseline:
        deviation% = (flow - mean) / mean * 100

    Thresholds (relative):
        < 20%   → NORMAL
        20–60%  → BACKGROUND LEAK
        60–150% → BURST
        > 150%  → CATASTROPHIC
    """
    if baseline_mean_lps <= 0:
        return {
            "classification": "NORMAL",
            "severity": 0.0,
            "estimated_leak_lps": 0.0,
        }

    deviation_pct = (flow_lps - baseline_mean_lps) / baseline_mean_lps * 100.0

    if deviation_pct < 20.0:
        classification = "NORMAL"
        severity = max(0.0, deviation_pct / 20.0 * 10.0)          # 0–10
    elif deviation_pct < 60.0:
        classification = "BACKGROUND LEAK"
        severity = 10.0 + (deviation_pct - 20.0) / 40.0 * 30.0   # 10–40
    elif deviation_pct < 150.0:
        classification = "BURST"
        severity = 40.0 + (deviation_pct - 60.0) / 90.0 * 40.0   # 40–80
    else:
        classification = "CATASTROPHIC"
        severity = min(100.0, 80.0 + (deviation_pct - 150.0) / 100.0 * 20.0)  # 80–100

    estimated_leak_lps = max(0.0, flow_lps - baseline_mean_lps)

    return {
        "classification":    classification,
        "severity":          round(severity, 1),
        "estimated_leak_lps": round(estimated_leak_lps, 3),
        "deviation_pct":     round(deviation_pct, 1),
        "baseline_mean_lps": round(baseline_mean_lps, 3),
    }


def classify_from_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Convenience wrapper: compute baseline from first 6 hours (night minimum),
    then classify the mean flow of the whole series.

    Uses first 6 rows as the 'normal' reference window.
    Falls back to full-series mean if < 6 rows available.
    """
    flow_col = "Flow Rate (L/s)"
    if flow_col not in df.columns or len(df) == 0:
        return {
            "classification": "NORMAL",
            "severity": 0.0,
            "estimated_leak_lps": 0.0,
        }

    flow_series = df[flow_col].values

    # Baseline = mean of first 6 hours (night, lowest demand)
    n_baseline = min(6, len(flow_series))
    baseline_mean = float(np.mean(flow_series[:n_baseline]))

    # Guard: if baseline is zero or near-zero, use full series mean
    if baseline_mean < 1e-3:
        baseline_mean = float(np.mean(flow_series))

    current_flow = float(np.mean(flow_series))

    return classify_flow_rate(current_flow, baseline_mean)


# ══════════════════════════════════════════════════════════════════════
# ISOLATION FOREST ДЕТЕКТОР
# ══════════════════════════════════════════════════════════════════════

class IsolationForestDetector:
    def __init__(self,
                 contamination: float = 0.05,
                 n_estimators: int = 200,
                 random_state: int = 42):
        if not _SKLEARN_OK:
            raise ImportError(
                "Для IsolationForestDetector нужен scikit-learn: "
                "pip install scikit-learn"
            )
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model: Optional[IsolationForest] = None
        self.scaler: Optional[StandardScaler] = None
        self._trained = False

    def train(self, scada_2018_df: pd.DataFrame) -> None:
        data = scada_2018_df.dropna(axis=1, how="all").dropna(how="any")
        if data.empty:
            raise ValueError("scada_2018_df пуст после удаления NaN")

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(data.values)

        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X_scaled)
        self._trained = True
        self._feature_cols = list(data.columns)
        logger.info(
            "IsolationForest обучен: %d строк, %d датчиков",
            len(data), len(self._feature_cols)
        )

    def _prepare(self, scada_df: pd.DataFrame) -> np.ndarray:
        if not self._trained:
            raise RuntimeError("Сначала вызовите .train()")

        df_aligned = pd.DataFrame(index=scada_df.index)
        for col in self._feature_cols:
            if col in scada_df.columns:
                df_aligned[col] = scada_df[col]
            else:
                df_aligned[col] = 0.0

        df_filled = df_aligned.fillna(df_aligned.mean())
        return self.scaler.transform(df_filled.values)

    def predict(self, scada_df: pd.DataFrame) -> pd.Series:
        X = self._prepare(scada_df)
        preds = self.model.predict(X)
        flags = pd.Series(preds == -1, index=scada_df.index)
        logger.debug("IF anomalies: %d / %d", flags.sum(), len(flags))
        return flags

    def anomaly_score(self, scada_df: pd.DataFrame) -> pd.Series:
        X = self._prepare(scada_df)
        raw_scores = self.model.decision_function(X)
        inverted = -raw_scores
        s_min, s_max = inverted.min(), inverted.max()
        if s_max - s_min < 1e-9:
            normalized = np.zeros_like(inverted)
        else:
            normalized = (inverted - s_min) / (s_max - s_min)
        return pd.Series(normalized, index=scada_df.index)


# ══════════════════════════════════════════════════════════════════════
# STREAMLIT КЭШИРОВАННАЯ ФАБРИКА
# ══════════════════════════════════════════════════════════════════════

def _df_hash(df: pd.DataFrame) -> str:
    return hashlib.md5(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()[:16]


def get_isolation_forest_model(
    scada_2018_df: pd.DataFrame,
    contamination: float = 0.05,
) -> IsolationForestDetector:
    if not _SKLEARN_OK:
        raise ImportError("pip install scikit-learn>=1.3.0")

    cache_key = _df_hash(scada_2018_df) + f"_{contamination}"

    if _STREAMLIT_OK:
        return _cached_model(cache_key, scada_2018_df, contamination)
    else:
        return _build_model(scada_2018_df, contamination)


def _build_model(
    scada_2018_df: pd.DataFrame,
    contamination: float,
) -> IsolationForestDetector:
    detector = IsolationForestDetector(contamination=contamination)
    detector.train(scada_2018_df)
    return detector


if _STREAMLIT_OK:
    import streamlit as st

    @st.cache_resource
    def _cached_model(
        cache_key: str,
        scada_2018_df: pd.DataFrame,
        contamination: float,
    ) -> IsolationForestDetector:
        logger.info("Обучаем IsolationForest (cache_key=%s)…", cache_key)
        return _build_model(scada_2018_df, contamination)
else:
    def _cached_model(cache_key, scada_2018_df, contamination):
        return _build_model(scada_2018_df, contamination)


# ══════════════════════════════════════════════════════════════════════
# Z-SCORE ДЕТЕКЦИЯ
# ══════════════════════════════════════════════════════════════════════

def detect_zscore(
    scada_2019_df: pd.DataFrame,
    baseline: "pd.DataFrame",
    z_threshold: float = 3.0,
    min_sensors: int = 2,
) -> pd.Series:
    sensors = [c for c in scada_2019_df.columns
               if f"mean_{c}" in baseline.columns
               and f"std_{c}" in baseline.columns]

    if not sensors:
        return pd.Series(False, index=scada_2019_df.index)

    intra = (scada_2019_df.index.hour * 60 + scada_2019_df.index.minute) // 5
    intra_vals = intra.values

    mean_cols = [f"mean_{c}" for c in sensors]
    std_cols  = [f"std_{c}"  for c in sensors]
    mu_matrix  = baseline[mean_cols].values.astype(float)
    sig_matrix = baseline[std_cols].values.astype(float)

    mu_aligned  = mu_matrix[intra_vals]
    sig_aligned = sig_matrix[intra_vals]
    data        = scada_2019_df[sensors].values.astype(float)

    with np.errstate(invalid="ignore", divide="ignore"):
        z = (mu_aligned - data) / np.where(sig_aligned < 1e-6, np.inf, sig_aligned)

    triggered = (z > z_threshold).sum(axis=1)
    return pd.Series(triggered >= min_sensors, index=scada_2019_df.index)


# ══════════════════════════════════════════════════════════════════════
# ОСНОВНАЯ ФУНКЦИЯ
# ══════════════════════════════════════════════════════════════════════

def detect_anomalies_ml(
    scada_2019_df: pd.DataFrame,
    method: str = "Isolation Forest",
    scada_2018_df: Optional[pd.DataFrame] = None,
    baseline: Optional[pd.DataFrame] = None,
    z_threshold: float = 3.0,
    min_sensors: int = 2,
    contamination: float = 0.05,
) -> Tuple[pd.Series, Dict[str, Any]]:
    meta: Dict[str, Any] = {"method": method}

    if method == "Z-score":
        if baseline is None:
            raise ValueError("Для Z-score нужен baseline (вызови build_baseline)")
        flags = detect_zscore(scada_2019_df, baseline, z_threshold, min_sensors)
        meta["z_score_detections"] = int(flags.sum())
        return flags, meta

    if method == "Isolation Forest":
        if not _SKLEARN_OK:
            logger.warning("scikit-learn отсутствует, fallback на Z-score")
            if baseline is not None:
                return detect_zscore(scada_2019_df, baseline, z_threshold, min_sensors), {
                    "method": "Z-score (fallback)", "reason": "sklearn missing"
                }
            return pd.Series(False, index=scada_2019_df.index), {"method": "none"}

        if scada_2018_df is None:
            raise ValueError("Для Isolation Forest нужны данные 2018")

        detector = get_isolation_forest_model(scada_2018_df, contamination)
        flags    = detector.predict(scada_2019_df)
        scores   = detector.anomaly_score(scada_2019_df)
        meta["if_detections"] = int(flags.sum())
        meta["if_score_mean"] = float(scores.mean())
        meta["anomaly_score"] = scores
        return flags, meta

    if method == "Ensemble":
        results: Dict[str, pd.Series] = {}

        if baseline is not None:
            results["Z-score"] = detect_zscore(
                scada_2019_df, baseline, z_threshold, min_sensors
            )
            meta["z_score_detections"] = int(results["Z-score"].sum())

        if scada_2018_df is not None and _SKLEARN_OK:
            detector = get_isolation_forest_model(scada_2018_df, contamination)
            results["IF"] = detector.predict(scada_2019_df)
            scores = detector.anomaly_score(scada_2019_df)
            meta["if_detections"] = int(results["IF"].sum())
            meta["anomaly_score"] = scores
        elif scada_2018_df is not None and not _SKLEARN_OK:
            meta["if_skipped"] = "sklearn missing"

        if not results:
            return pd.Series(False, index=scada_2019_df.index), {"method": "Ensemble (empty)"}

        combined = pd.Series(False, index=scada_2019_df.index)
        for flags in results.values():
            combined = combined | flags

        meta["ensemble_detections"] = int(combined.sum())
        meta["n_methods"]           = len(results)
        return combined, meta

    raise ValueError(f"Неизвестный метод: {method}. Допустимые: {METHODS}")


# ══════════════════════════════════════════════════════════════════════
# СРАВНИТЕЛЬНАЯ ТАБЛИЦА МЕТРИК
# ══════════════════════════════════════════════════════════════════════

def compare_methods(
    scada_2019_df: pd.DataFrame,
    leak_events: Optional[pd.DataFrame],
    scada_2018_df: Optional[pd.DataFrame] = None,
    baseline: Optional[pd.DataFrame] = None,
    z_threshold: float = 3.0,
    min_sensors: int = 2,
    contamination: float = 0.05,
) -> pd.DataFrame:
    from battledim_analysis import compute_metrics

    rows = []
    for method in METHODS:
        try:
            flags, meta = detect_anomalies_ml(
                scada_2019_df,
                method=method,
                scada_2018_df=scada_2018_df,
                baseline=baseline,
                z_threshold=z_threshold,
                min_sensors=min_sensors,
                contamination=contamination,
            )
            m = compute_metrics(flags, leak_events)
            rows.append({
                "Метод":       method,
                "Recall %":    m.get("recall"),
                "Precision %": m.get("precision"),
                "F1 %":        m.get("f1"),
                "TTD (ч)":     m.get("ttd_hours"),
                "Детекций":    int(flags.sum()),
                "TP":          m.get("detected"),
                "FP":          m.get("fp"),
            })
        except Exception as exc:
            rows.append({
                "Метод": method,
                "Recall %": None, "Precision %": None,
                "F1 %": None, "TTD (ч)": None,
                "Детекций": 0, "TP": 0, "FP": 0,
                "_error": str(exc),
            })

    return pd.DataFrame(rows)
