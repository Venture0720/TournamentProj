"""
Smart Shygyn PRO v3 — ML Engine
Раздел 1 ТЗ: Isolation Forest + Ensemble детекция утечек.

Использование:
    from ml_engine import get_isolation_forest_model, detect_anomalies_ml, METHODS

Методы:
    "Z-score"          — базовый алгоритм (из battledim_analysis.py)
    "Isolation Forest" — unsupervised ML, scikit-learn
    "Ensemble"         — аномалия если ХОТЯ БЫ ОДИН метод обнаружил
"""

import logging
import hashlib
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

logger = logging.getLogger("smart_shygyn.ml_engine")

# Опциональные зависимости
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
# НАЗВАНИЯ МЕТОДОВ (для UI selectbox)
# ══════════════════════════════════════════════════════════════════════

METHODS = ["Z-score", "Isolation Forest", "Ensemble"]


# ══════════════════════════════════════════════════════════════════════
# ISOLATION FOREST ДЕТЕКТОР
# ══════════════════════════════════════════════════════════════════════

class IsolationForestDetector:
    """
    Unsupervised детектор аномалий на основе Isolation Forest.
    Обучается на нормальных данных 2018 года,
    предсказывает аномалии в данных 2019 года.
    """

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
        """
        Обучить модель на данных 2018 (считаются нормальными).
        """
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
        """Выровнять столбцы и нормализовать данные."""
        if not self._trained:
            raise RuntimeError("Сначала вызовите .train()")

        common = [c for c in self._feature_cols if c in scada_df.columns]
        if not common:
            raise ValueError("Нет общих датчиков между 2018 и 2019 данными")

        df_aligned = pd.DataFrame(index=scada_df.index)
        for col in self._feature_cols:
            if col in scada_df.columns:
                df_aligned[col] = scada_df[col]
            else:
                df_aligned[col] = 0.0

        df_filled = df_aligned.fillna(df_aligned.mean())
        return self.scaler.transform(df_filled.values)

    def predict(self, scada_df: pd.DataFrame) -> pd.Series:
        """
        Обнаружить аномалии.
        Returns pd.Series[bool] — True = аномалия.
        """
        X = self._prepare(scada_df)
        preds = self.model.predict(X)   # -1 = аномалия, 1 = нормально
        flags = pd.Series(preds == -1, index=scada_df.index)
        logger.debug("IF anomalies: %d / %d", flags.sum(), len(flags))
        return flags

    def anomaly_score(self, scada_df: pd.DataFrame) -> pd.Series:
        """
        Непрерывная оценка аномальности в [0, 1].
        Выше = аномальнее.
        """
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
    """Быстрый хэш DataFrame для кэша."""
    return hashlib.md5(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()[:16]


def get_isolation_forest_model(
    scada_2018_df: pd.DataFrame,
    contamination: float = 0.05,
) -> IsolationForestDetector:
    """
    Получить (или создать) обученный IsolationForestDetector.
    Если Streamlit доступен — использует @st.cache_resource для кэша.
    """
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
# Z-SCORE ДЕТЕКЦИЯ (обёртка для Ensemble)
# ══════════════════════════════════════════════════════════════════════

def detect_zscore(
    scada_2019_df: pd.DataFrame,
    baseline: "pd.DataFrame",
    z_threshold: float = 3.0,
    min_sensors: int = 2,
) -> pd.Series:
    """
    Z-score детекция (алгоритм из battledim_analysis.py).
    Вынесен сюда для использования в Ensemble.
    """
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
# ОСНОВНАЯ ФУНКЦИЯ: detect_anomalies_ml
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
    """
    Единый интерфейс детекции аномалий для всех методов.

    Args:
        scada_2019_df: SCADA данные 2019
        method:        "Z-score" | "Isolation Forest" | "Ensemble"
        scada_2018_df: Данные 2018 (нужны для IF и Ensemble)
        baseline:      Результат build_baseline() (нужен для Z-score и Ensemble)
        z_threshold:   Порог Z-score
        min_sensors:   Минимум датчиков для тревоги (Z-score)
        contamination: Доля аномалий для IF

    Returns:
        (anomaly_flags, meta)
    """
    meta: Dict[str, Any] = {"method": method}

    # ── Z-score ───────────────────────────────────────────────────────
    if method == "Z-score":
        if baseline is None:
            raise ValueError("Для Z-score нужен baseline (вызови build_baseline)")
        flags = detect_zscore(scada_2019_df, baseline, z_threshold, min_sensors)
        meta["z_score_detections"] = int(flags.sum())
        return flags, meta

    # ── Isolation Forest ──────────────────────────────────────────────
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

    # ── Ensemble ──────────────────────────────────────────────────────
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
    """
    Запустить все три метода и вернуть сравнительную таблицу.
    """
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
