"""
Smart Shygyn PRO v3 — Data Loader
BattLeDIM dataset + Kazakhstan real data.

Strategy (priority order):
1. GitHub Releases (fast CDN, no rate limits)
2. Zenodo (fallback, slower)
- ALL files are optional — app works with whatever it gets
- Success = at least one SCADA file downloaded
"""

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import requests

logger = logging.getLogger("smart_shygyn.data_loader")

# ═══════════════════════════════════════════════════════════════════════════
# REAL KAZAKHSTAN DATA
# ═══════════════════════════════════════════════════════════════════════════

KAZAKHSTAN_REAL_DATA = {
    "tariff_almaty_kzt_per_m3":    91.96,
    "tariff_astana_kzt_per_m3":    85.00,
    "tariff_turkestan_kzt_per_m3": 70.00,
    "wear_almaty_pct":    54.5,
    "wear_astana_pct":    48.0,
    "wear_turkestan_pct": 62.0,
    "age_almaty_years":    45,
    "age_astana_years":    38,
    "age_turkestan_years": 50,
    "pressure_sensor_cost_kzt": 380_000,
    "flow_sensor_cost_kzt":     520_000,
    "smart_meter_cost_kzt":      45_000,
    "electricity_tariff_kzt_per_kwh": 22.0,
    "co2_kg_per_kwh":                  0.62,
    "min_pressure_residential_bar": 2.5,
    "max_pressure_residential_bar": 6.0,
}


def get_real_tariff(city_name: str) -> float:
    m = {"Алматы": 91.96, "Астана": 85.00, "Туркестан": 70.00}
    return m.get(city_name, 91.96) / 1000.0


def get_real_pipe_wear(city_name: str) -> float:
    m = {"Алматы": 54.5, "Астана": 48.0, "Туркестан": 62.0}
    return m.get(city_name, 50.0)


def get_estimated_pipe_age(city_name: str) -> int:
    m = {"Алматы": 45, "Астана": 38, "Туркестан": 50}
    return m.get(city_name, 40)


# ═══════════════════════════════════════════════════════════════════════════
# GITHUB RELEASES CONFIG  ← основной источник
# ═══════════════════════════════════════════════════════════════════════════

GITHUB_RELEASE_BASE = (
    "https://github.com/Venture0720/TournamentProj"
    "/releases/download/v1.0-data"
)

# Точные имена файлов как они названы в GitHub Release
GITHUB_FILES: Dict[str, str] = {
    "scada_2018":  "2018_SCADA.xlsx",
    "scada_2019":  "2019_SCADA.xlsx",
    "network_inp": "L-TOWN.inp",
    "leaks_2019":  "2019_Leakages.csv",
}

# ═══════════════════════════════════════════════════════════════════════════
# BATTLEDIM CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

BATTLEDIM_DATA_DIR = Path("data/battledim")
ZENODO_RECORD_ID   = "4017659"

# Zenodo fallback filenames (пробуем если GitHub не сработал)
FILENAME_ALTERNATIVES: Dict[str, List[str]] = {
    "scada_2018": [
        "2018 SCADA.xlsx",
        "2018_SCADA.xlsx",
        "2018 SCADA.xls",
        "SCADA_2018.xlsx",
    ],
    "scada_2019": [
        "2019 SCADA.xlsx",
        "2019_SCADA.xlsx",
        "2019 SCADA.xls",
        "SCADA_2019.xlsx",
    ],
    "network_inp": [
        "L-Town.inp",
        "L-TOWN.inp",
        "l-town.inp",
        "LTOWN.inp",
        "network.inp",
    ],
    "leaks_2019": [
        "Leak_Labels.xlsx",
        "Leak_Labels.csv",
        "2019 Leaks.csv",
        "2019_Leakages.csv",
        "GroundTruth.csv",
    ],
}

# Канонические имена для сохранения локально
CANONICAL_NAMES = {
    "scada_2018":  "2018_SCADA.xlsx",
    "scada_2019":  "2019_SCADA.xlsx",
    "network_inp": "L-Town.inp",
    "leaks_2019":  "Leak_Labels.xlsx",
}


# ═══════════════════════════════════════════════════════════════════════════
# LOADER
# ═══════════════════════════════════════════════════════════════════════════

class BattLeDIMLoader:
    """
    Loader for BattLeDIM 2020 (L-Town, Limassol, Cyprus).
    Downloads from GitHub Releases first, then falls back to Zenodo.
    DOI: 10.5281/zenodo.4017659
    """

    def __init__(self, data_dir: Path = BATTLEDIM_DATA_DIR):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ── Helpers ──────────────────────────────────────────────────────────

    def _resolve_path(self, key: str) -> Optional[Path]:
        """Return first existing non-empty file for this key."""
        all_names = FILENAME_ALTERNATIVES.get(key, []) + [CANONICAL_NAMES.get(key, "")]
        for name in all_names:
            if not name:
                continue
            p = self.data_dir / name
            if p.exists() and p.stat().st_size > 200:
                return p
        return None

    def check_files_exist(self) -> Dict[str, bool]:
        return {key: self._resolve_path(key) is not None
                for key in FILENAME_ALTERNATIVES}

    def required_files_present(self) -> bool:
        """Minimum: at least one SCADA file must exist."""
        return (self._resolve_path("scada_2018") is not None or
                self._resolve_path("scada_2019") is not None)

    def all_files_present(self) -> bool:
        return all(self.check_files_exist().values())

    # ── Universal URL downloader ──────────────────────────────────────────

    def _try_url(self, url: str, local_dest: Path) -> bool:
        """
        Download file from any direct URL.
        Works with GitHub Releases, Zenodo, or any CDN.
        Returns True on success.
        """
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 SmartShygyn/3.0",
                "Accept": "*/*",
            }
            with requests.get(url, stream=True, timeout=180,
                              headers=headers, allow_redirects=True) as r:
                if r.status_code != 200:
                    logger.debug("HTTP %d for %s", r.status_code, url)
                    return False
                # Reject HTML error pages (e.g. 404 pages from GitHub)
                ct = r.headers.get("Content-Type", "")
                if "text/html" in ct:
                    logger.debug("Got HTML instead of file from %s", url)
                    return False
                with open(local_dest, "wb") as f:
                    shutil.copyfileobj(r.raw, f)

            if local_dest.exists() and local_dest.stat().st_size > 200:
                logger.info("✓ Downloaded %s → %s (%.0f KB)",
                            url.split("/")[-1],
                            local_dest.name,
                            local_dest.stat().st_size / 1024)
                return True

            local_dest.unlink(missing_ok=True)
            return False

        except Exception as exc:
            logger.debug("Download failed %s: %s", url, exc)
            local_dest.unlink(missing_ok=True)
            return False

    # ── Zenodo fallback ───────────────────────────────────────────────────

    def _try_zenodo(self, zenodo_filename: str, local_dest: Path) -> bool:
        """Try to download one file from Zenodo using ?download=1 URL."""
        encoded = zenodo_filename.replace(" ", "%20")
        url = (f"https://zenodo.org/records/{ZENODO_RECORD_ID}"
               f"/files/{encoded}?download=1")
        return self._try_url(url, local_dest)

    # ── Main download ─────────────────────────────────────────────────────

    def download_dataset(self) -> Tuple[bool, str]:
        """
        Download all BattLeDIM files.

        Priority:
        1. GitHub Releases (fast, reliable)
        2. Zenodo (slower, fallback)

        Returns success if ANY SCADA file was downloaded.
        """
        results: Dict[str, bool] = {}

        for key in FILENAME_ALTERNATIVES:

            # Already have it locally?
            if self._resolve_path(key) is not None:
                results[key] = True
                logger.info("Already present: %s", key)
                continue

            canonical = CANONICAL_NAMES[key]
            dest      = self.data_dir / canonical
            got_it    = False

            # ── 1. Try GitHub Releases first ─────────────────────────────
            gh_filename = GITHUB_FILES.get(key)
            if gh_filename:
                url = f"{GITHUB_RELEASE_BASE}/{gh_filename}"
                logger.info("GitHub Release: %s → %s", gh_filename, key)
                got_it = self._try_url(url, dest)
                if got_it:
                    logger.info("✅ Got %s from GitHub Releases", key)

            # ── 2. Zenodo fallback ────────────────────────────────────────
            if not got_it:
                logger.info("GitHub failed for %s — trying Zenodo …", key)
                for zenodo_name in FILENAME_ALTERNATIVES[key]:
                    logger.info("  Zenodo: %s", zenodo_name)
                    if self._try_zenodo(zenodo_name, dest):
                        got_it = True
                        logger.info("✅ Got %s from Zenodo", key)
                        break

            if not got_it:
                logger.info("❌ Not available: %s (optional)", key)

            results[key] = got_it

        # ── Evaluate result ───────────────────────────────────────────────
        got  = [k for k, v in results.items() if v]
        miss = [k for k, v in results.items() if not v]

        scada_ok = results.get("scada_2018") or results.get("scada_2019")

        if not scada_ok:
            return False, (
                "❌ Не удалось скачать SCADA данные.\n"
                f"Попробуй вручную с GitHub:\n"
                f"  {GITHUB_RELEASE_BASE}\n"
                f"или с Zenodo:\n"
                f"  https://zenodo.org/records/{ZENODO_RECORD_ID}\n"
                "и положи файлы в папку 'data/battledim/'"
            )

        if miss:
            return True, (
                f"✅ Скачано: {got}\n"
                f"⚠️ Не найдено (опционально): {miss}\n"
                f"Приложение работает без них."
            )

        return True, f"✅ Все файлы BattLeDIM загружены! ({got})"

    # ── Data loading ──────────────────────────────────────────────────────

    def load_scada_2018(self) -> Optional[Dict[str, pd.DataFrame]]:
        path = self._resolve_path("scada_2018")
        if path is None:
            return None
        try:
            xl = pd.ExcelFile(path)
            df = xl.parse(xl.sheet_names[0], index_col=0, parse_dates=True)
            return {"pressures": df.apply(pd.to_numeric, errors="coerce")}
        except Exception as exc:
            logger.error("Cannot load 2018 SCADA: %s", exc)
            return None

    def load_scada_2019(self) -> Optional[Dict[str, pd.DataFrame]]:
        path = self._resolve_path("scada_2019")
        if path is None:
            return None
        try:
            xl = pd.ExcelFile(path)
            df = xl.parse(xl.sheet_names[0], index_col=0, parse_dates=True)
            return {"pressures": df.apply(pd.to_numeric, errors="coerce")}
        except Exception as exc:
            logger.error("Cannot load 2019 SCADA: %s", exc)
            return None

    def load_leaks_2019(self) -> Optional[pd.DataFrame]:
        path = self._resolve_path("leaks_2019")
        if path is None:
            return None
        try:
            if path.suffix.lower() in (".xlsx", ".xls"):
                return pd.read_excel(path)
            return pd.read_csv(path)
        except Exception as exc:
            logger.error("Cannot load leak labels: %s", exc)
            return None

    def get_network_statistics(self) -> Dict[str, Any]:
        """Published L-Town stats — optionally parsed from INP."""
        stats: Dict[str, Any] = {
            "n_junctions":      782,
            "n_pipes":          909,
            "total_length_km":  42.6,
            "n_leaks_2019":     23,
            "doi":    "10.5281/zenodo.4017659",
            "source": "BattLeDIM 2020 — L-Town, Limassol, Cyprus",
            "status": "HARDCODED",
        }
        inp = self._resolve_path("network_inp")
        if inp:
            try:
                import wntr  # type: ignore
                wn = wntr.network.WaterNetworkModel(str(inp))
                stats.update({
                    "n_junctions": wn.num_junctions,
                    "n_pipes":     wn.num_pipes,
                    "total_length_km": round(
                        sum(wn.get_link(p).length
                            for p in wn.pipe_name_list) / 1000, 2),
                    "status": "LOADED",
                })
            except Exception as exc:
                logger.warning("Cannot parse INP: %s", exc)
        return stats

    def get_pressure_timeseries(self,
                                year: int = 2018,
                                day:  int = 1) -> Optional[pd.DataFrame]:
        fn = self.load_scada_2018 if year == 2018 else self.load_scada_2019
        result = fn()
        if result is None:
            return None
        df = result["pressures"]
        if hasattr(df.index, "dayofyear"):
            try:
                sliced = df[df.index.dayofyear == day]
                return sliced if len(sliced) > 0 else df.iloc[:288]
            except Exception:
                pass
        samp  = 288
        start = (day - 1) * samp
        return df.iloc[start:start + samp] if start < len(df) else df.iloc[:samp]


# ═══════════════════════════════════════════════════════════════════════════
# SINGLETON & PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════

_loader_instance: Optional[BattLeDIMLoader] = None


def get_loader() -> BattLeDIMLoader:
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = BattLeDIMLoader(BATTLEDIM_DATA_DIR)
    return _loader_instance


def initialize_battledim(show_progress: bool = False) -> Tuple[bool, str]:
    """
    Check BattLeDIM presence; download if missing.
    Tries GitHub Releases first, then Zenodo.
    Returns (success, message). Success = at least SCADA files present.
    """
    loader = get_loader()
    if loader.required_files_present():
        status = loader.check_files_exist()
        have   = [k for k, v in status.items() if v]
        return True, f"✅ BattLeDIM готов — файлы присутствуют: {have}"
    logger.info("BattLeDIM отсутствует — скачиваем …")
    return loader.download_dataset()


# ═══════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    for city in ["Алматы", "Астана", "Туркестан"]:
        print(f"{city}: {get_real_tariff(city):.5f} ₸/л | "
              f"{get_real_pipe_wear(city)}% | {get_estimated_pipe_age(city)} лет")

    loader = get_loader()
    print("\nСтатус файлов:", loader.check_files_exist())
    ok, msg = initialize_battledim()
    print(f"Инициализация: {ok} — {msg}")
    print("Статистика сети:", loader.get_network_statistics())
