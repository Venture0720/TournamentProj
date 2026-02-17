"""
Smart Shygyn PRO v3 — Data Loader
BattLeDIM dataset integration and Kazakhstan real data.
Handles dataset download via gdown folder, validation, and real-data lookups.
"""

import os
import logging
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger("smart_shygyn.data_loader")

# ═══════════════════════════════════════════════════════════════════════════
# REAL KAZAKHSTAN DATA
# ═══════════════════════════════════════════════════════════════════════════

KAZAKHSTAN_REAL_DATA = {
    # Water tariffs (KZT per m³)
    "tariff_almaty_kzt_per_m3":    91.96,
    "tariff_astana_kzt_per_m3":    85.00,
    "tariff_turkestan_kzt_per_m3": 70.00,

    # Network wear (%) — КРЕМ МНЭ РК, 2024
    "wear_almaty_pct":    54.5,
    "wear_astana_pct":    48.0,
    "wear_turkestan_pct": 62.0,

    # Estimated average pipe age (years)
    "age_almaty_years":    45,
    "age_astana_years":    38,
    "age_turkestan_years": 50,

    # Sensor costs (KZT)
    "pressure_sensor_cost_kzt": 380_000,
    "flow_sensor_cost_kzt":     520_000,
    "smart_meter_cost_kzt":      45_000,

    # Energy & emissions
    "electricity_tariff_kzt_per_kwh": 22.0,
    "co2_kg_per_kwh":                  0.62,

    # Regulatory norms (СП РК 4.01-101-2012)
    "min_pressure_residential_bar": 2.5,
    "max_pressure_residential_bar": 6.0,
}


def get_real_tariff(city_name: str) -> float:
    """Return water tariff in KZT per LITRE for the given city."""
    mapping = {
        "Алматы":    KAZAKHSTAN_REAL_DATA["tariff_almaty_kzt_per_m3"],
        "Астана":    KAZAKHSTAN_REAL_DATA["tariff_astana_kzt_per_m3"],
        "Туркестан": KAZAKHSTAN_REAL_DATA["tariff_turkestan_kzt_per_m3"],
    }
    rate_per_m3 = mapping.get(city_name, KAZAKHSTAN_REAL_DATA["tariff_almaty_kzt_per_m3"])
    return rate_per_m3 / 1000.0  # KZT per litre


def get_real_pipe_wear(city_name: str) -> float:
    """Return real pipeline wear percentage for the given city."""
    mapping = {
        "Алматы":    KAZAKHSTAN_REAL_DATA["wear_almaty_pct"],
        "Астана":    KAZAKHSTAN_REAL_DATA["wear_astana_pct"],
        "Туркестан": KAZAKHSTAN_REAL_DATA["wear_turkestan_pct"],
    }
    return mapping.get(city_name, 50.0)


def get_estimated_pipe_age(city_name: str) -> int:
    """Return estimated average pipe age (years) for the given city."""
    mapping = {
        "Алматы":    KAZAKHSTAN_REAL_DATA["age_almaty_years"],
        "Астана":    KAZAKHSTAN_REAL_DATA["age_astana_years"],
        "Туркестан": KAZAKHSTAN_REAL_DATA["age_turkestan_years"],
    }
    return mapping.get(city_name, 40)


# ═══════════════════════════════════════════════════════════════════════════
# BATTLEDIM DATASET LOADER
# ═══════════════════════════════════════════════════════════════════════════

BATTLEDIM_DATA_DIR = Path("data/battledim")

# Your public Google Drive folder ID
GDRIVE_FOLDER_ID = "1OL2xEGTKEA-eoaxRgd0n8vUEsGzj9Ngq"

# Expected file names (primary)
BATTLEDIM_FILES = {
    "scada_2018":  "2018_SCADA.xlsx",
    "scada_2019":  "2019_SCADA.xlsx",
    "leaks_2019":  "2019_Leaks.csv",
    "network_inp": "L-TOWN.inp",
}

# Alternative names to handle minor naming differences in Drive
FILENAME_ALTERNATIVES: Dict[str, list] = {
    "scada_2018":  ["2018_SCADA.xlsx", "2018_scada.xlsx", "SCADA_2018.xlsx",
                    "2018_SCADA.xls"],
    "scada_2019":  ["2019_SCADA.xlsx", "2019_scada.xlsx", "SCADA_2019.xlsx",
                    "2019_SCADA.xls"],
    "leaks_2019":  ["2019_Leaks.csv", "2019_leaks.csv", "Leaks_2019.csv",
                    "Leak_Labels.csv", "leak_labels.csv", "leaks.csv"],
    "network_inp": ["L-TOWN.inp", "l-town.inp", "LTOWN.inp",
                    "L_TOWN.inp", "network.inp", "ltown.inp"],
}


class BattLeDIMLoader:
    """
    Loader for the BattLeDIM 2020 benchmark dataset.

    Downloads from Google Drive folder:
      https://drive.google.com/drive/folders/1OL2xEGTKEA-eoaxRgd0n8vUEsGzj9Ngq

    L-Town network (Limassol, Cyprus):
      782 nodes | 909 pipes | 42.6 km | 23 real leaks (2019)

    DOI: 10.5281/zenodo.4017659
    """

    def __init__(self, data_dir: Path = BATTLEDIM_DATA_DIR):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ── File resolution ──────────────────────────────────────────────────

    def _resolve_path(self, key: str) -> Optional[Path]:
        """Find actual file path for a key, trying alternative names."""
        for name in FILENAME_ALTERNATIVES.get(key, [BATTLEDIM_FILES[key]]):
            p = self.data_dir / name
            if p.exists():
                return p
        return None

    def check_files_exist(self) -> Dict[str, bool]:
        """Return {file_key: exists} for each expected dataset file."""
        return {key: self._resolve_path(key) is not None for key in BATTLEDIM_FILES}

    def all_files_present(self) -> bool:
        return all(self.check_files_exist().values())

    # ── Download ─────────────────────────────────────────────────────────

    def download_dataset(self) -> Tuple[bool, str]:
        """
        Download the entire BattLeDIM folder from Google Drive via gdown.

        Uses gdown.download_folder() — downloads ALL files in the folder
        automatically without needing individual file IDs.

        Returns:
            (success: bool, message: str)
        """
        try:
            import gdown  # type: ignore
        except ImportError:
            return False, "❌ gdown not installed. Add 'gdown>=5.1.0' to requirements.txt"

        folder_url = f"https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}"
        logger.info("Downloading BattLeDIM from %s …", folder_url)

        try:
            # Download into a temp location, then merge into data_dir
            output_parent = self.data_dir.parent
            output_parent.mkdir(parents=True, exist_ok=True)

            downloaded = gdown.download_folder(
                url=folder_url,
                output=str(output_parent),
                quiet=False,
                use_cookies=False,
                remaining_ok=True,   # don't fail on files that can't be downloaded
            )

            # Merge any downloaded files into data_dir
            _merge_downloads(output_parent, self.data_dir)

            # Evaluate what we now have
            status  = self.check_files_exist()
            found   = [k for k, v in status.items() if v]
            missing = [k for k, v in status.items() if not v]

            present = [f.name for f in self.data_dir.iterdir() if f.is_file()]
            logger.info("Files in data_dir after download: %s", present)

            if not missing:
                return True, "✅ BattLeDIM dataset downloaded successfully!"
            elif found:
                return True, (
                    f"⚠️ Partial download — found: {found} | missing: {missing}. "
                    f"Files present: {present}"
                )
            else:
                return False, (
                    f"❌ Download completed but expected files not found. "
                    f"Files in folder: {present}. "
                    f"Expected: {list(BATTLEDIM_FILES.values())}. "
                    f"Try downloading manually from: {folder_url}"
                )

        except Exception as exc:
            logger.exception("Error during BattLeDIM folder download")
            return False, (
                f"❌ Download error: {exc}. "
                f"Download manually from: {folder_url} → place in 'data/battledim/'"
            )

    # ── Data loading ─────────────────────────────────────────────────────

    def load_scada_2018(self) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Load 2018 SCADA training data.

        Returns:
            {"pressures": DataFrame(index=timestamps, cols=sensor_IDs)} or None
        """
        path = self._resolve_path("scada_2018")
        if path is None:
            return None
        try:
            xl    = pd.ExcelFile(path)
            sheet = xl.sheet_names[0]
            df    = xl.parse(sheet, index_col=0, parse_dates=True)
            df    = df.apply(pd.to_numeric, errors="coerce")
            return {"pressures": df}
        except Exception as exc:
            logger.error("Could not load 2018 SCADA: %s", exc)
            return None

    def load_scada_2019(self) -> Optional[Dict[str, pd.DataFrame]]:
        """Load 2019 SCADA test data (contains real leak periods)."""
        path = self._resolve_path("scada_2019")
        if path is None:
            return None
        try:
            xl    = pd.ExcelFile(path)
            sheet = xl.sheet_names[0]
            df    = xl.parse(sheet, index_col=0, parse_dates=True)
            df    = df.apply(pd.to_numeric, errors="coerce")
            return {"pressures": df}
        except Exception as exc:
            logger.error("Could not load 2019 SCADA: %s", exc)
            return None

    def load_leaks_2019(self) -> Optional[pd.DataFrame]:
        """
        Load labelled 2019 leak events (23 real leaks with timestamps).

        Returns:
            DataFrame with leak metadata, or None.
        """
        path = self._resolve_path("leaks_2019")
        if path is None:
            return None
        try:
            return pd.read_csv(path)
        except Exception as exc:
            logger.error("Could not load 2019 leaks: %s", exc)
            return None

    def get_network_statistics(self) -> Dict[str, Any]:
        """
        Return L-Town network statistics.
        Parses INP file if present; falls back to published hard-coded values.
        """
        stats: Dict[str, Any] = {
            "n_junctions":     782,
            "n_pipes":         909,
            "total_length_km": 42.6,
            "n_leaks_2019":    23,
            "doi":    "10.5281/zenodo.4017659",
            "source": "BattLeDIM 2020 — L-Town, Limassol, Cyprus",
            "status": "HARDCODED",
        }

        inp_path = self._resolve_path("network_inp")
        if inp_path is not None:
            try:
                import wntr  # type: ignore
                wn = wntr.network.WaterNetworkModel(str(inp_path))
                stats["n_junctions"]     = wn.num_junctions
                stats["n_pipes"]         = wn.num_pipes
                total_m = sum(wn.get_link(p).length for p in wn.pipe_name_list)
                stats["total_length_km"] = round(total_m / 1000.0, 2)
                stats["status"]          = "LOADED"
            except Exception as exc:
                logger.warning("Could not parse INP with WNTR: %s", exc)

        return stats

    def get_pressure_timeseries(self,
                                year: int = 2018,
                                day: int = 1) -> Optional[pd.DataFrame]:
        """
        Return pressure data for a specific day.

        Args:
            year: 2018 (training) or 2019 (test)
            day:  Day of year (1–365)

        Returns:
            DataFrame with 288 rows (5-min interval) × n_sensor columns
        """
        fn     = self.load_scada_2018 if year == 2018 else self.load_scada_2019
        result = fn()

        if result is None or "pressures" not in result:
            return None

        df = result["pressures"]

        # DatetimeIndex path
        if hasattr(df.index, "dayofyear"):
            try:
                mask     = df.index.dayofyear == day
                day_data = df[mask]
                return day_data if len(day_data) > 0 else df.iloc[:288]
            except Exception:
                pass

        # Numeric index: 288 five-minute samples per day
        samples_per_day = 288
        start = (day - 1) * samples_per_day
        end   = start + samples_per_day
        return df.iloc[start:end] if start < len(df) else df.iloc[:samples_per_day]


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _merge_downloads(src_root: Path, dest: Path) -> None:
    """
    Walk src_root recursively and move any files whose names match
    known BattLeDIM filenames into dest.
    """
    import shutil

    known_names: set = set()
    for alts in FILENAME_ALTERNATIVES.values():
        known_names.update(a.lower() for a in alts)

    for item in src_root.rglob("*"):
        if item.is_file() and item.name.lower() in known_names:
            target = dest / item.name
            if not target.exists():
                try:
                    shutil.move(str(item), str(target))
                    logger.info("Moved %s → %s", item.name, target)
                except Exception as exc:
                    logger.warning("Could not move %s: %s", item.name, exc)


# ═══════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL SINGLETON & PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════

_loader_instance: Optional[BattLeDIMLoader] = None


def get_loader() -> BattLeDIMLoader:
    """Return (or create) the module-level BattLeDIMLoader singleton."""
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = BattLeDIMLoader(BATTLEDIM_DATA_DIR)
    return _loader_instance


def initialize_battledim(show_progress: bool = False) -> Tuple[bool, str]:
    """
    Check whether the BattLeDIM dataset is present; auto-download if not.

    Called once at Streamlit startup (init_session_state in app.py).

    Returns:
        (success: bool, message: str)
    """
    loader = get_loader()

    if loader.all_files_present():
        return True, "✅ BattLeDIM dataset already present"

    logger.info("BattLeDIM files missing — attempting download …")
    return loader.download_dataset()


# ═══════════════════════════════════════════════════════════════════════════
# QUICK SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("data_loader.py — self-test")
    print("=" * 60)

    for city in ["Алматы", "Астана", "Туркестан"]:
        print(
            f"{city}: tariff={get_real_tariff(city):.5f} ₸/л | "
            f"wear={get_real_pipe_wear(city)}% | "
            f"age={get_estimated_pipe_age(city)} лет"
        )

    loader = get_loader()
    print("\nFile status:", loader.check_files_exist())

    success, msg = initialize_battledim()
    print(f"Init: {success} — {msg}")
    print("Network stats:", loader.get_network_statistics())
