"""
Smart Shygyn PRO v3 — Data Loader
BattLeDIM dataset integration and Kazakhstan real data.

Actual filenames on Zenodo 4017659 (official BattLeDIM):
  "2018 SCADA.xlsx"   ← space, not underscore
  "2019 SCADA.xlsx"
  "L-Town.inp"
  "Leak_Labels.xlsx"  ← released separately after competition

The leaks file is OPTIONAL — 3 SCADA + INP are sufficient for the app.
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
# BATTLEDIM CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

BATTLEDIM_DATA_DIR = Path("data/battledim")
ZENODO_RECORD_ID   = "4017659"
GDRIVE_FOLDER_ID   = "1OL2xEGTKEA-eoaxRgd0n8vUEsGzj9Ngq"

# ── REQUIRED files (3 of 4 — leaks is optional) ──────────────────────────
BATTLEDIM_REQUIRED = {
    "scada_2018":  "2018 SCADA.xlsx",    # space in official Zenodo filename!
    "scada_2019":  "2019 SCADA.xlsx",
    "network_inp": "L-Town.inp",
}

# ── OPTIONAL: leak labels (released after competition, not always on Zenodo)
BATTLEDIM_OPTIONAL = {
    "leaks_2019": "Leak_Labels.xlsx",
}

BATTLEDIM_FILES = {**BATTLEDIM_REQUIRED, **BATTLEDIM_OPTIONAL}

# All acceptable name variants per key (covers different uploads/mirrors)
FILENAME_ALTERNATIVES: Dict[str, List[str]] = {
    "scada_2018": [
        "2018 SCADA.xlsx",     # official Zenodo name (space)
        "2018_SCADA.xlsx",     # underscore variant
        "2018 SCADA.xls",
        "2018_SCADA.xls",
        "SCADA_2018.xlsx",
    ],
    "scada_2019": [
        "2019 SCADA.xlsx",     # official Zenodo name (space)
        "2019_SCADA.xlsx",
        "2019 SCADA.xls",
        "2019_SCADA.xls",
        "SCADA_2019.xlsx",
    ],
    "network_inp": [
        "L-Town.inp",
        "l-town.inp",
        "L_Town.inp",
        "LTOWN.inp",
        "ltown.inp",
        "network.inp",
    ],
    "leaks_2019": [
        "Leak_Labels.xlsx",
        "Leak_Labels.csv",
        "2019_Leaks.csv",
        "2019 Leaks.csv",
        "2019_Leaks.xlsx",
        "leaks.csv",
        "leak_labels.csv",
        "GroundTruth.csv",
        "GroundTruth.xlsx",
    ],
}

# Zenodo direct-download URL pattern (works for all public records)
def _zenodo_url(filename: str) -> str:
    return (
        f"https://zenodo.org/records/{ZENODO_RECORD_ID}"
        f"/files/{requests.utils.quote(filename)}?download=1"
    )


# ═══════════════════════════════════════════════════════════════════════════
# LOADER
# ═══════════════════════════════════════════════════════════════════════════

class BattLeDIMLoader:
    """
    Loader for the BattLeDIM 2020 benchmark dataset.

    Primary source : Zenodo 4017659 (direct HTTP, no auth)
    Fallback       : Google Drive via gdown

    L-Town, Limassol, Cyprus — 782 nodes | 909 pipes | 42.6 km | 23 leaks
    DOI: 10.5281/zenodo.4017659
    """

    def __init__(self, data_dir: Path = BATTLEDIM_DATA_DIR):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ── File resolution ──────────────────────────────────────────────────

    def _resolve_path(self, key: str) -> Optional[Path]:
        """Return first existing, non-empty path for a dataset key."""
        for name in FILENAME_ALTERNATIVES.get(key, [BATTLEDIM_FILES.get(key, "")]):
            p = self.data_dir / name
            if p.exists() and p.stat().st_size > 200:
                return p
        return None

    def check_files_exist(self) -> Dict[str, bool]:
        return {key: self._resolve_path(key) is not None for key in BATTLEDIM_FILES}

    def required_files_present(self) -> bool:
        """True when all REQUIRED files are present (leaks is optional)."""
        return all(
            self._resolve_path(k) is not None for k in BATTLEDIM_REQUIRED
        )

    def all_files_present(self) -> bool:
        """True when ALL files (including optional leaks) are present."""
        return all(self._resolve_path(k) is not None for k in BATTLEDIM_FILES)

    # ── Single-file download ─────────────────────────────────────────────

    def _download_file(self, url: str, dest: Path) -> bool:
        """Stream-download url → dest. Returns True on success."""
        try:
            headers = {"User-Agent": "SmartShygyn/3.0"}
            with requests.get(url, stream=True, timeout=120, headers=headers) as r:
                r.raise_for_status()
                # Reject tiny HTML error pages
                ct = r.headers.get("Content-Type", "")
                if "text/html" in ct:
                    logger.warning("Got HTML instead of file from %s", url)
                    return False
                with open(dest, "wb") as f:
                    shutil.copyfileobj(r.raw, f)

            if dest.exists() and dest.stat().st_size > 200:
                logger.info("✓ %s (%.1f KB)", dest.name, dest.stat().st_size / 1024)
                return True

            dest.unlink(missing_ok=True)
            return False

        except Exception as exc:
            logger.warning("Download failed %s: %s", url, exc)
            dest.unlink(missing_ok=True)
            return False

    # ── Zenodo download ──────────────────────────────────────────────────

    def download_via_zenodo(self) -> Tuple[bool, str]:
        """
        Download BattLeDIM files from Zenodo using their direct-download URL.

        URL pattern:  https://zenodo.org/records/{id}/files/{name}?download=1
        No API key needed — works for all CC-licensed public records.
        """
        downloaded, failed, skipped = [], [], []

        for key, primary_name in BATTLEDIM_FILES.items():
            is_optional = key in BATTLEDIM_OPTIONAL

            if self._resolve_path(key) is not None:
                downloaded.append(key)
                continue

            dest = self.data_dir / primary_name
            success = False

            # Try all alternative filenames on Zenodo
            for alt_name in FILENAME_ALTERNATIVES.get(key, [primary_name]):
                url = _zenodo_url(alt_name)
                logger.info("Trying: %s", url)
                if self._download_file(url, self.data_dir / alt_name):
                    success = True
                    downloaded.append(key)
                    break

            if not success:
                if is_optional:
                    skipped.append(key)
                    logger.info("Optional file %s not found on Zenodo — skipping", key)
                else:
                    failed.append(key)
                    logger.warning("Required file %s failed to download", key)

        required_ok = all(k not in failed for k in BATTLEDIM_REQUIRED)

        if failed:
            return False, (
                f"❌ Required files missing after Zenodo download: {failed}"
            )
        elif skipped:
            return True, (
                f"✅ Downloaded {len(downloaded)} file(s) from Zenodo "
                f"(optional files not found: {skipped} — app works without them)"
            )
        else:
            return True, f"✅ All {len(downloaded)} BattLeDIM files downloaded from Zenodo!"

    # ── Google Drive fallback ────────────────────────────────────────────

    def download_via_gdrive(self) -> Tuple[bool, str]:
        """Fallback: download folder from Google Drive via gdown."""
        try:
            import gdown  # type: ignore
        except ImportError:
            return False, "gdown not installed"

        folder_url = f"https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}"
        try:
            output_parent = self.data_dir.parent
            gdown.download_folder(
                url=folder_url,
                output=str(output_parent),
                quiet=False,
                use_cookies=False,
                remaining_ok=True,
            )
            _merge_downloads(output_parent, self.data_dir)

            if self.required_files_present():
                return True, "✅ Downloaded from Google Drive"
            found = [k for k, v in self.check_files_exist().items() if v]
            return bool(found), f"Partial Google Drive download: {found}"

        except Exception as exc:
            return False, f"Google Drive error: {exc}"

    # ── Master download ──────────────────────────────────────────────────

    def download_dataset(self) -> Tuple[bool, str]:
        """Try Zenodo first, then Google Drive as fallback."""
        ok, msg = self.download_via_zenodo()
        if ok:
            return ok, msg

        logger.warning("Zenodo incomplete (%s). Trying Google Drive …", msg)
        ok2, msg2 = self.download_via_gdrive()
        if ok2:
            return ok2, msg2

        return False, (
            f"❌ Both sources failed.\n"
            f"  Zenodo: {msg}\n"
            f"  Google Drive: {msg2}\n"
            f"Manual download: https://zenodo.org/records/{ZENODO_RECORD_ID}\n"
            f"Place files in: data/battledim/"
        )

    # ── Data loading ─────────────────────────────────────────────────────

    def load_scada_2018(self) -> Optional[Dict[str, pd.DataFrame]]:
        """Load 2018 SCADA training data."""
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
        """Load 2019 SCADA test data."""
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
        """Load 2019 leak labels (optional — may not be on Zenodo)."""
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
        """Network stats — parses INP if available, else uses published values."""
        stats: Dict[str, Any] = {
            "n_junctions": 782, "n_pipes": 909,
            "total_length_km": 42.6, "n_leaks_2019": 23,
            "doi": "10.5281/zenodo.4017659",
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
                        sum(wn.get_link(p).length for p in wn.pipe_name_list) / 1000, 2
                    ),
                    "status": "LOADED",
                })
            except Exception as exc:
                logger.warning("Cannot parse INP: %s", exc)
        return stats

    def get_pressure_timeseries(self, year: int = 2018, day: int = 1) -> Optional[pd.DataFrame]:
        """Return pressure data for one day (288 rows × n_sensor cols)."""
        fn     = self.load_scada_2018 if year == 2018 else self.load_scada_2019
        result = fn()
        if result is None:
            return None
        df = result["pressures"]

        if hasattr(df.index, "dayofyear"):
            try:
                mask = df.index.dayofyear == day
                sliced = df[mask]
                return sliced if len(sliced) > 0 else df.iloc[:288]
            except Exception:
                pass

        samp = 288
        start = (day - 1) * samp
        return df.iloc[start: start + samp] if start < len(df) else df.iloc[:samp]


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _merge_downloads(src_root: Path, dest: Path) -> None:
    """Move BattLeDIM files from gdown output subtree into dest."""
    known: set = set()
    for alts in FILENAME_ALTERNATIVES.values():
        known.update(a.lower() for a in alts)

    for item in src_root.rglob("*"):
        if item.is_file() and item.name.lower() in known:
            target = dest / item.name
            if not target.exists():
                try:
                    shutil.move(str(item), str(target))
                    logger.info("Moved %s → %s", item.name, target)
                except Exception as exc:
                    logger.warning("Could not move %s: %s", item.name, exc)


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
    Check BattLeDIM presence; auto-download from Zenodo if needed.
    Called once at Streamlit startup.
    """
    loader = get_loader()
    # Success if at minimum the 3 required files exist
    if loader.required_files_present():
        status = loader.check_files_exist()
        leaks_ok = status.get("leaks_2019", False)
        if leaks_ok:
            return True, "✅ BattLeDIM dataset fully present"
        return True, "✅ BattLeDIM dataset present (leak labels optional — not found)"
    logger.info("BattLeDIM files missing — downloading from Zenodo …")
    return loader.download_dataset()


# ═══════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("data_loader.py — self-test")
    print("=" * 60)

    for city in ["Алматы", "Астана", "Туркестан"]:
        print(f"{city}: {get_real_tariff(city):.5f} ₸/л | "
              f"{get_real_pipe_wear(city)}% | {get_estimated_pipe_age(city)} лет")

    loader = get_loader()
    print("\nFile status:", loader.check_files_exist())

    ok, msg = initialize_battledim()
    print(f"Init: {ok} — {msg}")
    print("Network stats:", loader.get_network_statistics())
