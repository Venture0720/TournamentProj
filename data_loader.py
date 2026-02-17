"""
Smart Shygyn PRO v3 — Data Loader
BattLeDIM dataset integration and Kazakhstan real data.

Download strategy (in order of priority):
  1. Zenodo REST API  → direct HTTP download, no auth needed, always works
  2. Google Drive     → fallback via gdown (may be rate-limited)

Zenodo record: https://zenodo.org/records/4017659
DOI: 10.5281/zenodo.4017659
"""

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import numpy as np
import requests

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
    return mapping.get(city_name, KAZAKHSTAN_REAL_DATA["tariff_almaty_kzt_per_m3"]) / 1000.0


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
# BATTLEDIM CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

BATTLEDIM_DATA_DIR  = Path("data/battledim")
ZENODO_RECORD_ID    = "4017659"
ZENODO_API_URL      = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
GDRIVE_FOLDER_ID    = "1OL2xEGTKEA-eoaxRgd0n8vUEsGzj9Ngq"

# Canonical file names we look for locally
BATTLEDIM_FILES = {
    "scada_2018":  "2018_SCADA.xlsx",
    "scada_2019":  "2019_SCADA.xlsx",
    "leaks_2019":  "2019_Leaks.csv",
    "network_inp": "L-TOWN.inp",
}

# Alternative names in case the dataset uses different capitalisation
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


# ═══════════════════════════════════════════════════════════════════════════
# BATTLEDIM LOADER
# ═══════════════════════════════════════════════════════════════════════════

class BattLeDIMLoader:
    """
    Loader for the BattLeDIM 2020 benchmark dataset.

    Primary download source: Zenodo REST API (direct HTTP, no auth).
    Fallback: Google Drive via gdown.

    L-Town network — Limassol, Cyprus:
      782 nodes | 909 pipes | 42.6 km | 23 real leaks (2019)

    DOI: 10.5281/zenodo.4017659
    """

    def __init__(self, data_dir: Path = BATTLEDIM_DATA_DIR):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ── File resolution ──────────────────────────────────────────────────

    def _resolve_path(self, key: str) -> Optional[Path]:
        """Return first existing path for dataset key, trying all alternatives."""
        for name in FILENAME_ALTERNATIVES.get(key, [BATTLEDIM_FILES[key]]):
            p = self.data_dir / name
            if p.exists() and p.stat().st_size > 500:   # ignore empty/corrupt files
                return p
        return None

    def check_files_exist(self) -> Dict[str, bool]:
        """Return {file_key: found} for each expected file."""
        return {key: self._resolve_path(key) is not None for key in BATTLEDIM_FILES}

    def all_files_present(self) -> bool:
        return all(self.check_files_exist().values())

    # ── Zenodo download ──────────────────────────────────────────────────

    def _get_zenodo_files(self) -> Optional[Dict[str, str]]:
        """
        Query Zenodo API and return {filename: download_url} dict.

        Zenodo API endpoint: GET /api/records/{record_id}
        Returns JSON with `files` list, each entry has `key` and `links.self`.

        Returns None if API call fails.
        """
        try:
            resp = requests.get(ZENODO_API_URL, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            # New Zenodo API structure
            files_info: Dict[str, str] = {}

            # Try "files" key (newer API)
            if "files" in data:
                for f in data["files"]:
                    fname = f.get("key") or f.get("filename", "")
                    url   = (f.get("links", {}).get("self") or
                             f.get("links", {}).get("download") or
                             f.get("url", ""))
                    if fname and url:
                        files_info[fname] = url

            # Try legacy "links" > "files" structure
            if not files_info and "links" in data and "files" in data:
                for f in data.get("files", []):
                    fname = f.get("key", "")
                    url   = f.get("links", {}).get("self", "")
                    if fname and url:
                        files_info[fname] = url

            # Construct direct content URL as last resort
            if not files_info:
                # Zenodo always supports this URL pattern for public records
                for key, fname in BATTLEDIM_FILES.items():
                    url = (f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
                           f"/files/{fname}/content")
                    files_info[fname] = url

            logger.info("Zenodo API returned %d file(s)", len(files_info))
            return files_info

        except Exception as exc:
            logger.warning("Zenodo API call failed: %s", exc)
            # Return hardcoded URLs — Zenodo always honours this pattern
            return {
                fname: (f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
                        f"/files/{fname}/content")
                for fname in BATTLEDIM_FILES.values()
            }

    def _download_file(self, url: str, dest: Path) -> bool:
        """
        Stream-download a single file from `url` to `dest`.

        Returns True on success.
        """
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                content_len = int(r.headers.get("content-length", 0))
                # Reject HTML error pages (< 10 KB for binary files)
                if content_len and content_len < 1024:
                    logger.warning("Suspiciously small file (%d B) for %s", content_len, url)
                    return False
                with open(dest, "wb") as f:
                    shutil.copyfileobj(r.raw, f)

            # Sanity-check
            if dest.exists() and dest.stat().st_size > 500:
                logger.info("Downloaded %s (%.1f KB)", dest.name,
                            dest.stat().st_size / 1024)
                return True
            else:
                dest.unlink(missing_ok=True)
                return False

        except Exception as exc:
            logger.warning("Failed to download %s: %s", url, exc)
            dest.unlink(missing_ok=True)
            return False

    def download_via_zenodo(self) -> Tuple[bool, str]:
        """
        Download BattLeDIM files directly from Zenodo (no gdown needed).

        Uses the public Zenodo REST API to get direct download URLs,
        then streams each file via requests.

        Returns:
            (success: bool, message: str)
        """
        zenodo_files = self._get_zenodo_files()
        if not zenodo_files:
            return False, "❌ Could not retrieve file list from Zenodo API"

        downloaded, failed = [], []

        for key, canonical_name in BATTLEDIM_FILES.items():
            if self._resolve_path(key) is not None:
                downloaded.append(key)
                continue  # already have it

            dest = self.data_dir / canonical_name

            # Find the matching URL (exact name or fuzzy match)
            url = zenodo_files.get(canonical_name)
            if not url:
                # Try alternatives
                for alt in FILENAME_ALTERNATIVES.get(key, []):
                    url = zenodo_files.get(alt)
                    if url:
                        break

            if not url:
                # Construct the URL directly — works for all public Zenodo records
                url = (f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
                       f"/files/{canonical_name}/content")

            logger.info("Downloading %s from %s", canonical_name, url)
            ok = self._download_file(url, dest)

            if ok:
                downloaded.append(key)
            else:
                failed.append(canonical_name)

        if not failed:
            return True, f"✅ BattLeDIM downloaded successfully from Zenodo ({len(downloaded)} files)"
        elif downloaded:
            return True, (
                f"⚠️ Partial download — got {len(downloaded)} file(s), "
                f"failed: {failed}"
            )
        else:
            return False, (
                f"❌ Zenodo download failed for all files: {failed}. "
                f"Download manually: https://zenodo.org/records/{ZENODO_RECORD_ID}"
            )

    def download_via_gdrive(self) -> Tuple[bool, str]:
        """
        Fallback: download from Google Drive via gdown.

        gdown.download_folder may be rate-limited by Google.
        """
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

            status  = self.check_files_exist()
            found   = [k for k, v in status.items() if v]
            missing = [k for k, v in status.items() if not v]

            if not missing:
                return True, "✅ Downloaded from Google Drive"
            elif found:
                return True, f"⚠️ Partial (Google Drive): found={found}, missing={missing}"
            else:
                return False, "❌ Google Drive download returned no usable files"

        except Exception as exc:
            return False, f"❌ Google Drive error: {exc}"

    def download_dataset(self) -> Tuple[bool, str]:
        """
        Master download method.

        Tries Zenodo first (reliable), then Google Drive as fallback.
        """
        # 1. Try Zenodo (primary — always works for public records)
        ok, msg = self.download_via_zenodo()
        if ok and self.all_files_present():
            return ok, msg

        logger.warning("Zenodo download incomplete (%s). Trying Google Drive…", msg)

        # 2. Try Google Drive fallback
        ok2, msg2 = self.download_via_gdrive()
        if ok2 and self.all_files_present():
            return ok2, msg2

        # Report combined failure
        return False, (
            f"❌ Both download sources failed.\n"
            f"  Zenodo: {msg}\n"
            f"  Google Drive: {msg2}\n"
            f"Please download manually from "
            f"https://zenodo.org/records/{ZENODO_RECORD_ID} "
            f"and place files in 'data/battledim/'"
        )

    # ── Data loading ─────────────────────────────────────────────────────

    def load_scada_2018(self) -> Optional[Dict[str, pd.DataFrame]]:
        """Load 2018 SCADA training data (pressure timeseries)."""
        path = self._resolve_path("scada_2018")
        if path is None:
            return None
        try:
            xl    = pd.ExcelFile(path)
            df    = xl.parse(xl.sheet_names[0], index_col=0, parse_dates=True)
            return {"pressures": df.apply(pd.to_numeric, errors="coerce")}
        except Exception as exc:
            logger.error("Cannot load 2018 SCADA: %s", exc)
            return None

    def load_scada_2019(self) -> Optional[Dict[str, pd.DataFrame]]:
        """Load 2019 SCADA test data (contains real leak periods)."""
        path = self._resolve_path("scada_2019")
        if path is None:
            return None
        try:
            xl    = pd.ExcelFile(path)
            df    = xl.parse(xl.sheet_names[0], index_col=0, parse_dates=True)
            return {"pressures": df.apply(pd.to_numeric, errors="coerce")}
        except Exception as exc:
            logger.error("Cannot load 2019 SCADA: %s", exc)
            return None

    def load_leaks_2019(self) -> Optional[pd.DataFrame]:
        """Load labelled 2019 leak events (23 real leaks)."""
        path = self._resolve_path("leaks_2019")
        if path is None:
            return None
        try:
            return pd.read_csv(path)
        except Exception as exc:
            logger.error("Cannot load 2019 leaks: %s", exc)
            return None

    def get_network_statistics(self) -> Dict[str, Any]:
        """
        Return L-Town network statistics.
        Parses INP if available; falls back to published hard-coded values.
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
        inp = self._resolve_path("network_inp")
        if inp:
            try:
                import wntr  # type: ignore
                wn = wntr.network.WaterNetworkModel(str(inp))
                stats["n_junctions"]     = wn.num_junctions
                stats["n_pipes"]         = wn.num_pipes
                total_m = sum(wn.get_link(p).length for p in wn.pipe_name_list)
                stats["total_length_km"] = round(total_m / 1000.0, 2)
                stats["status"]          = "LOADED"
            except Exception as exc:
                logger.warning("Cannot parse INP: %s", exc)
        return stats

    def get_pressure_timeseries(self,
                                year: int = 2018,
                                day: int = 1) -> Optional[pd.DataFrame]:
        """Return pressure data for one day (288 rows × n_sensor cols)."""
        fn     = self.load_scada_2018 if year == 2018 else self.load_scada_2019
        result = fn()
        if result is None or "pressures" not in result:
            return None

        df = result["pressures"]

        if hasattr(df.index, "dayofyear"):
            try:
                mask = df.index.dayofyear == day
                day_data = df[mask]
                return day_data if len(day_data) > 0 else df.iloc[:288]
            except Exception:
                pass

        samples_per_day = 288
        start = (day - 1) * samples_per_day
        end   = start + samples_per_day
        return df.iloc[start:end] if start < len(df) else df.iloc[:samples_per_day]


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _merge_downloads(src_root: Path, dest: Path) -> None:
    """Move any BattLeDIM files from src_root subtree into dest."""
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
    """Return (or create) the module-level BattLeDIMLoader singleton."""
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = BattLeDIMLoader(BATTLEDIM_DATA_DIR)
    return _loader_instance


def initialize_battledim(show_progress: bool = False) -> Tuple[bool, str]:
    """
    Check whether BattLeDIM dataset is present; auto-download from Zenodo if not.

    Called once at Streamlit startup (init_session_state in app.py).
    """
    loader = get_loader()
    if loader.all_files_present():
        return True, "✅ BattLeDIM dataset already present"
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
