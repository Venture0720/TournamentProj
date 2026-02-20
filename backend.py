"""
Smart Shygyn PRO v3 — BACKEND ENGINE
FIXED v3:
- EPANET result validation: physically impossible pressures trigger fallback
- Minimum pump head enforcement to prevent non-convergence
- run_simulation returns zero-results on invalid output (not just on exception)
- Leak area sanity clamped to prevent EPANET divergence
- build_network: reservoir head boosted automatically if too low for elevation
- MNF anomaly: fixed sign logic — anomaly only when flow ABOVE baseline (leak = excess flow)
- Failure probability: guard against zero-pressure fallback showing fake 50% everywhere
"""

import gc
import logging
import numpy as np
import pandas as pd
import wntr
import networkx as nx
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("smart_shygyn.backend")


# ═══════════════════════════════════════════════════════════════════════════
# PART 1A: GEOGRAPHIC & CITY PHYSICS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CityConfig:
    name: str
    lat: float
    lng: float
    zoom: int
    elev_min: float
    elev_max: float
    elev_direction: str
    ground_temp_celsius: float
    base_burst_multiplier: float
    water_stress_index: float
    description: str


class CityManager:
    CITIES: Dict[str, CityConfig] = {
        "Алматы": CityConfig(
            name="Алматы", lat=43.2220, lng=76.8512, zoom=15,
            elev_min=600.0, elev_max=1000.0, elev_direction="S→N",
            ground_temp_celsius=12.0, base_burst_multiplier=1.0,
            water_stress_index=0.35,
            description="High elevation gradient (South→North, 600-1000m). Moderate water stress."
        ),
        "Астана": CityConfig(
            name="Астана", lat=51.1605, lng=71.4704, zoom=15,
            elev_min=340.0, elev_max=360.0, elev_direction="Flat",
            ground_temp_celsius=-2.5, base_burst_multiplier=1.0,
            water_stress_index=0.55,
            description="Flat steppe. Extreme freeze-thaw pipe burst risk. High water stress."
        ),
        "Туркестан": CityConfig(
            name="Туркестан", lat=43.3016, lng=68.2730, zoom=15,
            elev_min=200.0, elev_max=280.0, elev_direction="SW→NE",
            ground_temp_celsius=22.0, base_burst_multiplier=0.8,
            water_stress_index=0.82,
            description="Arid climate. Extreme evaporation and water scarcity."
        ),
    }

    def __init__(self, city_name: str, season_temp_celsius: float = 10.0):
        if city_name not in self.CITIES:
            raise ValueError(f"Unknown city: {city_name}. Choose from {list(self.CITIES.keys())}")
        self.config = self.CITIES[city_name]
        self.season_temp = season_temp_celsius
        self.burst_multiplier = self._calculate_burst_multiplier()

    def _calculate_burst_multiplier(self) -> float:
        if self.config.name == "Астана":
            delta = abs(self.season_temp - self.config.ground_temp_celsius)
            return 1.0 + 0.05 * min(delta, 20.0)
        return self.config.base_burst_multiplier

    def node_elevation(self, i: int, j: int, grid_size: int = 4) -> float:
        lo, hi = self.config.elev_min, self.config.elev_max
        direction = self.config.elev_direction
        if direction == "S→N":
            frac = j / max(1, grid_size - 1)
        elif direction == "SW→NE":
            frac = (i + j) / max(1, 2 * (grid_size - 1))
        else:
            frac = 0.5
        return hi - frac * (hi - lo)

    def grid_to_latlon(self, i: int, j: int,
                       lat_step: float = 0.0009) -> Tuple[float, float]:
        lat = self.config.lat + j * lat_step
        lng_step = lat_step / math.cos(math.radians(self.config.lat))
        lng = self.config.lng + i * lng_step
        return lat, lng

    def get_water_stress_factor(self) -> float:
        return self.config.water_stress_index

    def get_evaporation_rate(self) -> float:
        rate = max(0.0, min(0.95, (self.config.ground_temp_celsius - 10.0) / 50.0))
        return rate

    def min_required_pump_head(self, grid_size: int = 4) -> float:
        """
        Minimum pump head so EPANET can converge.
        = elev_max + 40m service pressure margin.
        """
        return self.config.elev_max + 40.0


# ═══════════════════════════════════════════════════════════════════════════
# PART 1B: HYDRAULIC PHYSICS
# ═══════════════════════════════════════════════════════════════════════════

class HydraulicPhysics:
    HAZEN_WILLIAMS_BASE: Dict[str, float] = {
        "Пластик (ПНД)": 150.0,
        "Сталь": 140.0,
        "Чугун": 100.0,
    }
    DECAY_RATE: Dict[str, float] = {
        "Пластик (ПНД)": 0.10,
        "Сталь": 0.30,
        "Чугун": 0.50,
    }

    @staticmethod
    def temperature_correction_factor(temp: float) -> float:
        if temp >= 10.0:
            return 1.0
        degrees_below = 10.0 - temp
        reduction_pct = (degrees_below / 5.0) * 0.01
        return max(0.0, 1.0 - reduction_pct)

    @staticmethod
    def hazen_williams_roughness(material: str, age_years: float,
                                  temp: float = 10.0) -> float:
        base_c = HydraulicPhysics.HAZEN_WILLIAMS_BASE.get(material, 130.0)
        decay  = HydraulicPhysics.DECAY_RATE.get(material, 0.30)
        age_degraded_c = base_c - (decay * age_years)
        temp_factor    = HydraulicPhysics.temperature_correction_factor(temp)
        current_c      = age_degraded_c * temp_factor
        return max(40.0, current_c)

    @staticmethod
    def degradation_percentage(material: str, age_years: float,
                                temp: float = 10.0) -> float:
        base_c    = HydraulicPhysics.HAZEN_WILLIAMS_BASE.get(material, 130.0)
        current_c = HydraulicPhysics.hazen_williams_roughness(material, age_years, temp)
        return max(0.0, min(100.0, (1.0 - current_c / base_c) * 100.0))

    @staticmethod
    def torricelli_leak_flow(area_m2: float, head_m: float,
                             discharge_coeff: float = 0.61) -> float:
        if head_m <= 0:
            return 0.0
        g = 9.81
        return discharge_coeff * area_m2 * math.sqrt(2.0 * g * head_m)

    @staticmethod
    def emitter_coefficient_from_area(area_cm2: float,
                                      pressure_bar: float = 3.0,
                                      exponent: float = 0.5) -> float:
        area_m2  = area_cm2 / 10000.0
        head_m   = pressure_bar * 10.197
        q_m3s    = HydraulicPhysics.torricelli_leak_flow(area_m2, head_m)
        q_lps    = q_m3s * 1000.0
        K = q_lps / (head_m ** exponent) if head_m > 0 else 0.0
        return K

    @staticmethod
    def create_demand_pattern(hours: int = 24) -> List[float]:
        pattern = []
        for h in range(hours):
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

    @staticmethod
    def failure_probability(pressure_bar: float,
                            degradation_pct: float,
                            burst_multiplier: float = 1.0) -> float:
        P_MAX = 5.0
        ALPHA = 0.5
        BETA  = 2.0
        GAMMA = 1.5
        pressure_factor    = (1.0 - min(max(pressure_bar, 0.0), P_MAX) / P_MAX) ** BETA
        degradation_factor = (degradation_pct / 100.0) ** GAMMA
        prob = ALPHA * pressure_factor * degradation_factor * burst_multiplier
        return min(100.0, prob * 100.0)


# ═══════════════════════════════════════════════════════════════════════════
# PART 1C: HYDRAULIC ENGINE
# ═══════════════════════════════════════════════════════════════════════════

_MIN_PRESSURE_BAR  = -5.0
_MAX_PRESSURE_BAR  = 150.0
_MAX_LEAK_AREA_CM2 = 5.0


def _make_zero_results(wn: wntr.network.WaterNetworkModel,
                       n_hours: int = 24) -> Dict[str, pd.DataFrame]:
    time_index = pd.RangeIndex(n_hours, name="time")
    node_names = wn.node_name_list
    link_names = wn.link_name_list
    return {
        "pressure": pd.DataFrame(0.0, index=time_index, columns=node_names),
        "flow":     pd.DataFrame(0.0, index=time_index, columns=link_names),
        "age":      pd.DataFrame(0.0, index=time_index, columns=node_names),
    }


def _validate_results(results: Dict[str, pd.DataFrame]) -> bool:
    """
    Returns True only if EPANET output is physically reasonable.
    Catches silent divergence where EPANET exits without exception
    but produces nonsense values (e.g. -75 bar).
    """
    pressure = results["pressure"]
    if pressure.empty:
        return False
    p_min = pressure.min().min()
    p_max = pressure.max().max()
    if p_min < _MIN_PRESSURE_BAR:
        logger.warning("EPANET: min pressure %.2f bar < %.1f — silent divergence",
                       p_min, _MIN_PRESSURE_BAR)
        return False
    if p_max > _MAX_PRESSURE_BAR:
        logger.warning("EPANET: max pressure %.2f bar > %.1f — silent divergence",
                       p_max, _MAX_PRESSURE_BAR)
        return False
    return True


class HydraulicEngine:
    def __init__(self, city_manager: CityManager):
        self.city    = city_manager
        self.physics = HydraulicPhysics()

    def _safe_pump_head(self, pump_head_m: float) -> float:
        min_head = self.city.min_required_pump_head()
        if pump_head_m < min_head:
            logger.info(
                "Pump head %.1fm too low for %s (min %.1fm) — auto-boosting.",
                pump_head_m, self.city.config.name, min_head
            )
            return min_head
        return pump_head_m

    def build_network(self,
                      grid_size: int = 4,
                      material: str = "Пластик (ПНД)",
                      pipe_age: float = 15.0,
                      pipe_length_m: float = 100.0,
                      pipe_diameter_m: float = 0.2,
                      pump_head_m: float = 40.0,
                      smart_pump: bool = False,
                      leak_node: Optional[str] = None,
                      leak_area_cm2: float = 0.8,
                      leak_start_hour: float = 12.0,
                      contingency_pipe: Optional[str] = None) -> wntr.network.WaterNetworkModel:

        pump_head_m   = self._safe_pump_head(pump_head_m)
        leak_area_cm2 = min(leak_area_cm2, _MAX_LEAK_AREA_CM2)

        wn = wntr.network.WaterNetworkModel()
        roughness = self.physics.hazen_williams_roughness(
            material, pipe_age, temp=self.city.season_temp
        )
        demand_pattern = self.physics.create_demand_pattern()
        wn.add_pattern("demand_pattern", demand_pattern)

        for i in range(grid_size):
            for j in range(grid_size):
                node_name = f"N_{i}_{j}"
                elevation = self.city.node_elevation(i, j, grid_size)
                wn.add_junction(node_name, base_demand=0.001,
                                elevation=elevation,
                                demand_pattern="demand_pattern")
                wn.get_node(node_name).coordinates = (i * pipe_length_m,
                                                       j * pipe_length_m)
                if i > 0:
                    wn.add_pipe(f"PH_{i}_{j}", f"N_{i-1}_{j}", node_name,
                                length=pipe_length_m, diameter=pipe_diameter_m,
                                roughness=roughness)
                if j > 0:
                    wn.add_pipe(f"PV_{i}_{j}", f"N_{i}_{j-1}", node_name,
                                length=pipe_length_m, diameter=pipe_diameter_m,
                                roughness=roughness)

        wn.add_reservoir("Res", base_head=pump_head_m)
        wn.get_node("Res").coordinates = (-pipe_length_m, -pipe_length_m)
        wn.add_pipe("P_Main", "Res", "N_0_0",
                    length=pipe_length_m, diameter=pipe_diameter_m * 2.0,
                    roughness=roughness)

        if smart_pump:
            night_head = pump_head_m * 0.70
            wn.add_pattern("pump_pattern", [
                night_head / pump_head_m if (h >= 23 or h < 6) else 1.0
                for h in range(24)
            ])
            res = wn.get_node("Res")
            res.head_pattern_name = "pump_pattern"
            res.base_head = pump_head_m

        if leak_node and leak_node in wn.node_name_list:
            node = wn.get_node(leak_node)
            effective_pressure_bar = max(
                (pump_head_m - self.city.config.elev_max) * 0.098, 0.5
            )
            emitter_k = self.physics.emitter_coefficient_from_area(
                leak_area_cm2,
                pressure_bar=effective_pressure_bar,
                exponent=0.5
            )
            node.emitter_coefficient = emitter_k

        if contingency_pipe and contingency_pipe in wn.link_name_list:
            wn.remove_link(contingency_pipe)

        wn.options.time.duration           = 24 * 3600
        wn.options.time.hydraulic_timestep = 3600
        wn.options.time.report_timestep    = 3600
        wn.options.quality.parameter       = "AGE"
        return wn

    def run_simulation(self, wn: wntr.network.WaterNetworkModel,
                       sampling_rate_hz: int = 1) -> Dict[str, pd.DataFrame]:
        wn.options.time.report_timestep = int(3600 / sampling_rate_hz)
        n_hours = int(wn.options.time.duration / wn.options.time.report_timestep)

        try:
            sim = wntr.sim.EpanetSimulator(wn)
            raw = sim.run_sim()

            results = {
                "pressure": raw.node["pressure"] * 0.1,
                "flow":     raw.link["flowrate"] * 1000.0,
                "age":      raw.node["quality"] / 3600.0,
            }

            if not _validate_results(results):
                logger.error("EPANET silent divergence — using zero fallback.")
                return _make_zero_results(wn, n_hours=max(n_hours, 24))

            return results

        except Exception as exc:
            logger.error("EPANET exception — zero fallback. %s: %s",
                         type(exc).__name__, exc)
            return _make_zero_results(wn, n_hours=max(n_hours, 24))

    def apply_signal_smoothing(self, series: pd.Series, window: int = 3) -> pd.Series:
        return series.rolling(window, center=True, min_periods=1).mean()

    def add_sensor_noise(self, series: pd.Series, noise_std: float = 0.04) -> pd.Series:
        noise = np.random.normal(0, noise_std, len(series))
        return series + noise


# ═══════════════════════════════════════════════════════════════════════════
# PART 1D: LEAK DETECTION
# ═══════════════════════════════════════════════════════════════════════════

class LeakDetectionAnalytics:
    SENSOR_COVERAGE = 0.30

    @staticmethod
    def place_sensors(node_list: List[str], seed: int = 42,
                      coverage: float = 0.30) -> List[str]:
        rng        = np.random.default_rng(seed)
        candidates = [n for n in node_list if n != "Res"]
        n_sensors  = max(1, int(len(candidates) * coverage))
        return list(rng.choice(candidates, size=n_sensors, replace=False))

    @staticmethod
    def build_healthy_baseline(engine: HydraulicEngine,
                                material: str, pipe_age: float,
                                pump_head_m: float) -> Dict[str, float]:
        wn_healthy = engine.build_network(
            material=material, pipe_age=pipe_age,
            pump_head_m=pump_head_m, leak_node=None
        )
        results = engine.run_simulation(wn_healthy, sampling_rate_hz=1)
        baseline = {}
        for node in wn_healthy.node_name_list:
            if node != "Res" and node in results["pressure"].columns:
                val = float(results["pressure"][node].mean())
                baseline[node] = val if val > 0 else 0.0
        return baseline

    @staticmethod
    def residual_matrix_localization(
            healthy_baseline: Dict[str, float],
            observed_pressures: Dict[str, float],
            sensor_nodes: List[str],
            wn: wntr.network.WaterNetworkModel
    ) -> Tuple[str, Dict[str, float], float]:
        graph = wn.get_graph()

        sensor_residuals: Dict[str, float] = {}
        for sensor in sensor_nodes:
            h_p = healthy_baseline.get(sensor, 1.0)
            o_p = observed_pressures.get(sensor, h_p)
            if h_p > 1e-6:
                sensor_residuals[sensor] = (h_p - o_p) / h_p
            else:
                sensor_residuals[sensor] = 0.0

        all_nodes = [n for n in wn.node_name_list if n != "Res"]
        residuals: Dict[str, float] = {}

        for node in all_nodes:
            if node in sensor_residuals:
                residuals[node] = sensor_residuals[node]
            else:
                total_w = 0.0
                weighted = 0.0
                for sensor, r in sensor_residuals.items():
                    try:
                        dist = nx.shortest_path_length(graph, node, sensor)
                    except nx.NetworkXNoPath:
                        dist = 100
                    w = 1.0 / (dist + 1.0)
                    total_w  += w
                    weighted += w * r
                residuals[node] = weighted / total_w if total_w > 0 else 0.0

        if not residuals:
            return "N_0_0", {}, 0.0

        predicted_node = max(residuals, key=residuals.get)

        values   = np.array(list(residuals.values()))
        max_val  = float(np.max(values))
        mean_val = float(np.mean(values))
        std_val  = float(np.std(values))

        if std_val < 1e-9:
            confidence = 0.0
        else:
            z_score    = (max_val - mean_val) / std_val
            confidence = min(100.0, max(0.0, z_score * 15.0))

        abs_residuals = {
            node: residuals[node] * healthy_baseline.get(node, 1.0)
            for node in residuals
        }
        return predicted_node, abs_residuals, round(confidence, 1)

    @staticmethod
    def detect_mnf_anomaly(df: pd.DataFrame,
                           expected_mnf_lps: float = 0.4,
                           threshold_pct: float = 15.0) -> Tuple[bool, float]:
        """
        FIXED: anomaly only when night flow is ABOVE baseline.
        Leak = extra flow at night. Flow BELOW baseline = no supply, not a leak.
        """
        night_mask = (df["Hour"] >= 2) & (df["Hour"] <= 5)
        night_data = df[night_mask]
        if len(night_data) == 0:
            return False, 0.0

        actual_mnf  = night_data["Flow Rate (L/s)"].mean()

        # FIXED: signed percentage — positive = excess flow = leak signal
        #        negative = low flow = normal or no supply (not a leak)
        anomaly_pct = ((actual_mnf - expected_mnf_lps) / expected_mnf_lps) * 100.0

        # Only flag as anomaly when flow exceeds baseline (not when it's below)
        is_anomaly = anomaly_pct > threshold_pct

        return is_anomaly, round(anomaly_pct, 1)


# ═══════════════════════════════════════════════════════════════════════════
# PART 1E: N-1 CONTINGENCY
# ═══════════════════════════════════════════════════════════════════════════

class ContingencyAnalysis:
    POPULATION_PER_NODE = 250
    LOCAL_TANK_VOLUME_L = 5000

    @staticmethod
    def simulate_n1_failure(wn: wntr.network.WaterNetworkModel,
                            failed_pipe: str,
                            avg_demand_lps: float) -> Dict:
        if failed_pipe not in wn.link_name_list:
            return {"error": f"Pipe '{failed_pipe}' not found in network"}
        graph = wn.get_graph().copy()
        try:
            link  = wn.get_link(failed_pipe)
            if hasattr(link, "start_node_name") and hasattr(link, "end_node_name"):
                graph.remove_edge(link.start_node_name, link.end_node_name)
        except Exception as e:
            return {"error": f"Failed to remove pipe: {str(e)}"}
        try:
            connected_nodes = nx.descendants(graph, "Res") | {"Res"}
        except Exception:
            connected_nodes = set(wn.node_name_list)

        affected_nodes    = [n for n in wn.node_name_list
                             if n != "Res" and n not in connected_nodes]
        n_affected         = len(affected_nodes)
        virtual_citizens   = n_affected * ContingencyAnalysis.POPULATION_PER_NODE
        total_tank_vol_l   = n_affected * ContingencyAnalysis.LOCAL_TANK_VOLUME_L
        consumption_rate   = n_affected * max(avg_demand_lps, 0.001)
        time_to_critical_h = min(total_tank_vol_l / consumption_rate / 3600.0, 72.0) \
                             if consumption_rate > 0 else 72.0

        return {
            "affected_nodes":        affected_nodes,
            "virtual_citizens":      virtual_citizens,
            "time_to_criticality_h": round(time_to_critical_h, 1),
            "best_isolation_valve":  failed_pipe,
            "impact_level": (
                "CRITICAL"  if virtual_citizens > 1000 else
                "MODERATE"  if virtual_citizens > 500  else
                "LOW"
            ),
        }

    @staticmethod
    def find_isolation_valves(wn: wntr.network.WaterNetworkModel,
                              leak_node: str) -> Tuple[List[str], List[str]]:
        graph     = wn.get_graph()
        neighbors = list(graph.neighbors(leak_node))
        pipes_to_close = []
        for link_name in wn.link_name_list:
            link = wn.get_link(link_name)
            if not (hasattr(link, "start_node_name") and hasattr(link, "end_node_name")):
                continue
            s, e = link.start_node_name, link.end_node_name
            if (s == leak_node and e in neighbors) or (e == leak_node and s in neighbors):
                pipes_to_close.append(link_name)
        return pipes_to_close, neighbors


# ═══════════════════════════════════════════════════════════════════════════
# PART 1F: ECONOMIC MODEL
# ═══════════════════════════════════════════════════════════════════════════

class EconomicModel:
    SENSOR_UNIT_COST_KZT    = 450_000
    ENERGY_COST_KZT_PER_KWH = 22.0
    KWH_PER_M3_PUMPED       = 0.4
    CO2_KG_PER_KWH          = 0.62
    PUMP_MOTOR_KW           = 15.0

    @staticmethod
    def calculate_water_losses(df: pd.DataFrame, leak_threshold_bar: float,
                               sampling_rate_hz: int,
                               water_tariff_kzt_per_liter: float) -> Dict:
        df_leak           = df[df["Pressure (bar)"] < leak_threshold_bar]
        time_per_sample_s = 3600.0 / sampling_rate_hz
        lost_liters       = (df_leak["Flow Rate (L/s)"].sum() * time_per_sample_s
                             if len(df_leak) > 0 else 0.0)
        total_flow_liters = df["Flow Rate (L/s)"].sum() * time_per_sample_s
        nrw_pct           = (lost_liters / total_flow_liters * 100.0
                             if total_flow_liters > 0 else 0.0)
        return {
            "lost_liters":       round(lost_liters, 2),
            "total_flow_liters": round(total_flow_liters, 2),
            "nrw_percentage":    round(nrw_pct, 2),
            "direct_loss_kzt":   round(lost_liters * water_tariff_kzt_per_liter, 2),
        }

    @staticmethod
    def calculate_energy_savings(pump_head_m: float, smart_pump_enabled: bool) -> Dict:
        NIGHT_HOURS  = 7
        DAY_HOURS    = 24 - NIGHT_HOURS
        baseline_kwh = EconomicModel.PUMP_MOTOR_KW * 24.0
        if smart_pump_enabled:
            actual_kwh = (EconomicModel.PUMP_MOTOR_KW * DAY_HOURS +
                          EconomicModel.PUMP_MOTOR_KW * 0.7 * NIGHT_HOURS)
        else:
            actual_kwh = baseline_kwh
        saved_kwh = baseline_kwh - actual_kwh
        return {
            "baseline_energy_kwh": round(baseline_kwh, 2),
            "actual_energy_kwh":   round(actual_kwh, 2),
            "energy_saved_kwh":    round(saved_kwh, 2),
            "energy_saved_pct":    round(saved_kwh / baseline_kwh * 100.0, 1),
            "energy_saved_kzt":    round(saved_kwh * EconomicModel.ENERGY_COST_KZT_PER_KWH, 2),
            "co2_saved_kg":        round(saved_kwh * EconomicModel.CO2_KG_PER_KWH, 2),
        }

    @staticmethod
    def calculate_roi(n_sensors: int, water_loss_dict: Dict,
                      energy_savings_dict: Dict, repair_cost_kzt: float) -> Dict:
        capex_kzt          = n_sensors * EconomicModel.SENSOR_UNIT_COST_KZT
        indirect_cost_kzt  = repair_cost_kzt if water_loss_dict["lost_liters"] > 0 else 0.0
        total_damage_kzt   = water_loss_dict["direct_loss_kzt"] + indirect_cost_kzt
        monthly_water      = water_loss_dict["direct_loss_kzt"] * 30.0
        monthly_energy     = energy_savings_dict["energy_saved_kzt"] * 30.0
        monthly_total      = monthly_water + monthly_energy
        payback_months     = capex_kzt / monthly_total if monthly_total > 0 else 9999.0
        return {
            "capex_kzt":                  capex_kzt,
            "indirect_cost_kzt":          indirect_cost_kzt,
            "total_damage_kzt":           round(total_damage_kzt, 2),
            "monthly_water_savings_kzt":  round(monthly_water, 2),
            "monthly_energy_savings_kzt": round(monthly_energy, 2),
            "monthly_total_savings_kzt":  round(monthly_total, 2),
            "payback_months":             round(payback_months, 1),
            "roi_positive":               payback_months < 24.0,
        }

    @staticmethod
    def full_economic_report(df: pd.DataFrame, leak_threshold_bar: float,
                             sampling_rate_hz: int, water_tariff_kzt: float,
                             pump_head_m: float, smart_pump: bool,
                             n_sensors: int, repair_cost_kzt: float) -> Dict:
        water_losses   = EconomicModel.calculate_water_losses(
            df, leak_threshold_bar, sampling_rate_hz, water_tariff_kzt)
        energy_savings = EconomicModel.calculate_energy_savings(pump_head_m, smart_pump)
        roi            = EconomicModel.calculate_roi(
            n_sensors, water_losses, energy_savings, repair_cost_kzt)
        return {**water_losses, **energy_savings, **roi}


# ═══════════════════════════════════════════════════════════════════════════
# PART 1G: MASTER CONTROLLER
# ═══════════════════════════════════════════════════════════════════════════

class SmartShygynBackend:
    def __init__(self, city_name: str, season_temp_celsius: float = 10.0):
        self.city_manager  = CityManager(city_name, season_temp_celsius)
        self.engine        = HydraulicEngine(self.city_manager)
        self.leak_detector = LeakDetectionAnalytics()
        self.contingency   = ContingencyAnalysis()
        self.economics     = EconomicModel()

    def run_full_simulation(self,
                            material: str = "Пластик (ПНД)",
                            pipe_age: float = 15.0,
                            pump_head_m: float = 40.0,
                            smart_pump: bool = False,
                            sampling_rate_hz: int = 1,
                            leak_node: Optional[str] = None,
                            leak_area_cm2: float = 0.8,
                            contingency_pipe: Optional[str] = None,
                            water_tariff_kzt: float = 0.55,
                            leak_threshold_bar: float = 2.7,
                            repair_cost_kzt: float = 50_000) -> Dict:

        wn_leaky = self.engine.build_network(
            material=material, pipe_age=pipe_age,
            pump_head_m=pump_head_m, smart_pump=smart_pump,
            leak_node=leak_node, leak_area_cm2=leak_area_cm2,
            contingency_pipe=contingency_pipe
        )

        results = self.engine.run_simulation(wn_leaky, sampling_rate_hz)

        focal = (leak_node
                 if leak_node and leak_node in results["pressure"].columns
                 else "N_0_0")

        pressure = results["pressure"][focal]
        age      = results["age"][focal]
        flow     = results["flow"]["P_Main"]

        pressure_smooth = self.engine.apply_signal_smoothing(
            self.engine.add_sensor_noise(pressure, noise_std=0.04), window=3)
        flow_smooth = self.engine.apply_signal_smoothing(
            self.engine.add_sensor_noise(flow, noise_std=0.08), window=3)

        n_points       = len(pressure_smooth)
        hours          = np.arange(n_points) / sampling_rate_hz
        demand_pattern = HydraulicPhysics.create_demand_pattern()

        df = pd.DataFrame({
            "Hour":            hours,
            "Pressure (bar)":  pressure_smooth.values,
            "Flow Rate (L/s)": np.abs(flow_smooth.values),
            "Water Age (h)":   age.values,
            "Demand Pattern":  np.tile(demand_pattern,
                                       n_points // 24 + 1)[:n_points],
        })

        df["Pump Head (m)"] = [
            pump_head_m * 0.7 if (int(h) % 24 >= 23 or int(h) % 24 < 6)
            else pump_head_m
            for h in hours
        ] if smart_pump else pump_head_m

        sensors = self.leak_detector.place_sensors(list(wn_leaky.node_name_list))

        healthy_pressures = self.leak_detector.build_healthy_baseline(
            self.engine, material, pipe_age, pump_head_m
        )

        observed_pressures = {
            node: float(results["pressure"][node].mean())
            for node in wn_leaky.node_name_list
            if node != "Res" and node in results["pressure"].columns
        }

        predicted_node, residuals, confidence = \
            self.leak_detector.residual_matrix_localization(
                healthy_pressures, observed_pressures, sensors, wn_leaky
            )

        mnf_anomaly, mnf_pct = self.leak_detector.detect_mnf_anomaly(df)

        degradation_pct = HydraulicPhysics.degradation_percentage(
            material, pipe_age, temp=self.city_manager.season_temp
        )
        burst_mult = self.city_manager.burst_multiplier

        failure_probs: Dict[str, float] = {}
        for node in wn_leaky.node_name_list:
            if node == "Res":
                failure_probs[node] = 0.0
            elif node in results["pressure"].columns:
                node_avg_pressure = float(results["pressure"][node].mean())
                if node_avg_pressure <= 0.0:
                    failure_probs[node] = 0.0
                else:
                    failure_probs[node] = HydraulicPhysics.failure_probability(
                        node_avg_pressure, degradation_pct, burst_mult
                    )
            else:
                failure_probs[node] = 0.0

        n1_result = None
        if contingency_pipe:
            avg_demand_lps = df["Flow Rate (L/s)"].mean() / 16
            n1_result = self.contingency.simulate_n1_failure(
                wn_leaky, contingency_pipe, avg_demand_lps
            )

        econ_report = self.economics.full_economic_report(
            df=df, leak_threshold_bar=leak_threshold_bar,
            sampling_rate_hz=sampling_rate_hz,
            water_tariff_kzt=water_tariff_kzt,
            pump_head_m=pump_head_m, smart_pump=smart_pump,
            n_sensors=len(sensors), repair_cost_kzt=repair_cost_kzt
        )

        isolation_pipes, isolation_neighbors = [], []
        if leak_node:
            isolation_pipes, isolation_neighbors = \
                self.contingency.find_isolation_valves(wn_leaky, leak_node)

        return {
            "network":               wn_leaky,
            "dataframe":             df,
            "leak_node":             leak_node or "None",
            "sensors":               sensors,
            "healthy_pressures":     healthy_pressures,
            "observed_pressures":    observed_pressures,
            "predicted_leak":        predicted_node,
            "residuals":             residuals,
            "confidence":            confidence,
            "failure_probabilities": failure_probs,
            "mnf_anomaly":           mnf_anomaly,
            "mnf_percentage":        mnf_pct,
            "n1_result":             n1_result,
            "economics":             econ_report,
            "isolation_pipes":       isolation_pipes,
            "isolation_neighbors":   isolation_neighbors,
            "city_config": {
                "name":               self.city_manager.config.name,
                "lat":                self.city_manager.config.lat,
                "lng":                self.city_manager.config.lng,
                "zoom":               self.city_manager.config.zoom,
                "elev_min":           self.city_manager.config.elev_min,
                "elev_max":           self.city_manager.config.elev_max,
                "elev_direction":     self.city_manager.config.elev_direction,
                "burst_multiplier":   self.city_manager.burst_multiplier,
                "water_stress_index": self.city_manager.config.water_stress_index,
                "description":        self.city_manager.config.description,
            },
            "material":        material,
            "pipe_age":        pipe_age,
            "roughness":       HydraulicPhysics.hazen_williams_roughness(
                material, pipe_age, temp=self.city_manager.season_temp),
            "degradation_pct": degradation_pct,
        }


__all__ = [
    "CityManager", "CityConfig", "HydraulicPhysics", "HydraulicEngine",
    "LeakDetectionAnalytics", "ContingencyAnalysis", "EconomicModel",
    "SmartShygynBackend",
]
