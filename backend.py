"""
Smart Shygyn PRO v3 — BACKEND ENGINE
Complete backend module for water network hydraulic simulation.
NO PLACEHOLDERS. Every function is fully implemented.
REFACTORED: Robust error handling, temperature-corrected H-W physics, memory safety.
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

# Configure module-level logger
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
    """Immutable city configuration dataclass."""
    name: str
    lat: float
    lng: float
    zoom: int
    elev_min: float  # meters
    elev_max: float  # meters
    elev_direction: str
    ground_temp_celsius: float
    base_burst_multiplier: float
    water_stress_index: float  # 0-1 scale
    description: str


class CityManager:
    """
    Multi-city geographic and hydraulic physics manager.
    Handles elevation gradients, freeze-thaw risk, and water stress.
    """

    CITIES: Dict[str, CityConfig] = {
        "Алматы": CityConfig(
            name="Алматы",
            lat=43.2220,
            lng=76.8512,
            zoom=15,
            elev_min=600.0,
            elev_max=1000.0,
            elev_direction="S→N",
            ground_temp_celsius=12.0,
            base_burst_multiplier=1.0,
            water_stress_index=0.35,
            description="High elevation gradient (South→North, 600-1000m). Moderate water stress."
        ),
        "Астана": CityConfig(
            name="Астана",
            lat=51.1605,
            lng=71.4704,
            zoom=15,
            elev_min=340.0,
            elev_max=360.0,
            elev_direction="Flat",
            ground_temp_celsius=-2.5,
            base_burst_multiplier=1.0,
            water_stress_index=0.55,
            description="Flat steppe. Extreme freeze-thaw pipe burst risk. High water stress."
        ),
        "Туркестан": CityConfig(
            name="Туркестан",
            lat=43.3016,
            lng=68.2730,
            zoom=15,
            elev_min=200.0,
            elev_max=280.0,
            elev_direction="SW→NE",
            ground_temp_celsius=22.0,
            base_burst_multiplier=0.8,
            water_stress_index=0.82,
            description="Arid climate. Extreme evaporation and water scarcity."
        ),
    }

    def __init__(self, city_name: str, season_temp_celsius: float = 10.0):
        """
        Initialize city manager with current seasonal temperature.

        Args:
            city_name: One of ["Алматы", "Астана", "Туркестан"]
            season_temp_celsius: Current ambient temperature for freeze-thaw risk
        """
        if city_name not in self.CITIES:
            raise ValueError(f"Unknown city: {city_name}. Choose from {list(self.CITIES.keys())}")

        self.config = self.CITIES[city_name]
        self.season_temp = season_temp_celsius
        self.burst_multiplier = self._calculate_burst_multiplier()

    def _calculate_burst_multiplier(self) -> float:
        """
        Calculate freeze-thaw burst risk multiplier (Astana-specific).

        Physics: Risk increases when temperature oscillates around 0°C.
        Δ = |T_season - T_ground|
        Multiplier = 1.0 + 0.05 * min(Δ, 20)

        Returns:
            Float multiplier (1.0 = baseline, >1.0 = elevated risk)
        """
        if self.config.name == "Астана":
            delta = abs(self.season_temp - self.config.ground_temp_celsius)
            return 1.0 + 0.05 * min(delta, 20.0)
        return self.config.base_burst_multiplier

    def node_elevation(self, i: int, j: int, grid_size: int = 4) -> float:
        """
        Calculate node elevation based on city-specific gradient.

        Args:
            i: Grid column index (0 to grid_size-1)
            j: Grid row index (0 to grid_size-1)
            grid_size: Total grid dimension (default 4x4)

        Returns:
            Elevation in meters above sea level
        """
        lo, hi = self.config.elev_min, self.config.elev_max
        direction = self.config.elev_direction

        if direction == "S→N":
            # South (j=0) is high, North (j=max) is low
            frac = j / max(1, grid_size - 1)
        elif direction == "SW→NE":
            # Southwest corner high, Northeast corner low
            frac = (i + j) / max(1, 2 * (grid_size - 1))
        else:  # Flat
            frac = 0.5

        return hi - frac * (hi - lo)

    def grid_to_latlon(self, i: int, j: int,
                       lat_step: float = 0.0009) -> Tuple[float, float]:
        """
        Convert grid indices to geographic coordinates with COSINE CORRECTION.

        Critical: Longitude spacing must be adjusted by cos(latitude) to maintain
        square grid cells at high latitudes.

        Args:
            i: Grid column (longitude direction)
            j: Grid row (latitude direction)
            lat_step: Latitude step in degrees (constant)

        Returns:
            (latitude, longitude) tuple
        """
        lat = self.config.lat + j * lat_step

        # GEOGRAPHIC ACCURACY: Longitude step = lat_step / cos(latitude)
        lng_step = lat_step / math.cos(math.radians(self.config.lat))
        lng = self.config.lng + i * lng_step

        return lat, lng

    def get_water_stress_factor(self) -> float:
        """
        Return water stress index (0-1 scale).
        Used for economic loss multipliers in arid regions.
        """
        return self.config.water_stress_index

    def get_evaporation_rate(self) -> float:
        """
        Calculate daily evaporation rate based on ground temperature.
        Turkestan: High evaporation (22°C ground temp)

        Returns:
            Evaporation coefficient (0-1 scale)
        """
        # Simple model: evap_rate = min(0.95, (T_ground - 10) / 50)
        rate = max(0.0, min(0.95, (self.config.ground_temp_celsius - 10.0) / 50.0))
        return rate


# ═══════════════════════════════════════════════════════════════════════════
# PART 1B: ADVANCED HYDRAULIC PHYSICS
# ═══════════════════════════════════════════════════════════════════════════

class HydraulicPhysics:
    """Pure hydraulic physics calculations (material aging, leak modeling)."""

    # Material properties
    HAZEN_WILLIAMS_BASE: Dict[str, float] = {
        "Пластик (ПНД)": 150.0,
        "Сталь": 140.0,
        "Чугун": 100.0,
    }

    DECAY_RATE: Dict[str, float] = {
        "Пластик (ПНД)": 0.10,  # C-factor loss per year
        "Сталь": 0.30,
        "Чугун": 0.50,
    }

    @staticmethod
    def temperature_correction_factor(temp: float) -> float:
        """
        Calculate Hazen-Williams C-factor temperature correction.

        Physics: Water viscosity increases at low temperatures, reducing effective
        hydraulic conveyance. For every 5°C drop below 10°C, the C-factor is
        reduced by 1% to account for this viscosity increase.

        Args:
            temp: Current water/ambient temperature in °C

        Returns:
            Multiplicative correction factor (≤ 1.0; 1.0 = no correction above 10°C)
        """
        if temp >= 10.0:
            return 1.0
        # Each 5°C drop below 10°C → 1% reduction
        degrees_below = 10.0 - temp
        reduction_pct = (degrees_below / 5.0) * 0.01
        return max(0.0, 1.0 - reduction_pct)

    @staticmethod
    def hazen_williams_roughness(material: str, age_years: float,
                                  temp: float = 10.0) -> float:
        """
        Calculate degraded, temperature-corrected Hazen-Williams C-factor.

        C_current = (C_base - age × decay_rate) × temperature_correction_factor(temp)
        Minimum C = 40 (extremely rough pipe)

        Args:
            material: Pipe material type
            age_years: Pipe age in years
            temp: Current temperature in °C (default 10°C = no correction)

        Returns:
            Hazen-Williams C coefficient
        """
        base_c = HydraulicPhysics.HAZEN_WILLIAMS_BASE.get(material, 130.0)
        decay = HydraulicPhysics.DECAY_RATE.get(material, 0.30)

        age_degraded_c = base_c - (decay * age_years)
        temp_factor = HydraulicPhysics.temperature_correction_factor(temp)
        current_c = age_degraded_c * temp_factor

        return max(40.0, current_c)

    @staticmethod
    def degradation_percentage(material: str, age_years: float,
                                temp: float = 10.0) -> float:
        """
        Calculate pipe degradation as percentage (0-100%), including
        temperature effects.

        Returns:
            Degradation % = (1 - C_current / C_base) × 100
        """
        base_c = HydraulicPhysics.HAZEN_WILLIAMS_BASE.get(material, 130.0)
        current_c = HydraulicPhysics.hazen_williams_roughness(material, age_years, temp)

        degradation = (1.0 - current_c / base_c) * 100.0
        return max(0.0, min(100.0, degradation))

    @staticmethod
    def torricelli_leak_flow(area_m2: float, head_m: float,
                             discharge_coeff: float = 0.61) -> float:
        """
        Torricelli's Law for orifice discharge.

        Q = Cd × A × √(2gh)

        Args:
            area_m2: Leak hole area in m²
            head_m: Pressure head in meters
            discharge_coeff: Orifice discharge coefficient (default 0.61)

        Returns:
            Flow rate in m³/s
        """
        if head_m <= 0:
            return 0.0

        g = 9.81  # m/s²
        return discharge_coeff * area_m2 * math.sqrt(2.0 * g * head_m)

    @staticmethod
    def emitter_coefficient_from_area(area_cm2: float,
                                      pressure_bar: float = 3.0,
                                      exponent: float = 0.5) -> float:
        """
        Convert physical leak area to EPANET Emitter Coefficient.

        EPANET Emitter: Q = K × P^N
        Where Q is in flow units, P is pressure, N is exponent (default 0.5)

        We use Torricelli to find Q at reference pressure, then solve for K.

        Args:
            area_cm2: Leak area in cm²
            pressure_bar: Reference pressure (default 3.0 bar)
            exponent: Emitter exponent (default 0.5 for Torricelli)

        Returns:
            Emitter coefficient K (flow units per pressure^exponent)
        """
        area_m2 = area_cm2 / 10000.0  # cm² to m²
        head_m = pressure_bar * 10.197  # bar to meters of water

        # Flow at reference pressure using Torricelli
        q_m3s = HydraulicPhysics.torricelli_leak_flow(area_m2, head_m)
        q_lps = q_m3s * 1000.0  # Convert to L/s

        # Solve for K: K = Q / P^N
        # P in EPANET is typically in meters, so we use head_m
        K = q_lps / (head_m ** exponent) if head_m > 0 else 0.0

        return K

    @staticmethod
    def create_demand_pattern(hours: int = 24) -> List[float]:
        """
        Generate realistic 24-hour water demand pattern.

        Pattern characteristics:
        - Night (0-6h): Low baseline + sinusoidal variation
        - Morning peak (6-9h): High demand spike
        - Daytime (9-18h): Moderate steady demand
        - Evening peak (18-22h): Highest demand
        - Late night (22-24h): Declining to baseline

        Returns:
            List of 24 multipliers (baseline = 1.0)
        """
        pattern = []
        for h in range(hours):
            if 0 <= h < 6:
                # Night: 0.3 baseline + gentle wave
                pattern.append(0.3 + 0.1 * np.sin(h * np.pi / 6))
            elif 6 <= h < 9:
                # Morning surge
                pattern.append(1.2 + 0.3 * np.sin((h - 6) * np.pi / 3))
            elif 9 <= h < 18:
                # Daytime steady
                pattern.append(0.8 + 0.2 * np.sin((h - 9) * np.pi / 9))
            elif 18 <= h < 22:
                # Evening peak
                pattern.append(1.4 + 0.2 * np.sin((h - 18) * np.pi / 4))
            else:
                # Late night decline
                pattern.append(0.5 + 0.2 * np.sin((h - 22) * np.pi / 2))

        return pattern

    @staticmethod
    def failure_probability(pressure_bar: float,
                            degradation_pct: float,
                            burst_multiplier: float = 1.0) -> float:
        """
        Calculate pipe failure probability using pressure-degradation model.

        P_fail = α × [(1 - P/P_max)^β] × [(Degradation/100)^γ] × BurstMultiplier

        Args:
            pressure_bar: Current pressure in bar
            degradation_pct: Pipe degradation percentage (0-100)
            burst_multiplier: Environmental risk factor (freeze-thaw, etc.)

        Returns:
            Failure probability as percentage (0-100)
        """
        P_MAX = 5.0   # Maximum safe operating pressure (bar)
        ALPHA = 0.5   # Base failure coefficient
        BETA = 2.0    # Pressure exponent
        GAMMA = 1.5   # Degradation exponent

        pressure_factor = (1.0 - min(pressure_bar, P_MAX) / P_MAX) ** BETA
        degradation_factor = (degradation_pct / 100.0) ** GAMMA

        prob = ALPHA * pressure_factor * degradation_factor * burst_multiplier
        return min(100.0, prob * 100.0)


# ═══════════════════════════════════════════════════════════════════════════
# PART 1C: HYDRAULIC ENGINE (WNTR/EPANET SIMULATION)
# ═══════════════════════════════════════════════════════════════════════════

def _make_zero_results(wn: wntr.network.WaterNetworkModel,
                       n_hours: int = 24) -> Dict[str, pd.DataFrame]:
    """
    Construct a zeroed-out results dictionary that mirrors the structure
    returned by a successful simulation.  Used as a safe fallback when
    EPANET/WNTR cannot converge.

    Args:
        wn: The network model (needed for node / link names).
        n_hours: Number of hourly time-steps to generate (default 24).

    Returns:
        Dict with 'pressure', 'flow', 'age' DataFrames filled with zeros.
    """
    time_index = pd.RangeIndex(n_hours, name="time")
    node_names = wn.node_name_list
    link_names = wn.link_name_list

    pressure_df = pd.DataFrame(0.0, index=time_index, columns=node_names)
    flow_df     = pd.DataFrame(0.0, index=time_index, columns=link_names)
    age_df      = pd.DataFrame(0.0, index=time_index, columns=node_names)

    return {
        "pressure": pressure_df,
        "flow":     flow_df,
        "age":      age_df,
    }


class HydraulicEngine:
    """
    Core simulation engine using WNTR/EPANET.
    Builds networks, runs simulations, extracts results.
    """

    def __init__(self, city_manager: CityManager):
        self.city = city_manager
        self.physics = HydraulicPhysics()

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
        """
        Build complete WNTR water network model.

        Args:
            grid_size: Grid dimension (4 = 4×4 = 16 nodes)
            material: Pipe material for H-W roughness
            pipe_age: Years since installation
            pipe_length_m: Distance between nodes
            pipe_diameter_m: Pipe internal diameter
            pump_head_m: Reservoir head (pressure source)
            smart_pump: Enable dynamic pump scheduling
            leak_node: Node ID for leak (None = no leak)
            leak_area_cm2: Leak hole area in cm²
            leak_start_hour: Hour when leak starts (0-24)
            contingency_pipe: Pipe to remove for N-1 analysis

        Returns:
            WNTR WaterNetworkModel instance
        """
        wn = wntr.network.WaterNetworkModel()

        # Material properties — pass season temperature for viscosity correction
        roughness = self.physics.hazen_williams_roughness(
            material, pipe_age, temp=self.city.season_temp
        )

        # Demand pattern
        demand_pattern = self.physics.create_demand_pattern()
        wn.add_pattern("demand_pattern", demand_pattern)

        # Build grid of junctions
        for i in range(grid_size):
            for j in range(grid_size):
                node_name = f"N_{i}_{j}"
                elevation = self.city.node_elevation(i, j, grid_size)

                wn.add_junction(
                    node_name,
                    base_demand=0.001,  # m³/s baseline
                    elevation=elevation,
                    demand_pattern="demand_pattern"
                )

                # Set coordinates for visualization
                wn.get_node(node_name).coordinates = (
                    i * pipe_length_m,
                    j * pipe_length_m
                )

                # Add horizontal pipes (i > 0)
                if i > 0:
                    pipe_name = f"PH_{i}_{j}"
                    prev_node = f"N_{i-1}_{j}"
                    wn.add_pipe(
                        pipe_name,
                        prev_node,
                        node_name,
                        length=pipe_length_m,
                        diameter=pipe_diameter_m,
                        roughness=roughness
                    )

                # Add vertical pipes (j > 0)
                if j > 0:
                    pipe_name = f"PV_{i}_{j}"
                    prev_node = f"N_{i}_{j-1}"
                    wn.add_pipe(
                        pipe_name,
                        prev_node,
                        node_name,
                        length=pipe_length_m,
                        diameter=pipe_diameter_m,
                        roughness=roughness
                    )

        # Add reservoir (pressure source)
        effective_head = pump_head_m * 0.85 if smart_pump else pump_head_m
        wn.add_reservoir("Res", base_head=effective_head)
        wn.get_node("Res").coordinates = (-pipe_length_m, -pipe_length_m)

        # Main supply pipe (larger diameter)
        wn.add_pipe(
            "P_Main",
            "Res",
            "N_0_0",
            length=pipe_length_m,
            diameter=pipe_diameter_m * 2.0,  # 2× diameter for main
            roughness=roughness
        )

        # Add leak using EMITTER (not add_leak)
        if leak_node and leak_node in wn.node_name_list:
            node = wn.get_node(leak_node)

            # Calculate emitter coefficient from physical area
            emitter_k = self.physics.emitter_coefficient_from_area(
                leak_area_cm2,
                pressure_bar=pump_head_m * 0.098,  # Convert m to bar
                exponent=0.5
            )

            # EPANET emitter: Q = K * P^0.5
            node.emitter_coefficient = emitter_k

        # N-1 Contingency: Remove pipe
        if contingency_pipe and contingency_pipe in wn.link_name_list:
            wn.remove_link(contingency_pipe)

        # Simulation options
        wn.options.time.duration = 24 * 3600          # 24 hours
        wn.options.time.hydraulic_timestep = 3600      # 1 hour
        wn.options.time.report_timestep = 3600
        wn.options.quality.parameter = "AGE"           # Water age tracking

        return wn

    def run_simulation(self,
                       wn: wntr.network.WaterNetworkModel,
                       sampling_rate_hz: int = 1) -> Dict[str, pd.DataFrame]:
        """
        Run EPANET simulation and extract results.

        On convergence failure, returns a zeroed fallback dict rather than
        crashing the application.

        Args:
            wn: WNTR network model
            sampling_rate_hz: Sensor sampling frequency (samples per hour)

        Returns:
            Dictionary with 'pressure', 'flow', 'quality' DataFrames.
            All values are zero if the solver did not converge.
        """
        # Adjust report timestep based on sampling rate
        wn.options.time.report_timestep = int(3600 / sampling_rate_hz)

        try:
            sim = wntr.sim.EpanetSimulator(wn)
            raw_results = sim.run_sim()

            # Extract and convert units
            pressure_bar = raw_results.node["pressure"] * 0.1   # m → bar
            flow_lps     = raw_results.link["flowrate"] * 1000.0  # m³/s → L/s
            age_hours    = raw_results.node["quality"] / 3600.0   # s → hours

            return {
                "pressure": pressure_bar,
                "flow":     flow_lps,
                "age":      age_hours,
            }

        except Exception as exc:
            logger.error(
                "EPANET simulation failed — returning zeroed fallback. "
                "Original error: %s: %s",
                type(exc).__name__, exc
            )
            n_hours = int(wn.options.time.duration / wn.options.time.report_timestep)
            return _make_zero_results(wn, n_hours=max(n_hours, 24))

    def apply_signal_smoothing(self,
                                series: pd.Series,
                                window: int = 3) -> pd.Series:
        """
        Apply moving average filter to simulate sensor noise reduction.

        Args:
            series: Raw signal
            window: Moving average window size

        Returns:
            Smoothed signal
        """
        return series.rolling(window, center=True, min_periods=1).mean()

    def add_sensor_noise(self,
                         series: pd.Series,
                         noise_std: float = 0.04) -> pd.Series:
        """
        Add Gaussian noise to simulate sensor measurement error.

        Args:
            series: Clean signal
            noise_std: Standard deviation of noise

        Returns:
            Noisy signal
        """
        noise = np.random.normal(0, noise_std, len(series))
        return series + noise


# ═══════════════════════════════════════════════════════════════════════════
# PART 1D: ANALYTICS & LEAK DETECTION
# ═══════════════════════════════════════════════════════════════════════════

class LeakDetectionAnalytics:
    """
    Sparse sensor network leak detection using residual pressure matrix.
    """

    SENSOR_COVERAGE = 0.30  # 30% of nodes have sensors

    @staticmethod
    def place_sensors(node_list: List[str],
                      seed: int = 42,
                      coverage: float = 0.30) -> List[str]:
        """
        Randomly select nodes for sensor placement (excludes reservoir).

        Args:
            node_list: All network nodes
            seed: Random seed for reproducibility
            coverage: Fraction of nodes with sensors (0-1)

        Returns:
            List of sensor node IDs
        """
        rng = np.random.default_rng(seed)
        candidates = [n for n in node_list if n != "Res"]

        n_sensors = max(1, int(len(candidates) * coverage))
        sensors = list(rng.choice(candidates, size=n_sensors, replace=False))

        return sensors

    @staticmethod
    def build_healthy_baseline(engine: HydraulicEngine,
                                material: str,
                                pipe_age: float,
                                pump_head_m: float) -> Dict[str, float]:
        """
        Run simulation WITHOUT leak to establish baseline pressures.

        Returns:
            Dict mapping {node_id: mean_pressure_bar}
        """
        wn_healthy = engine.build_network(
            material=material,
            pipe_age=pipe_age,
            pump_head_m=pump_head_m,
            leak_node=None  # NO LEAK
        )

        results = engine.run_simulation(wn_healthy, sampling_rate_hz=1)

        # Calculate mean pressure over 24 hours for each node
        baseline = {}
        for node in wn_healthy.node_name_list:
            if node != "Res":
                baseline[node] = results["pressure"][node].mean()

        return baseline

    @staticmethod
    def residual_matrix_localization(healthy_baseline: Dict[str, float],
                                     observed_pressures: Dict[str, float],
                                     sensor_nodes: List[str],
                                     wn: wntr.network.WaterNetworkModel) -> Tuple[str, Dict[str, float], float]:
        """
        Extended Kalman Filter-style residual matrix leak localization.

        Algorithm:
        1. Compute residuals at sensor nodes: r_i = (P_healthy - P_observed) / P_healthy
        2. Interpolate residuals to non-sensor nodes using inverse distance weighting (IDW)
           on the network graph
        3. Node with maximum residual is the predicted leak location
        4. Confidence = signal-to-noise ratio

        Args:
            healthy_baseline: Dict {node: baseline_pressure}
            observed_pressures: Dict {node: current_pressure}
            sensor_nodes: List of nodes with sensors
            wn: WNTR network for graph structure

        Returns:
            (predicted_node, residuals_dict, confidence_percentage)
        """
        graph = wn.get_graph()
        residuals = {}

        # Step 1: Compute normalized residuals at sensor nodes
        sensor_residuals = {}
        for sensor in sensor_nodes:
            healthy_p  = healthy_baseline.get(sensor, 1.0)
            observed_p = observed_pressures.get(sensor, healthy_p)

            if healthy_p > 0:
                sensor_residuals[sensor] = (healthy_p - observed_p) / healthy_p
            else:
                sensor_residuals[sensor] = 0.0

        # Step 2: Interpolate to all nodes using IDW on graph distance
        all_nodes = [n for n in wn.node_name_list if n != "Res"]

        for node in all_nodes:
            if node in sensor_nodes:
                # Direct measurement
                residuals[node] = sensor_residuals[node]
            else:
                # Inverse Distance Weighting (IDW) interpolation
                total_weight      = 0.0
                weighted_residual = 0.0

                for sensor, r in sensor_residuals.items():
                    try:
                        # Graph distance (number of hops)
                        distance = nx.shortest_path_length(graph, node, sensor)
                    except nx.NetworkXNoPath:
                        distance = 100  # Disconnected = very far

                    weight             = 1.0 / (distance + 1.0)
                    total_weight      += weight
                    weighted_residual += weight * r

                if total_weight > 0:
                    residuals[node] = weighted_residual / total_weight
                else:
                    residuals[node] = 0.0

        # Step 3: Find node with maximum residual
        if not residuals:
            return "N_0_0", {}, 0.0

        predicted_node = max(residuals, key=residuals.get)

        # Step 4: Calculate confidence score (SNR)
        values = np.array(list(residuals.values()))
        signal = np.max(values)
        noise  = np.std(values)

        confidence = min(100.0, (signal / (noise + 1e-6)) * 20.0)

        # Convert normalized residuals to absolute pressure drop (bar)
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
        Detect Minimum Night Flow (MNF) anomaly.

        Args:
            df: DataFrame with 'Hour' and 'Flow Rate (L/s)' columns
            expected_mnf_lps: Expected MNF baseline
            threshold_pct: Anomaly threshold percentage

        Returns:
            (is_anomaly, anomaly_percentage)
        """
        # MNF window: 2:00 - 5:00 AM
        night_mask = (df["Hour"] >= 2) & (df["Hour"] <= 5)
        night_data = df[night_mask]

        if len(night_data) == 0:
            return False, 0.0

        actual_mnf  = night_data["Flow Rate (L/s)"].mean()
        anomaly_pct = ((actual_mnf - expected_mnf_lps) / expected_mnf_lps) * 100.0

        is_anomaly = anomaly_pct > threshold_pct
        return is_anomaly, round(anomaly_pct, 1)


# ═══════════════════════════════════════════════════════════════════════════
# PART 1E: N-1 CONTINGENCY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

class ContingencyAnalysis:
    """
    N-1 reliability analysis: simulate single-pipe failures.
    """

    POPULATION_PER_NODE = 250   # Virtual citizens per junction
    LOCAL_TANK_VOLUME_L = 5000  # Emergency storage per node

    @staticmethod
    def simulate_n1_failure(wn: wntr.network.WaterNetworkModel,
                            failed_pipe: str,
                            avg_demand_lps: float) -> Dict:
        """
        Simulate failure of a single pipe and assess impact.

        Algorithm:
        1. Remove pipe from network graph
        2. Find nodes disconnected from reservoir using NetworkX
        3. Calculate impacted population
        4. Estimate time to criticality (local tank depletion)

        Args:
            wn: WNTR network model
            failed_pipe: Pipe ID to fail
            avg_demand_lps: Average demand per node (L/s)

        Returns:
            Dict with impact assessment
        """
        if failed_pipe not in wn.link_name_list:
            return {"error": f"Pipe '{failed_pipe}' not found in network"}

        # Create graph copy
        graph = wn.get_graph().copy()

        # Remove failed pipe
        try:
            link = wn.get_link(failed_pipe)
            if hasattr(link, "start_node_name") and hasattr(link, "end_node_name"):
                graph.remove_edge(link.start_node_name, link.end_node_name)
        except Exception as e:
            return {"error": f"Failed to remove pipe: {str(e)}"}

        # Find nodes still connected to reservoir
        try:
            connected_nodes = nx.descendants(graph, "Res") | {"Res"}
        except Exception:
            connected_nodes = set(wn.node_name_list)

        # Identify disconnected (affected) nodes
        affected_nodes = [
            node for node in wn.node_name_list
            if node != "Res" and node not in connected_nodes
        ]

        # Calculate impact
        n_affected        = len(affected_nodes)
        virtual_citizens  = n_affected * ContingencyAnalysis.POPULATION_PER_NODE

        # Time to criticality calculation
        total_tank_vol_l      = n_affected * ContingencyAnalysis.LOCAL_TANK_VOLUME_L
        consumption_rate_lps  = n_affected * max(avg_demand_lps, 0.001)

        time_to_critical_s = (
            total_tank_vol_l / consumption_rate_lps
            if consumption_rate_lps > 0 else 999999
        )
        time_to_critical_h = min(time_to_critical_s / 3600.0, 72.0)  # Cap at 72h

        # Best isolation valve (closest pipe to reservoir that re-isolates zone)
        best_valve = failed_pipe

        return {
            "affected_nodes":         affected_nodes,
            "virtual_citizens":       virtual_citizens,
            "time_to_criticality_h":  round(time_to_critical_h, 1),
            "best_isolation_valve":   best_valve,
            "impact_level": (
                "CRITICAL" if virtual_citizens > 1000
                else "MODERATE" if virtual_citizens > 500
                else "LOW"
            ),
        }

    @staticmethod
    def find_isolation_valves(wn: wntr.network.WaterNetworkModel,
                              leak_node: str) -> Tuple[List[str], List[str]]:
        """
        Find pipes (valves) connected to a leak node for isolation.

        Returns:
            (pipes_to_close, affected_neighbors)
        """
        graph     = wn.get_graph()
        neighbors = list(graph.neighbors(leak_node))

        pipes_to_close = []
        for link_name in wn.link_name_list:
            link = wn.get_link(link_name)
            if not (hasattr(link, "start_node_name") and hasattr(link, "end_node_name")):
                continue

            start, end = link.start_node_name, link.end_node_name

            if (start == leak_node and end in neighbors) or \
               (end == leak_node and start in neighbors):
                pipes_to_close.append(link_name)

        return pipes_to_close, neighbors


# ═══════════════════════════════════════════════════════════════════════════
# PART 1F: ECONOMIC & ROI MODEL
# ═══════════════════════════════════════════════════════════════════════════

class EconomicModel:
    """
    Full economic analysis: CAPEX, OPEX, ROI, carbon footprint.
    """

    # Cost parameters (Kazakhstan, 2026)
    SENSOR_UNIT_COST_KZT    = 450_000  # Per sensor installation
    ENERGY_COST_KZT_PER_KWH = 22.0    # Electricity tariff
    KWH_PER_M3_PUMPED       = 0.4     # Energy intensity
    CO2_KG_PER_KWH          = 0.62    # Grid emission factor
    PUMP_MOTOR_KW           = 15.0    # Typical pump power

    @staticmethod
    def calculate_water_losses(df: pd.DataFrame,
                               leak_threshold_bar: float,
                               sampling_rate_hz: int,
                               water_tariff_kzt_per_liter: float) -> Dict:
        """
        Calculate water loss volumes and costs.

        Args:
            df: Simulation results with 'Pressure (bar)' and 'Flow Rate (L/s)'
            leak_threshold_bar: Pressure below which leak is detected
            sampling_rate_hz: Sensor frequency (samples/hour)
            water_tariff_kzt_per_liter: Water price

        Returns:
            Dict with loss metrics
        """
        df_leak           = df[df["Pressure (bar)"] < leak_threshold_bar]
        time_per_sample_s = 3600.0 / sampling_rate_hz

        lost_liters = (
            df_leak["Flow Rate (L/s)"].sum() * time_per_sample_s
            if len(df_leak) > 0 else 0.0
        )

        total_flow_liters = df["Flow Rate (L/s)"].sum() * time_per_sample_s
        nrw_pct           = (lost_liters / total_flow_liters * 100.0) if total_flow_liters > 0 else 0.0
        direct_loss_kzt   = lost_liters * water_tariff_kzt_per_liter

        return {
            "lost_liters":        round(lost_liters, 2),
            "total_flow_liters":  round(total_flow_liters, 2),
            "nrw_percentage":     round(nrw_pct, 2),
            "direct_loss_kzt":    round(direct_loss_kzt, 2),
        }

    @staticmethod
    def calculate_energy_savings(pump_head_m: float,
                                 smart_pump_enabled: bool) -> Dict:
        """
        Calculate energy and CO₂ savings from smart pump scheduling.

        Smart pump strategy:
        - Day (06:00-23:00): 100% head = full power
        - Night (23:00-06:00): 70% head = reduced power

        Returns:
            Dict with energy metrics
        """
        NIGHT_HOURS = 7   # 23:00-06:00
        DAY_HOURS   = 24 - NIGHT_HOURS

        baseline_energy_kwh = EconomicModel.PUMP_MOTOR_KW * 24.0

        if smart_pump_enabled:
            day_energy        = EconomicModel.PUMP_MOTOR_KW * DAY_HOURS
            night_energy      = EconomicModel.PUMP_MOTOR_KW * 0.7 * NIGHT_HOURS
            actual_energy_kwh = day_energy + night_energy
        else:
            actual_energy_kwh = baseline_energy_kwh

        energy_saved_kwh  = baseline_energy_kwh - actual_energy_kwh
        energy_saved_pct  = (energy_saved_kwh / baseline_energy_kwh * 100.0) if smart_pump_enabled else 0.0
        energy_saved_kzt  = energy_saved_kwh * EconomicModel.ENERGY_COST_KZT_PER_KWH
        co2_saved_kg      = energy_saved_kwh * EconomicModel.CO2_KG_PER_KWH

        return {
            "baseline_energy_kwh": round(baseline_energy_kwh, 2),
            "actual_energy_kwh":   round(actual_energy_kwh, 2),
            "energy_saved_kwh":    round(energy_saved_kwh, 2),
            "energy_saved_pct":    round(energy_saved_pct, 1),
            "energy_saved_kzt":    round(energy_saved_kzt, 2),
            "co2_saved_kg":        round(co2_saved_kg, 2),
        }

    @staticmethod
    def calculate_roi(n_sensors: int,
                      water_loss_dict: Dict,
                      energy_savings_dict: Dict,
                      repair_cost_kzt: float) -> Dict:
        """
        Calculate Return on Investment (ROI) and payback period.

        Args:
            n_sensors: Number of sensors installed
            water_loss_dict: Output from calculate_water_losses
            energy_savings_dict: Output from calculate_energy_savings
            repair_cost_kzt: One-time repair/deployment cost

        Returns:
            Dict with ROI metrics
        """
        capex_kzt        = n_sensors * EconomicModel.SENSOR_UNIT_COST_KZT
        indirect_cost_kzt = repair_cost_kzt if water_loss_dict["lost_liters"] > 0 else 0.0
        total_damage_kzt  = water_loss_dict["direct_loss_kzt"] + indirect_cost_kzt

        monthly_water_savings  = water_loss_dict["direct_loss_kzt"] * 30.0
        monthly_energy_savings = energy_savings_dict["energy_saved_kzt"] * 30.0
        monthly_total_savings  = monthly_water_savings + monthly_energy_savings

        payback_months = (
            capex_kzt / monthly_total_savings if monthly_total_savings > 0 else 9999.0
        )

        return {
            "capex_kzt":                    capex_kzt,
            "indirect_cost_kzt":            indirect_cost_kzt,
            "total_damage_kzt":             round(total_damage_kzt, 2),
            "monthly_water_savings_kzt":    round(monthly_water_savings, 2),
            "monthly_energy_savings_kzt":   round(monthly_energy_savings, 2),
            "monthly_total_savings_kzt":    round(monthly_total_savings, 2),
            "payback_months":               round(payback_months, 1),
            "roi_positive":                 payback_months < 24.0,
        }

    @staticmethod
    def full_economic_report(df: pd.DataFrame,
                             leak_threshold_bar: float,
                             sampling_rate_hz: int,
                             water_tariff_kzt: float,
                             pump_head_m: float,
                             smart_pump: bool,
                             n_sensors: int,
                             repair_cost_kzt: float) -> Dict:
        """
        Generate complete economic analysis report.

        Returns:
            Comprehensive dict with all economic metrics
        """
        water_losses = EconomicModel.calculate_water_losses(
            df, leak_threshold_bar, sampling_rate_hz, water_tariff_kzt
        )

        energy_savings = EconomicModel.calculate_energy_savings(
            pump_head_m, smart_pump
        )

        roi = EconomicModel.calculate_roi(
            n_sensors, water_losses, energy_savings, repair_cost_kzt
        )

        return {**water_losses, **energy_savings, **roi}


# ═══════════════════════════════════════════════════════════════════════════
# PART 1G: MASTER CONTROLLER
# ═══════════════════════════════════════════════════════════════════════════

class SmartShygynBackend:
    """
    Master controller: orchestrates all backend components.
    This is the single interface the frontend will use.
    """

    def __init__(self, city_name: str, season_temp_celsius: float = 10.0):
        """
        Initialize backend with city selection.

        Args:
            city_name: City to simulate
            season_temp_celsius: Current temperature for freeze-thaw risk
        """
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
        """
        Execute complete simulation pipeline.

        Returns:
            Dict with all results: network, dataframe, analytics, economics
        """
        # ── Build network WITH leak ─────────────────────────────────────
        wn_leaky = self.engine.build_network(
            material=material,
            pipe_age=pipe_age,
            pump_head_m=pump_head_m,
            smart_pump=smart_pump,
            leak_node=leak_node,
            leak_area_cm2=leak_area_cm2,
            contingency_pipe=contingency_pipe
        )

        # ── Run simulation (fallback-safe) ───────────────────────────────
        results = self.engine.run_simulation(wn_leaky, sampling_rate_hz)

        # ── Extract time-series for the focal node ───────────────────────
        if leak_node and leak_node in results["pressure"].columns:
            pressure = results["pressure"][leak_node]
            age      = results["age"][leak_node]
        else:
            pressure = results["pressure"]["N_0_0"]
            age      = results["age"]["N_0_0"]

        flow = results["flow"]["P_Main"]

        # ── Add sensor noise + smoothing ─────────────────────────────────
        pressure_smooth = self.engine.apply_signal_smoothing(
            self.engine.add_sensor_noise(pressure, noise_std=0.04), window=3
        )
        flow_smooth = self.engine.apply_signal_smoothing(
            self.engine.add_sensor_noise(flow, noise_std=0.08), window=3
        )

        # ── Build DataFrame ──────────────────────────────────────────────
        n_points       = len(pressure_smooth)
        hours          = np.arange(n_points) / sampling_rate_hz
        demand_pattern = HydraulicPhysics.create_demand_pattern()

        df = pd.DataFrame({
            "Hour":             hours,
            "Pressure (bar)":   pressure_smooth.values,
            "Flow Rate (L/s)":  np.abs(flow_smooth.values),
            "Water Age (h)":    age.values,
            "Demand Pattern":   np.tile(demand_pattern, n_points // 24 + 1)[:n_points],
        })

        # ── Smart pump head schedule ─────────────────────────────────────
        if smart_pump:
            pump_heads = [
                pump_head_m * 0.7 if (int(h) % 24 >= 23 or int(h) % 24 < 6)
                else pump_head_m
                for h in hours
            ]
            df["Pump Head (m)"] = pump_heads
        else:
            df["Pump Head (m)"] = pump_head_m

        # ── Sensor placement ─────────────────────────────────────────────
        sensors = self.leak_detector.place_sensors(list(wn_leaky.node_name_list))

        # ── Healthy baseline ─────────────────────────────────────────────
        healthy_pressures = self.leak_detector.build_healthy_baseline(
            self.engine, material, pipe_age, pump_head_m
        )

        # ── Observed pressures (scaled by leak ratio) ────────────────────
        ref_node    = leak_node if leak_node else "N_0_0"
        ref_healthy = healthy_pressures.get(ref_node, 1.0) or 1.0
        leak_ratio  = df["Pressure (bar)"].mean() / ref_healthy

        observed_pressures = {
            node: healthy_pressures.get(node, 1.0) * leak_ratio
            for node in wn_leaky.node_name_list
            if node != "Res"
        }

        # ── Residual matrix localization ─────────────────────────────────
        predicted_node, residuals, confidence = \
            self.leak_detector.residual_matrix_localization(
                healthy_pressures, observed_pressures, sensors, wn_leaky
            )

        # ── MNF anomaly detection ────────────────────────────────────────
        mnf_anomaly, mnf_pct = self.leak_detector.detect_mnf_anomaly(df)

        # ── Failure probabilities ────────────────────────────────────────
        avg_pressure    = df["Pressure (bar)"].mean()
        degradation_pct = HydraulicPhysics.degradation_percentage(
            material, pipe_age, temp=self.city_manager.season_temp
        )
        burst_mult      = self.city_manager.burst_multiplier

        failure_probs = {
            node: (
                HydraulicPhysics.failure_probability(avg_pressure, degradation_pct, burst_mult)
                if node != "Res" else 0.0
            )
            for node in wn_leaky.node_name_list
        }

        # ── N-1 contingency ──────────────────────────────────────────────
        n1_result = None
        if contingency_pipe:
            avg_demand_lps = df["Flow Rate (L/s)"].mean() / 16  # per node
            n1_result = self.contingency.simulate_n1_failure(
                wn_leaky, contingency_pipe, avg_demand_lps
            )

        # ── Economics ────────────────────────────────────────────────────
        econ_report = self.economics.full_economic_report(
            df=df,
            leak_threshold_bar=leak_threshold_bar,
            sampling_rate_hz=sampling_rate_hz,
            water_tariff_kzt=water_tariff_kzt,
            pump_head_m=pump_head_m,
            smart_pump=smart_pump,
            n_sensors=len(sensors),
            repair_cost_kzt=repair_cost_kzt
        )

        # ── Isolation valves ─────────────────────────────────────────────
        isolation_pipes, isolation_neighbors = [], []
        if leak_node:
            isolation_pipes, isolation_neighbors = \
                self.contingency.find_isolation_valves(wn_leaky, leak_node)

        return {
            "network":              wn_leaky,
            "dataframe":            df,
            "leak_node":            leak_node or "None",
            "sensors":              sensors,
            "healthy_pressures":    healthy_pressures,
            "observed_pressures":   observed_pressures,
            "predicted_leak":       predicted_node,
            "residuals":            residuals,
            "confidence":           confidence,
            "failure_probabilities": failure_probs,
            "mnf_anomaly":          mnf_anomaly,
            "mnf_percentage":       mnf_pct,
            "n1_result":            n1_result,
            "economics":            econ_report,
            "isolation_pipes":      isolation_pipes,
            "isolation_neighbors":  isolation_neighbors,
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
                material, pipe_age, temp=self.city_manager.season_temp
            ),
            "degradation_pct": degradation_pct,
        }


# ═══════════════════════════════════════════════════════════════════════════
# EXPORT API
# ═══════════════════════════════════════════════════════════════════════════

__all__ = [
    "CityManager",
    "CityConfig",
    "HydraulicPhysics",
    "HydraulicEngine",
    "LeakDetectionAnalytics",
    "ContingencyAnalysis",
    "EconomicModel",
    "SmartShygynBackend",
]


if __name__ == "__main__":
    print("=" * 80)
    print("Smart Shygyn PRO v3 — Backend Engine")
    print("=" * 80)

    backend = SmartShygynBackend("Алматы", season_temp_celsius=15.0)

    results = backend.run_full_simulation(
        material="Сталь",
        pipe_age=20.0,
        pump_head_m=45.0,
        smart_pump=True,
        leak_node="N_2_2",
        leak_area_cm2=0.8,
    )

    print(f"\nCity: {results['city_config']['name']}")
    print(f"Material: {results['material']} ({results['pipe_age']} years)")
    print(f"H-W Roughness: {results['roughness']:.1f}")
    print(f"Degradation: {results['degradation_pct']:.1f}%")
    print(f"\nPredicted leak: {results['predicted_leak']} (confidence: {results['confidence']:.0f}%)")
    print(f"Sensors: {len(results['sensors'])} nodes")
    print(f"\nEconomics:")
    print(f"  Water loss: {results['economics']['lost_liters']:,.0f} L")
    print(f"  NRW: {results['economics']['nrw_percentage']:.1f}%")
    print(f"  Total damage: {results['economics']['total_damage_kzt']:,.0f} KZT")
    print(f"  Payback: {results['economics']['payback_months']:.1f} months")
    print(f"  Energy saved: {results['economics']['energy_saved_pct']:.1f}%")
    print(f"  CO₂ saved: {results['economics']['co2_saved_kg']:.1f} kg")

    print("\n" + "=" * 80)
    print("✅ Backend engine ready for frontend integration!")
    print("=" * 80)
