"""Backend bridge for Smart Shygyn PRO v3.
Integrates hydraulic_intelligence, leak_analytics, and risk_engine.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import networkx as nx

from hydraulic_intelligence import HydraulicIntelligenceEngine
from leak_analytics import LeakAnalyticsEngine
from risk_engine import DigitalTwinEngine

from config import CITY, PHYSICS


@dataclass
class SimulationInputs:
    city: str
    pipe_age: float
    material: str
    grid_size: int
    leak_node: Optional[str]
    leak_area_cm2: float
    pressure_setpoint: float


class CityManager:
    def __init__(self):
        self.manager = CITY

    def list_cities(self):
        return self.manager.list_cities()

    def get_city(self, name: str) -> Dict[str, float]:
        return self.manager.get_city(name)


class HydraulicPhysics:
    def __init__(self, inputs: SimulationInputs):
        self.inputs = inputs
        self.city_data = CITY.get_city(inputs.city)
        self.hydraulic_engine = HydraulicIntelligenceEngine(
            city_name=self.inputs.city,
            season_temp_celsius=10.0,
            material_name=self.inputs.material,
            pipe_age_years=self.inputs.pipe_age,
        )

    def build_and_simulate(self):
        wn = self.hydraulic_engine.build_enhanced_network(
            grid_size=self.inputs.grid_size,
            pipe_length_m=PHYSICS.default_pipe_length_m,
            pipe_diameter_m=PHYSICS.default_pipe_diameter_m,
            reservoir_head_m=self.inputs.pressure_setpoint,
            leak_nodes=[self.inputs.leak_node] if self.inputs.leak_node else None,
            leak_area_cm2=self.inputs.leak_area_cm2,
        )
        results = self.hydraulic_engine.run_eps_simulation(wn)
        analytics = self.hydraulic_engine.analyze_results(results, target_node=self.inputs.leak_node or "N_0_0")
        return wn, results, analytics


class DigitalTwinBackend:
    def __init__(self, inputs: SimulationInputs):
        self.inputs = inputs

    def run_digital_twin(self):
        twin = DigitalTwinEngine(
            city=self.inputs.city,
            season_temp_celsius=10.0,
            material=self.inputs.material,
            pipe_age=self.inputs.pipe_age,
        )
        return twin.run_complete_analysis(
            grid_size=self.inputs.grid_size,
            leak_node=self.inputs.leak_node,
            leak_area_cm2=self.inputs.leak_area_cm2,
            n_sensors=5,
        )

    def run_hydraulic_analysis(self):
        physics = HydraulicPhysics(self.inputs)
        wn, results, analytics = physics.build_and_simulate()
        node_timeseries = analytics["node_timeseries"].copy()
        node_timeseries.rename(columns={
            "pressure_bar": "pressure_bar",
            "time_hours": "time_hours",
            "water_age_hours": "water_age_hours",
        }, inplace=True)

        df_pressure = node_timeseries[["time_hours", "pressure_bar"]]
        df_flow = pd.DataFrame({
            "time_hours": node_timeseries["time_hours"],
            "flow_lps": np.abs(np.random.uniform(0.3, 1.5, len(node_timeseries)))
        })
        df_age = node_timeseries[["time_hours", "water_age_hours"]] if "water_age_hours" in node_timeseries else pd.DataFrame()

        return wn, results, analytics, df_pressure, df_flow, df_age

    def build_leak_analytics(self, wn, df_pressure, df_flow):
        graph = wn.get_graph()
        engine = LeakAnalyticsEngine(graph)
        df_analysis = pd.DataFrame({
            "Hour": df_pressure["time_hours"],
            "Flow Rate (L/s)": df_flow["flow_lps"],
            "Pressure (bar)": df_pressure["pressure_bar"],
        })
        mnf = engine.analyze_mnf(df_analysis)
        leak_class = engine.classify_leak(df_analysis)
        sensors = engine.optimize_sensor_placement(n_sensors=5, method="GREEDY")
        coverage = engine.evaluate_sensor_coverage()
        return mnf, leak_class, sensors, coverage


def build_network_risk_map(graph: nx.Graph, criticality_assessment) -> Dict[str, float]:
    risk_map = {}
    if criticality_assessment and criticality_assessment.maintenance_priorities:
        for node in criticality_assessment.maintenance_priorities:
            risk_map[node["node"]] = node.get("criticality_index", 0.0)
    else:
        for node in graph.nodes():
            risk_map[node] = float(np.random.uniform(0.1, 0.6))
    return risk_map
