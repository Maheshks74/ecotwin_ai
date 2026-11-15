"""Gymnasium environment for EcoTwin industrial energy optimization."""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from ecotwin.twin.factory import FactoryFloorSim, StepResult
from ecotwin.data.tariff import TariffManager
from ecotwin.data.weather import WeatherManager
from ecotwin.data.scenarios import ScenarioManager


class EcoTwinEnv(gym.Env):
    """Gymnasium Environment wrapping the EcoTwin factory digital twin.

    Observation space: Box(58,)
      - Time of day sin/cos: [2]
      - Current price (INR/kWh normalized): [1]
      - Next-6h price forecast: [24]
      - Outside temperature (deg C normalized): [1]
      - Next-6h temperature forecast: [24]
      - Zone temperatures (deg C normalized): [2]
      - Queue length (normalized): [1]
      - Min slack to deadline (normalized): [1]
      - Jobs completed vs quota ratio: [1]
      - Total current power (normalized): [1]

    Action space: MultiDiscrete([2, 2, 2, 2, 2, 2, 5, 5])
      - 6 machines: {0 = HOLD, 1 = START}
      - 2 HVAC zones: {0..4 setpoint level}
    """

    metadata = {"render_modes": ["human"]}

    DEFAULT_REWARD_WEIGHTS = {
        "energy_cost_weight": 1.0,
        "demand_charge_weight": 1.0,
        "comfort_violation_weight": 60.0,
        "quota_penalty_weight": 600.0,
        "late_penalty_weight": 80.0,
        "reward_scale": 0.005, # Scaling factor for stable RL gradients
    }

    def __init__(
        self,
        config_path: Optional[str] = None,
        scenario_name: Optional[str] = None,
        randomize_scenarios: bool = False,
        reward_weights: Optional[Dict[str, float]] = None,
        include_weather_forecast: bool = True
    ):
        super().__init__()
        self.sim = FactoryFloorSim(config_path)
        self.tariff_mgr = TariffManager()
        self.weather_mgr = WeatherManager()
        self.scenario_mgr = ScenarioManager()

        self.scenario_name = scenario_name or "summer_day"
        self.randomize_scenarios = randomize_scenarios
        self.include_weather_forecast = include_weather_forecast
        self.forecast_horizon = 24 # 6 hours = 24 * 15 min

        # Reward weights (can be updated dynamically by LLM instructions)
        self.reward_weights = dict(self.DEFAULT_REWARD_WEIGHTS)
        if reward_weights:
            self.reward_weights.update(reward_weights)

        # Action Space: 6 machines (binary 0/1) + 2 zones (5 levels 0..4)
        self.action_space = spaces.MultiDiscrete([2, 2, 2, 2, 2, 2, 5, 5])

        # Observation Space: 58 dimensions
        # Time(2) + Price(1) + PriceFC(24) + OutT(1) + OutTFC(24) + ZoneT(2) + Queue(1) + Slack(1) + Quota(1) + Power(1) = 58
        self.obs_dim = 58
        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(self.obs_dim,),
            dtype=np.float32
        )

        self.machine_ids = sorted(list(self.sim.machines.keys())) # ["M1", "M2", "M3", "M4", "M5", "M6"]
        self.zone_ids = sorted(list(self.sim.zones.keys())) # ["zone_1", "zone_2"]

        self.current_step = 0
        self.current_seed: Optional[int] = None
        self.last_result: Optional[StepResult] = None
        self.np_random = np.random.default_rng()

    def set_reward_weights(self, weights: Dict[str, float]):
        """Dynamically update reward weights (used by LLM instruction parser)."""
        for k, v in weights.items():
            if k in self.reward_weights:
                self.reward_weights[k] = float(v)

    def _get_obs(self) -> np.ndarray:
        step = self.sim.current_step
        # 1. Time of day cyclic features: sin and cos
        fraction_of_day = (step % 96) / 96.0
        sin_time = float(np.sin(2 * np.pi * fraction_of_day))
        cos_time = float(np.cos(2 * np.pi * fraction_of_day))

        # 2. Tariff: current price and forecast
        curr_price = self.tariff_mgr.get_price(step)
        norm_price = (curr_price - 8.0) / 6.0 # centered around typical price
        price_fc = self.tariff_mgr.get_price_forecast(step, self.forecast_horizon)
        norm_price_fc = (price_fc - 8.0) / 6.0

        # 3. Weather: current temperature and forecast
        curr_temp = self.weather_mgr.get_temperature(step)
        norm_temp = (curr_temp - 25.0) / 10.0

        if self.include_weather_forecast:
            # Deterministic forecast jitter for reproducibility
            fc_seed = int((self.current_seed or 42) * 1000 + step)
            weather_fc = self.weather_mgr.get_weather_forecast(step, self.forecast_horizon, seed=fc_seed)
            norm_weather_fc = (weather_fc - 25.0) / 10.0
        else:
            # Ablation mode: zero out weather forecast
            norm_weather_fc = np.zeros(self.forecast_horizon, dtype=np.float32)

        # 4. Zone temperatures
        z_temps = [
            (self.sim.zones[zid].current_temp_c - 24.0) / 4.0
            for zid in self.zone_ids
        ]

        # 5. Factory queue & progress states
        pending_count = len(self.sim.job_queue.pending_queue)
        norm_queue = (pending_count - 15.0) / 15.0

        min_slack = self.sim.job_queue.min_slack(step)
        norm_slack = np.clip(min_slack / 48.0, -2.0, 2.0)

        quota_ratio = (self.sim.job_queue.num_completed / max(1, self.sim.quota)) - 0.5

        # 6. Floor power
        total_p = self.last_result.total_floor_power_kw if self.last_result else 20.0
        norm_power = (total_p - 60.0) / 60.0

        obs = np.concatenate([
            np.array([sin_time, cos_time, norm_price], dtype=np.float32),
            norm_price_fc.astype(np.float32),
            np.array([norm_temp], dtype=np.float32),
            norm_weather_fc.astype(np.float32),
            np.array(z_temps, dtype=np.float32),
            np.array([norm_queue, norm_slack, quota_ratio, norm_power], dtype=np.float32),
        ])

        return np.clip(obs, -5.0, 5.0)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.current_seed = seed
            self.np_random = np.random.default_rng(seed)

        # Select scenario
        if options and "scenario" in options:
            self.scenario_name = options["scenario"]
        elif self.randomize_scenarios:
            all_scens = self.scenario_mgr.list_scenarios()
            self.scenario_name = self.np_random.choice(all_scens)

        scen_dict = self.scenario_mgr.get_scenario(self.scenario_name)
        self.tariff_mgr.set_scenario_tariff(scen_dict)
        self.weather_mgr.set_scenario_weather(scen_dict, seed=seed)

        quota = scen_dict.get("quota", self.sim.config.get("default_daily_quota", 30))
        base_jobs = self.sim.generate_default_jobs(seed=seed, quota=quota)
        scenario_jobs = self.scenario_mgr.build_scenario_jobs(self.scenario_name, base_jobs)

        self.sim.reset(seed=seed, quota=quota, custom_jobs=scenario_jobs)
        self.current_step = 0
        self.last_result = None

        obs = self._get_obs()
        info = {
            "scenario": self.scenario_name,
            "quota": quota,
            "step": 0,
            "state_summary": self.sim.get_state_summary()
        }
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Map actions
        # First 6 elements: machines M1..M6
        m_actions = {
            mid: int(action[i])
            for i, mid in enumerate(self.machine_ids)
        }
        # Last 2 elements: HVAC zones
        h_actions = {
            zid: int(action[len(self.machine_ids) + j])
            for j, zid in enumerate(self.zone_ids)
        }

        # Step twin
        curr_price = self.tariff_mgr.get_price(self.sim.current_step)
        curr_temp = self.weather_mgr.get_temperature(self.sim.current_step)
        result = self.sim.step(
            machine_actions=m_actions,
            hvac_actions=h_actions,
            price_per_kwh=curr_price,
            outside_temp_c=curr_temp
        )
        self.last_result = result
        self.current_step += 1

        terminated = self.sim.current_step >= self.sim.episode_steps
        truncated = False

        # Calculate reward
        w_energy = self.reward_weights["energy_cost_weight"]
        w_comfort = self.reward_weights["comfort_violation_weight"]
        w_demand = self.reward_weights["demand_charge_weight"]
        w_quota = self.reward_weights["quota_penalty_weight"]
        w_late = self.reward_weights["late_penalty_weight"]
        scale = self.reward_weights["reward_scale"]

        step_cost_penalty = result.step_cost_inr * w_energy
        comfort_penalty = result.total_comfort_violation * w_comfort

        terminal_demand_penalty = 0.0
        terminal_quota_penalty = 0.0
        terminal_late_penalty = 0.0

        if terminated:
            terminal_demand_penalty = self.tariff_mgr.compute_demand_charge(result.peak_power_kw) * w_demand
            missed_jobs = max(0, result.quota - result.completed_jobs_count)
            terminal_quota_penalty = missed_jobs * w_quota
            terminal_late_penalty = result.late_jobs_count * w_late

        raw_step_penalty = (
            step_cost_penalty
            + comfort_penalty
            + terminal_demand_penalty
            + terminal_quota_penalty
            + terminal_late_penalty
        )
        # Reward is negative penalty
        reward = float(-raw_step_penalty * scale)

        obs = self._get_obs()
        info = {
            "step": result.step,
            "cost_inr": result.step_cost_inr,
            "cumulative_cost_inr": result.cumulative_cost_inr,
            "cumulative_kwh": result.cumulative_energy_kwh,
            "peak_kw": result.peak_power_kw,
            "completed_jobs": result.completed_jobs_count,
            "quota": result.quota,
            "comfort_violations": result.total_comfort_violation,
            "late_jobs": result.late_jobs_count,
            "reward_breakdown": {
                "energy_cost": -step_cost_penalty * scale,
                "comfort_violation": -comfort_penalty * scale,
                "demand_charge": -terminal_demand_penalty * scale,
                "quota_penalty": -terminal_quota_penalty * scale,
                "late_penalty": -terminal_late_penalty * scale,
            }
        }

        return obs, reward, terminated, truncated, info
