"""FactoryFloorSim: SimPy-based Digital Twin of the manufacturing floor."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import simpy
import yaml
import numpy as np

from ecotwin.twin.machines import Machine, MachineState
from ecotwin.twin.jobs import Job, JobQueue
from ecotwin.twin.hvac import HVACZone


@dataclass
class StepResult:
    """Record of all state variables and metrics after executing a 15-minute step."""
    step: int
    dt_hours: float
    total_floor_power_kw: float
    machines_power_kw: float
    hvac_power_kw: float
    step_energy_kwh: float
    price_per_kwh: float
    step_cost_inr: float
    cumulative_cost_inr: float
    cumulative_energy_kwh: float
    peak_power_kw: float
    zone_temperatures: Dict[str, float]
    comfort_violations: Dict[str, float]
    total_comfort_violation: float
    machine_states: Dict[str, str]
    machine_powers_kw: Dict[str, float]
    completed_jobs_count: int
    quota: int
    min_slack: float
    late_jobs_count: int


class FactoryFloorSim:
    """SimPy-based Industrial Manufacturing Digital Twin."""

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.dt_hours: float = float(self.config.get("time_step_hours", 0.25))
        self.episode_steps: int = int(self.config.get("episode_steps", 96))
        self.quota: int = int(self.config.get("default_daily_quota", 30))

        # Build physical components
        self.zones: Dict[str, HVACZone] = self._build_zones()
        self.machines: Dict[str, Machine] = self._build_machines()
        self.job_queue: JobQueue = JobQueue(quota=self.quota)

        # SimPy env and state trackers
        self.sim_env: Optional[simpy.Environment] = None
        self.current_step: int = 0
        self.peak_power_kw: float = 0.0
        self.cumulative_cost_inr: float = 0.0
        self.cumulative_energy_kwh: float = 0.0
        self.history: List[StepResult] = []

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        if config_path is not None and Path(config_path).exists():
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)

        # Fallback candidates
        candidates = [
            Path("config/factory.yaml"),
            Path("ecotwin/config/factory.yaml"),
            Path(__file__).parent.parent / "config" / "factory.yaml",
        ]
        for c in candidates:
            if c.exists():
                with open(c, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f)

        raise FileNotFoundError("Could not find factory.yaml config file.")

    def _build_zones(self) -> Dict[str, HVACZone]:
        zones = {}
        for zid, zcfg in self.config.get("zones", {}).items():
            zones[zid] = HVACZone(
                zone_id=zid,
                name=zcfg.get("name", zid),
                thermal_resistance_R=float(zcfg.get("thermal_resistance_R", 2.0)),
                thermal_capacitance_C=float(zcfg.get("thermal_capacitance_C", 20.0)),
                target_temp_c=float(zcfg.get("target_temp_c", 24.0)),
                comfort_min_c=float(zcfg.get("comfort_min_c", 22.0)),
                comfort_max_c=float(zcfg.get("comfort_max_c", 26.0)),
                initial_temp_c=float(zcfg.get("initial_temp_c", 24.0)),
                cop_nominal=float(zcfg.get("cop_nominal", 3.5)),
                cop_temp_coefficient=float(zcfg.get("cop_temp_coefficient", 0.04)),
                cop_min=float(zcfg.get("cop_min", 1.5)),
                cop_max=float(zcfg.get("cop_max", 4.5)),
                max_cooling_kw=float(zcfg.get("max_cooling_kw", 40.0)),
                setpoint_options=list(zcfg.get("setpoint_options", [22.0, 23.0, 24.0, 25.0, 26.0])),
            )
        return zones

    def _build_machines(self) -> Dict[str, Machine]:
        machines = {}
        for mcfg in self.config.get("machines", []):
            mid = mcfg["id"]
            machines[mid] = Machine(
                id=mid,
                name=mcfg.get("name", mid),
                zone_id=mcfg.get("zone_id", "zone_1"),
                rated_power_kw=float(mcfg.get("rated_power_kw", 15.0)),
                idle_power_fraction=float(mcfg.get("idle_power_fraction", 0.2)),
                heat_emission_fraction=float(mcfg.get("heat_emission_fraction", 0.75)),
                processing_steps=dict(mcfg.get("processing_steps", {"TypeA": 2, "TypeB": 2, "TypeC": 2})),
            )
        return machines

    def generate_default_jobs(self, seed: Optional[int] = None, quota: Optional[int] = None) -> List[Job]:
        """Generates synthetic initial jobs for a 24h episode."""
        rng = np.random.default_rng(seed)
        q = quota if quota is not None else self.quota
        # Create slightly more jobs than quota (e.g. quota + 10) so the agent can select
        total_pool = q + 10
        job_types = list(self.config.get("job_types", {}).keys()) or ["TypeA", "TypeB", "TypeC"]
        jobs = []

        for i in range(total_pool):
            jtype = rng.choice(job_types)
            type_cfg = self.config.get("job_types", {}).get(jtype, {})
            base_deadline = type_cfg.get("default_deadline_step", 72)
            # Stagger release steps across first half of the day (steps 0 to 48)
            release = int(rng.integers(0, 36)) if i >= 10 else 0
            # Deadline with some jitter
            jitter = int(rng.integers(-8, 12))
            due = int(np.clip(base_deadline + jitter, release + 8, self.episode_steps - 1))
            priority = float(type_cfg.get("base_priority", 1.0))

            jobs.append(
                Job(
                    job_id=f"Job_{i+1:03d}",
                    job_type=jtype,
                    release_step=release,
                    due_step=due,
                    duration_steps=2, # Will be adapted per machine processing speed
                    priority=priority,
                )
            )
        return jobs

    def reset(
        self,
        seed: Optional[int] = None,
        quota: Optional[int] = None,
        custom_jobs: Optional[List[Job]] = None,
        initial_temps: Optional[Dict[str, float]] = None
    ):
        """Reset the factory digital twin to the start of an episode."""
        self.sim_env = simpy.Environment()
        self.current_step = 0
        self.peak_power_kw = 0.0
        self.cumulative_cost_inr = 0.0
        self.cumulative_energy_kwh = 0.0
        self.history.clear()

        if quota is not None:
            self.quota = quota

        # Reset zones
        for zid, zone in self.zones.items():
            init_t = initial_temps.get(zid) if initial_temps else None
            zone.reset(initial_temp=init_t)

        # Reset machines
        for machine in self.machines.values():
            machine.reset()

        # Reset job queue
        jobs = custom_jobs if custom_jobs is not None else self.generate_default_jobs(seed=seed, quota=self.quota)
        self.job_queue.reset(initial_jobs=jobs, quota=self.quota)

        return self.get_state_summary()

    def step(
        self,
        machine_actions: Dict[str, int],
        hvac_actions: Dict[str, int],
        price_per_kwh: float,
        outside_temp_c: float
    ) -> StepResult:
        """Step the digital twin forward by one 15-minute time step.

        Parameters
        ----------
        machine_actions : Dict[str, int]
            Map of machine_id -> action (0 = HOLD, 1 = START).
        hvac_actions : Dict[str, int]
            Map of zone_id -> setpoint_option index (0..4).
        price_per_kwh : float
            Current electricity tariff (INR/kWh).
        outside_temp_c : float
            Current ambient temperature (deg C).
        """
        # 1. Step Machines and collect heat by zone
        zone_heat: Dict[str, float] = {zid: 0.0 for zid in self.zones}
        total_machines_power_kw = 0.0
        machine_states = {}
        machine_powers = {}

        for mid, machine in self.machines.items():
            act = machine_actions.get(mid, 1) # default to start
            kwh, heat_kw = machine.step(
                action=act,
                current_step=self.current_step,
                job_queue=self.job_queue,
                dt_hours=self.dt_hours
            )
            zone_heat[machine.zone_id] += heat_kw
            total_machines_power_kw += machine.step_power_kw
            machine_states[mid] = machine.state.value
            machine_powers[mid] = machine.step_power_kw

        # 2. Step HVAC zones with machine heat loads
        total_hvac_power_kw = 0.0
        zone_temps = {}
        zone_violations = {}

        for zid, zone in self.zones.items():
            sp_idx = hvac_actions.get(zid, 2) # default to index 2 (24 C)
            new_t, cooling_kw, viol = zone.step(
                setpoint_idx=sp_idx,
                q_machines_kw=zone_heat[zid],
                outside_temp_c=outside_temp_c,
                dt_hours=self.dt_hours
            )
            total_hvac_power_kw += cooling_kw
            zone_temps[zid] = new_t
            zone_violations[zid] = viol

        # 3. Floor-wide totals and electrical cost
        floor_power_kw = total_machines_power_kw + total_hvac_power_kw
        step_kwh = floor_power_kw * self.dt_hours
        step_cost = step_kwh * price_per_kwh

        self.cumulative_energy_kwh += step_kwh
        self.cumulative_cost_inr += step_cost
        if floor_power_kw > self.peak_power_kw:
            self.peak_power_kw = floor_power_kw

        # Advance SimPy environment time
        if self.sim_env is not None:
            self.sim_env.run(until=(self.current_step + 1) * self.dt_hours)

        total_comfort_violation = float(sum(zone_violations.values()))

        result = StepResult(
            step=self.current_step,
            dt_hours=self.dt_hours,
            total_floor_power_kw=float(floor_power_kw),
            machines_power_kw=float(total_machines_power_kw),
            hvac_power_kw=float(total_hvac_power_kw),
            step_energy_kwh=float(step_kwh),
            price_per_kwh=float(price_per_kwh),
            step_cost_inr=float(step_cost),
            cumulative_cost_inr=float(self.cumulative_cost_inr),
            cumulative_energy_kwh=float(self.cumulative_energy_kwh),
            peak_power_kw=float(self.peak_power_kw),
            zone_temperatures=zone_temps,
            comfort_violations=zone_violations,
            total_comfort_violation=total_comfort_violation,
            machine_states=machine_states,
            machine_powers_kw=machine_powers,
            completed_jobs_count=self.job_queue.num_completed,
            quota=self.quota,
            min_slack=self.job_queue.min_slack(self.current_step),
            late_jobs_count=self.job_queue.late_count(self.current_step),
        )

        self.history.append(result)
        self.current_step += 1
        return result

    def get_state_summary(self) -> Dict[str, Any]:
        """Summary dictionary of current plant state."""
        return {
            "step": self.current_step,
            "quota": self.quota,
            "jobs_completed": self.job_queue.num_completed,
            "cumulative_kwh": self.cumulative_energy_kwh,
            "cumulative_cost": self.cumulative_cost_inr,
            "peak_kw": self.peak_power_kw,
            "zone_temps": {zid: z.current_temp_c for zid, z in self.zones.items()},
        }

