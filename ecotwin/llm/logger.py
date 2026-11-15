"""Telemetry logger for EcoTwin digital twin runs."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

from ecotwin.twin.factory import StepResult


class FloorLogger:
    """Logs detailed step-by-step state, action, price, weather, and component metrics to Parquet / CSV."""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.records: List[Dict[str, Any]] = []

    def log_step(
        self,
        result: StepResult,
        action: np.ndarray,
        outside_temp: float,
        price_per_kwh: float,
        reward: float,
        reward_breakdown: Optional[Dict[str, float]] = None,
        scenario_name: str = "default"
    ):
        """Append one time step telemetry row."""
        rec = {
            "step": result.step,
            "hour": result.step * 0.25,
            "scenario": scenario_name,
            "price_inr_per_kwh": price_per_kwh,
            "outside_temp_c": outside_temp,
            "total_power_kw": result.total_floor_power_kw,
            "machines_power_kw": result.machines_power_kw,
            "hvac_power_kw": result.hvac_power_kw,
            "step_energy_kwh": result.step_energy_kwh,
            "step_cost_inr": result.step_cost_inr,
            "cumulative_cost_inr": result.cumulative_cost_inr,
            "cumulative_energy_kwh": result.cumulative_energy_kwh,
            "peak_power_kw": result.peak_power_kw,
            "zone_1_temp_c": result.zone_temperatures.get("zone_1", 24.0),
            "zone_2_temp_c": result.zone_temperatures.get("zone_2", 24.0),
            "total_comfort_violation": result.total_comfort_violation,
            "completed_jobs": result.completed_jobs_count,
            "quota": result.quota,
            "min_slack": result.min_slack,
            "late_jobs": result.late_jobs_count,
            "action_m1": int(action[0]) if len(action) > 0 else 0,
            "action_m2": int(action[1]) if len(action) > 1 else 0,
            "action_m3": int(action[2]) if len(action) > 2 else 0,
            "action_m4": int(action[3]) if len(action) > 3 else 0,
            "action_m5": int(action[4]) if len(action) > 4 else 0,
            "action_m6": int(action[5]) if len(action) > 5 else 0,
            "action_hvac_z1": int(action[6]) if len(action) > 6 else 2,
            "action_hvac_z2": int(action[7]) if len(action) > 7 else 2,
            "reward": reward,
        }

        # Include machine states
        for mid, state in result.machine_states.items():
            rec[f"state_{mid}"] = state

        if reward_breakdown:
            for k, v in reward_breakdown.items():
                rec[f"reward_{k}"] = v

        self.records.append(rec)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)

    def save(self, filename: str = "floor_log.parquet") -> str:
        """Saves telemetry to parquet (or fallback to CSV)."""
        df = self.to_dataframe()
        out_path = self.log_dir / filename
        try:
            df.to_parquet(out_path, index=False)
        except Exception:
            # Fallback to CSV if parquet engine not available
            csv_path = out_path.with_suffix(".csv")
            df.to_csv(csv_path, index=False)
            return str(csv_path)
        return str(out_path)

    def load(self, filepath: str) -> pd.DataFrame:
        path = Path(filepath)
        if not path.exists():
            path = self.log_dir / filepath
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)

    def clear(self):
        self.records.clear()

