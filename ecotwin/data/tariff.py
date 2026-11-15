"""Tariff manager: TOU slabs, IEX Day-Ahead loader, and demand charges."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
import numpy as np
import pandas as pd


class TariffManager:
    """Manages electricity tariffs, time-of-use pricing slabs, IEX prices, and demand charges."""

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.demand_charge_inr_per_kw = float(
            self.config.get("tou", {}).get("demand_charge_inr_per_kw", 12.0)
        )
        self.episode_steps = 96
        self.prices_inr_per_kwh = self._build_default_tou_profile()

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        if config_path is not None and Path(config_path).exists():
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)

        candidates = [
            Path("config/tariff.yaml"),
            Path("ecotwin/config/tariff.yaml"),
            Path(__file__).parent.parent / "config" / "tariff.yaml",
        ]
        for c in candidates:
            if c.exists():
                with open(c, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f)

        return {"tou": {"demand_charge_inr_per_kw": 12.0}}

    def _build_default_tou_profile(self) -> np.ndarray:
        """Constructs 96-step price array according to TOU slabs in tariff.yaml."""
        prices = np.full(self.episode_steps, 7.2) # Default normal rate
        slabs = self.config.get("tou", {}).get("slabs", [])

        for slab in slabs:
            s_hr = slab.get("start_hour", 0)
            e_hr = slab.get("end_hour", 24)
            rate = float(slab.get("rate_inr_per_kwh", 7.2))

            if s_hr < e_hr:
                start_step = s_hr * 4
                end_step = e_hr * 4
                prices[start_step:end_step] = rate
            else:
                # Wraps around midnight (e.g. 22 to 6)
                start_step = s_hr * 4
                end_step = e_hr * 4
                prices[start_step:self.episode_steps] = rate
                prices[0:end_step] = rate

        return prices

    def load_iex_csv(self, csv_path: str) -> np.ndarray:
        """Loads IEX Day-Ahead prices from CSV and interpolates hourly to 96 steps."""
        path = Path(csv_path)
        if not path.exists():
            # Try relative to project root
            path = Path(__file__).parent.parent.parent / csv_path
        if not path.exists():
            return self._build_default_tou_profile()

        df = pd.read_csv(path)
        if "price_inr_per_kwh" in df.columns:
            hourly = df["price_inr_per_kwh"].values
        elif "price" in df.columns:
            hourly = df["price"].values
        else:
            hourly = df.iloc[:, -1].values

        if len(hourly) == 24:
            # Linear interpolation across 24 hourly centers to 96 steps
            orig_steps = np.linspace(0, 95, 24)
            target_steps = np.arange(96)
            interp_prices = np.interp(target_steps, orig_steps, hourly)
            self.prices_inr_per_kwh = interp_prices.astype(float)
        elif len(hourly) == 96:
            self.prices_inr_per_kwh = hourly.astype(float)
        else:
            self.prices_inr_per_kwh = self._build_default_tou_profile()

        return self.prices_inr_per_kwh

    def set_scenario_tariff(self, scenario_dict: Dict[str, Any]):
        """Adjusts tariff array for special scenarios (e.g., price spike, custom slabs)."""
        tariff_cfg = scenario_dict.get("tariff", {})
        t_type = tariff_cfg.get("type", "tou")

        if t_type == "spike":
            self.prices_inr_per_kwh = self._build_default_tou_profile()
            s_step = int(tariff_cfg.get("spike_start_step", 48))
            e_step = int(tariff_cfg.get("spike_end_step", 64))
            spike_rate = float(tariff_cfg.get("spike_rate_inr", 18.0))
            self.prices_inr_per_kwh[s_step:e_step] = spike_rate
        elif t_type == "iex":
            csv_path = tariff_cfg.get("csv_path", "data/sample_iex_prices.csv")
            self.load_iex_csv(csv_path)
        else:
            self.prices_inr_per_kwh = self._build_default_tou_profile()

        multiplier = float(tariff_cfg.get("demand_charge_multiplier", 1.0))
        self.demand_charge_inr_per_kw = float(
            self.config.get("tou", {}).get("demand_charge_inr_per_kw", 12.0)
        ) * multiplier

    def get_price(self, step: int) -> float:
        """Returns tariff price (INR/kWh) for given step index."""
        safe_step = int(np.clip(step, 0, self.episode_steps - 1))
        return float(self.prices_inr_per_kwh[safe_step])

    def get_price_forecast(self, current_step: int, horizon_steps: int = 24) -> np.ndarray:
        """Returns price forecast for next `horizon_steps` (default 24 steps = 6 hours)."""
        forecast = np.zeros(horizon_steps, dtype=np.float32)
        for i in range(horizon_steps):
            target_step = min(current_step + i, self.episode_steps - 1)
            forecast[i] = self.prices_inr_per_kwh[target_step]
        return forecast

    def compute_demand_charge(self, peak_kw: float) -> float:
        """Calculates demand charge in INR based on peak kW drawn during episode."""
        return float(max(0.0, peak_kw) * self.demand_charge_inr_per_kw)

