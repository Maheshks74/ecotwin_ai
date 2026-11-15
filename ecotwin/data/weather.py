"""Weather manager: Open-Meteo API loader, CSV caching, synthetic generator, noisy forecast."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import requests
import numpy as np
import pandas as pd


class WeatherManager:
    """Fetches outdoor temperature from Open-Meteo API, caches to CSV, and generates noisy forecasts."""

    def __init__(
        self,
        cache_dir: Optional[str] = "data/cache",
        latitude: float = 18.5204, # Pune/Pimpri-Chinchwad manufacturing hub
        longitude: float = 73.8567,
        random_seed: Optional[int] = None
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.latitude = latitude
        self.longitude = longitude
        self.rng = np.random.default_rng(random_seed)
        self.episode_steps = 96
        self.temps_c = self.generate_synthetic_profile(mean_temp_c=28.0, amplitude_temp_c=6.0)

    def generate_synthetic_profile(
        self,
        mean_temp_c: float = 28.0,
        amplitude_temp_c: float = 6.0,
        peak_hour: float = 14.0,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """Generates realistic diurnal temperature cycle for 96 steps (24 hours)."""
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        steps = np.arange(self.episode_steps)
        hours = steps * 0.25 # 0.0 to 23.75

        # Diurnal curve: peak at peak_hour, minimum ~peak_hour - 12
        # cos((hour - peak_hour) * 2 * pi / 24)
        diurnal = amplitude_temp_c * np.cos((hours - peak_hour) * 2 * np.pi / 24.0)
        # Small smooth stochastic noise
        smooth_noise = rng.normal(0, 0.4, size=self.episode_steps)
        temps = mean_temp_c + diurnal + smooth_noise
        return temps.astype(np.float32)

    def fetch_open_meteo(self, timeout_sec: float = 4.0) -> np.ndarray:
        """Attempts to load hourly temperature from Open-Meteo API, caches to CSV, or falls back."""
        cache_file = self.cache_dir / f"weather_lat{self.latitude:.2f}_lon{self.longitude:.2f}.csv"

        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={self.latitude}&longitude={self.longitude}"
            f"&hourly=temperature_2m&forecast_days=1&timezone=auto"
        )
        try:
            resp = requests.get(url, timeout=timeout_sec)
            if resp.status_code == 200:
                data = resp.json()
                hourly_temps = data.get("hourly", {}).get("temperature_2m", [])
                if len(hourly_temps) >= 24:
                    # Save to cache
                    df = pd.DataFrame({
                        "hour": np.arange(24),
                        "temp_c": hourly_temps[:24]
                    })
                    df.to_csv(cache_file, index=False)
                    return self._interpolate_24h_to_96(np.array(hourly_temps[:24], dtype=float))
        except Exception:
            # Offline or timeout: check cache
            pass

        # Try cache
        if cache_file.exists():
            try:
                df = pd.read_csv(cache_file)
                if "temp_c" in df.columns and len(df) >= 24:
                    return self._interpolate_24h_to_96(df["temp_c"].values[:24].astype(float))
            except Exception:
                pass

        # Synthetic fallback
        return self.generate_synthetic_profile(mean_temp_c=28.0, amplitude_temp_c=6.0)

    def _interpolate_24h_to_96(self, hourly_array: np.ndarray) -> np.ndarray:
        orig_steps = np.linspace(0, 95, 24)
        target_steps = np.arange(96)
        interp_temps = np.interp(target_steps, orig_steps, hourly_array)
        return interp_temps.astype(np.float32)

    def set_scenario_weather(self, scenario_dict: Dict[str, Any], seed: Optional[int] = None):
        """Sets weather temperature profile according to scenario specification."""
        w_cfg = scenario_dict.get("weather", {})
        w_type = w_cfg.get("type", "synthetic")

        if w_type == "open_meteo":
            self.temps_c = self.fetch_open_meteo()
        else:
            mean_t = float(w_cfg.get("mean_temp_c", 28.0))
            amp_t = float(w_cfg.get("amplitude_temp_c", 6.0))
            pk_hr = float(w_cfg.get("peak_hour", 14.0))
            self.temps_c = self.generate_synthetic_profile(
                mean_temp_c=mean_t,
                amplitude_temp_c=amp_t,
                peak_hour=pk_hr,
                seed=seed
            )

    def get_temperature(self, step: int) -> float:
        """Returns actual temperature at given step index."""
        safe_step = int(np.clip(step, 0, self.episode_steps - 1))
        return float(self.temps_c[safe_step])

    def get_weather_forecast(
        self,
        current_step: int,
        horizon_steps: int = 24,
        base_sigma: float = 0.2,
        horizon_alpha: float = 0.15,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        """Generates noisy weather forecast where uncertainty grows with horizon.

        sigma(h) = base_sigma + horizon_alpha * sqrt(h)
        Forecast at h=0 has 0 error.
        """
        if rng is None:
            rng = np.random.default_rng(seed) if seed is not None else self.rng
        forecast = np.zeros(horizon_steps, dtype=np.float32)

        for h in range(horizon_steps):
            target_step = min(current_step + h, self.episode_steps - 1)
            actual_t = self.temps_c[target_step]
            if h == 0:
                forecast[h] = actual_t
            else:
                sigma_h = base_sigma + horizon_alpha * np.sqrt(h)
                error = float(rng.normal(0.0, sigma_h))
                forecast[h] = actual_t + error

        return forecast
