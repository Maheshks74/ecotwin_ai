"""Tests for Phase 2: Data Pipeline (Tariff, Weather, Scenarios)."""
import pytest
import numpy as np

from ecotwin.data.tariff import TariffManager
from ecotwin.data.weather import WeatherManager
from ecotwin.data.scenarios import ScenarioManager


def test_tou_tariff_slabs():
    """Verify TOU tariff slabs (off-peak, normal, peak)."""
    tm = TariffManager()
    assert tm.get_price(0) == 4.5 # 00:00 is off-peak
    assert tm.get_price(10) == 4.5 # 02:30 is off-peak
    assert tm.get_price(30) == 7.2 # 07:30 is normal
    assert tm.get_price(76) == 12.5 # 19:00 is peak
    assert tm.get_price(92) == 4.5 # 23:00 is off-peak


def test_iex_csv_loader():
    """Verify IEX CSV loader loads and interpolates to 96 steps."""
    tm = TariffManager()
    prices = tm.load_iex_csv("data/sample_iex_prices.csv")
    assert len(prices) == 96
    assert prices[0] > 0
    assert prices.max() > prices.min()
    assert np.all(prices > 0)


def test_demand_charge_calculation():
    """Verify demand charge on peak kW."""
    tm = TariffManager()
    peak_kw = 100.0
    demand_cost = tm.compute_demand_charge(peak_kw)
    expected = 100.0 * tm.demand_charge_inr_per_kw
    assert np.isclose(demand_cost, expected)


def test_weather_synthetic_bounds():
    """Verify synthetic weather generator produces physical temperatures."""
    wm = WeatherManager(random_seed=42)
    temps = wm.generate_synthetic_profile(mean_temp_c=30.0, amplitude_temp_c=8.0)
    assert len(temps) == 96
    assert 20.0 <= temps.min() <= 40.0
    assert 25.0 <= temps.max() <= 45.0
    # Peak should occur around early afternoon (steps 50-65)
    max_step = np.argmax(temps)
    assert 48 <= max_step <= 68


def test_weather_forecast_uncertainty_growth():
    """Verify forecast error variance grows with horizon."""
    wm = WeatherManager(random_seed=123)
    forecast_errors_short = []
    forecast_errors_long = []

    for seed in range(50):
        fc = wm.get_weather_forecast(current_step=20, horizon_steps=24, seed=seed)
        actual_step_0 = wm.get_temperature(20)
        actual_step_23 = wm.get_temperature(20 + 23)

        # h=0 must have zero error
        assert np.isclose(fc[0], actual_step_0, atol=1e-4)

        forecast_errors_short.append(abs(fc[1] - wm.get_temperature(21)))
        forecast_errors_long.append(abs(fc[23] - actual_step_23))

    # Error at horizon 23 should be significantly higher on average than at horizon 1
    assert np.mean(forecast_errors_long) > np.mean(forecast_errors_short)


def test_scenario_manager_loading():
    """Verify all 5 scenarios load with valid structures."""
    sm = ScenarioManager()
    scenarios = sm.list_scenarios()
    expected = ["summer_day", "winter_day", "price_spike", "heavy_quota", "rush_order"]
    for exp in expected:
        assert exp in scenarios

    rush = sm.get_scenario("rush_order")
    assert "rush_batch" in rush
    heavy = sm.get_scenario("heavy_quota")
    assert heavy["quota"] == 45

