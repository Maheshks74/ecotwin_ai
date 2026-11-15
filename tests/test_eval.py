"""Tests for Phase 8: Evaluation Pipeline, Metrics, and Visualizations."""
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from ecotwin.env.ecotwin_env import EcoTwinEnv
from ecotwin.controllers.fixed import FixedController
from ecotwin.controllers.rule_based import RuleBasedController
from ecotwin.evaluation.metrics import compute_controller_metrics
from ecotwin.evaluation.plots import plot_cost_comparison, plot_power_profile, plot_temperature_traces
from ecotwin.evaluation.run_eval import run_single_episode


def test_metrics_calculation():
    """Verify compute_controller_metrics computes delta and percentage savings relative to baseline."""
    dummy_records = [
        {"controller": "Fixed Schedule", "scenario": "summer_day", "cost_inr": 6000.0, "peak_kw": 120.0,
         "kwh": 900.0, "demand_charge_inr": 1400.0, "completed_jobs": 30, "quota": 30, "late_jobs": 0, "comfort_violations_deg": 2.0},
        {"controller": "Rule-Based Heuristic", "scenario": "summer_day", "cost_inr": 5400.0, "peak_kw": 108.0,
         "kwh": 880.0, "demand_charge_inr": 1200.0, "completed_jobs": 30, "quota": 30, "late_jobs": 0, "comfort_violations_deg": 1.0},
    ]

    metrics_df = compute_controller_metrics(dummy_records, baseline_controller_name="Fixed Schedule")
    assert len(metrics_df) == 2
    rule_row = metrics_df[metrics_df["controller"] == "Rule-Based Heuristic"].iloc[0]

    assert np.isclose(rule_row["cost_saved_pct"], 10.0) # (6000 - 5400) / 6000 * 100 = 10%
    assert np.isclose(rule_row["peak_reduction_kw"], 12.0)
    assert np.isclose(rule_row["peak_reduction_pct"], 10.0)
    assert np.isclose(rule_row["quota_completion_pct"], 100.0)


def test_plot_generation(tmp_path):
    """Verify plotting functions generate existing PNG and HTML files."""
    df_metrics = pd.DataFrame([
        {"controller": "Fixed Schedule", "scenario": "summer_day", "cost_inr": 6000.0},
        {"controller": "Rule-Based Heuristic", "scenario": "summer_day", "cost_inr": 5400.0},
    ])
    png1, html1 = plot_cost_comparison(df_metrics, output_dir=str(tmp_path))
    assert Path(png1).exists()
    assert Path(html1).exists()

    df_telemetry = pd.DataFrame({
        "hour": np.linspace(0, 24, 96),
        "total_power_kw": 50.0 + 30.0 * np.sin(np.linspace(0, 3.14, 96)),
        "outside_temp_c": 28.0 + 5.0 * np.sin(np.linspace(0, 3.14, 96)),
        "zone_1_temp_c": np.full(96, 24.0),
        "zone_2_temp_c": np.full(96, 24.5),
    })
    png2, html2 = plot_power_profile({"Rule": df_telemetry}, output_dir=str(tmp_path))
    assert Path(png2).exists()
    assert Path(html2).exists()

    png3, html3 = plot_temperature_traces(df_telemetry, output_dir=str(tmp_path))
    assert Path(png3).exists()
    assert Path(html3).exists()


def test_run_single_episode():
    """Verify run_single_episode executes completely and returns valid metric dictionary."""
    env = EcoTwinEnv(scenario_name="winter_day")
    ctrl = RuleBasedController()
    res = run_single_episode(ctrl, env, seed=42, scenario_name="winter_day")

    assert res["scenario"] == "winter_day"
    assert res["completed_jobs"] >= res["quota"]
    assert res["cost_inr"] > 0
    assert res["peak_kw"] > 0

