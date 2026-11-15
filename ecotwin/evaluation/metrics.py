"""Evaluation metrics computation for EcoTwin benchmark."""
from __future__ import annotations
from typing import Any, Dict, List
import pandas as pd
import numpy as np


def compute_controller_metrics(
    eval_records: List[Dict[str, Any]],
    baseline_controller_name: str = "Fixed Schedule"
) -> pd.DataFrame:
    """Aggregates episode records across controllers and scenarios, calculating cost savings

    and peak kW reductions relative to the baseline controller.
    """
    df = pd.DataFrame(eval_records)
    if df.empty:
        return pd.DataFrame()

    # Calculate average across seeds for each (controller, scenario)
    grouped = df.groupby(["controller", "scenario"]).agg({
        "cost_inr": "mean",
        "peak_kw": "mean",
        "kwh": "mean",
        "demand_charge_inr": "mean",
        "completed_jobs": "mean",
        "quota": "mean",
        "late_jobs": "mean",
        "comfort_violations_deg": "mean"
    }).reset_index()

    # Merge with baseline controller for delta / percentage savings
    base_df = grouped[grouped["controller"] == baseline_controller_name][
        ["scenario", "cost_inr", "peak_kw"]
    ].rename(columns={"cost_inr": "base_cost", "peak_kw": "base_peak"})

    merged = pd.merge(grouped, base_df, on="scenario", how="left")
    merged["cost_saved_pct"] = (
        (merged["base_cost"] - merged["cost_inr"]) / np.maximum(1.0, merged["base_cost"])
    ) * 100.0

    merged["peak_reduction_kw"] = merged["base_peak"] - merged["peak_kw"]
    merged["peak_reduction_pct"] = (
        merged["peak_reduction_kw"] / np.maximum(1.0, merged["base_peak"])
    ) * 100.0

    merged["quota_completion_pct"] = (
        merged["completed_jobs"] / np.maximum(1.0, merged["quota"])
    ) * 100.0

    return merged

