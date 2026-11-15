"""Plotting utilities for EcoTwin benchmark evaluation (Matplotlib and Plotly)."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px


def plot_cost_comparison(
    summary_df: pd.DataFrame,
    output_dir: str = "evaluation"
) -> Tuple[str, str]:
    """Generates cost comparison bar chart across controllers and scenarios."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    png_path = out / "cost_comparison.png"
    html_path = out / "cost_comparison.html"

    # Matplotlib static plot
    plt.figure(figsize=(10, 6))
    scenarios = summary_df["scenario"].unique()
    controllers = summary_df["controller"].unique()
    x = np.arange(len(scenarios))
    width = 0.8 / max(1, len(controllers))

    colors = ["#4A5568", "#3182CE", "#38A169", "#D69E2E"]

    for i, ctrl in enumerate(controllers):
        sub = summary_df[summary_df["controller"] == ctrl]
        vals = [sub[sub["scenario"] == s]["cost_inr"].values[0] if len(sub[sub["scenario"] == s]) > 0 else 0 for s in scenarios]
        plt.bar(x + i * width - 0.4 + width / 2, vals, width, label=ctrl, color=colors[i % len(colors)])

    plt.xlabel("Manufacturing Scenario")
    plt.ylabel("Electricity Cost (INR)")
    plt.title("EcoTwin Controller Cost Benchmark Across Scenarios")
    plt.xticks(x, [s.replace("_", " ").title() for s in scenarios], rotation=15)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()

    # Plotly interactive chart
    fig = px.bar(
        summary_df,
        x="scenario",
        y="cost_inr",
        color="controller",
        barmode="group",
        title="EcoTwin Electricity Cost Benchmark by Scenario",
        labels={"cost_inr": "Electricity Cost (INR)", "scenario": "Scenario"}
    )
    fig.write_html(html_path)

    return str(png_path), str(html_path)


def plot_power_profile(
    telemetry_by_controller: Dict[str, pd.DataFrame],
    scenario_name: str = "summer_day",
    output_dir: str = "evaluation"
) -> Tuple[str, str]:
    """Plots 24-hour total floor electrical power demand traces for each controller."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    png_path = out / f"power_profile_{scenario_name}.png"
    html_path = out / f"power_profile_{scenario_name}.html"

    plt.figure(figsize=(12, 6))
    fig_plotly = go.Figure()

    colors = {"Fixed Schedule": "#4A5568", "Rule-Based Heuristic": "#3182CE", "PPO Hand-Reward": "#38A169", "PPO Preference-Reward": "#D69E2E"}

    for ctrl_name, df in telemetry_by_controller.items():
        if df.empty or "hour" not in df.columns or "total_power_kw" not in df.columns:
            continue
        c = colors.get(ctrl_name, "#718096")
        plt.plot(df["hour"], df["total_power_kw"], label=ctrl_name, color=c, linewidth=2.0)
        fig_plotly.add_trace(go.Scatter(x=df["hour"], y=df["total_power_kw"], mode="lines", name=ctrl_name, line=dict(color=c, width=2)))

    # Shade peak slab 18:00 to 22:00
    plt.axvspan(18.0, 22.0, color="#FED7D7", alpha=0.3, label="Peak Tariff Window (18-22h)")
    plt.xlabel("Hour of Day (0..24)")
    plt.ylabel("Total Floor Power (kW)")
    plt.title(f"24-Hour Factory Power Profile ({scenario_name.replace('_', ' ').title()})")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()

    fig_plotly.add_vrect(x0=18.0, x1=22.0, fillcolor="red", opacity=0.15, layer="below", line_width=0, annotation_text="Peak Tariff")
    fig_plotly.update_layout(title=f"24-Hour Factory Power Profile ({scenario_name})", xaxis_title="Hour of Day", yaxis_title="Power (kW)")
    fig_plotly.write_html(html_path)

    return str(png_path), str(html_path)


def plot_temperature_traces(
    df: pd.DataFrame,
    controller_name: str = "Rule-Based Heuristic",
    scenario_name: str = "summer_day",
    output_dir: str = "evaluation"
) -> Tuple[str, str]:
    """Plots zone temperatures over 24h against outdoor temperature and comfort band."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    png_path = out / f"temperature_traces_{scenario_name}.png"
    html_path = out / f"temperature_traces_{scenario_name}.html"

    plt.figure(figsize=(12, 6))
    if not df.empty and "hour" in df.columns:
        plt.plot(df["hour"], df["outside_temp_c"], label="Ambient Outdoor Temp", color="#E53E3E", linestyle="--", linewidth=1.8)
        if "zone_1_temp_c" in df.columns:
            plt.plot(df["hour"], df["zone_1_temp_c"], label="Zone 1 (Machining)", color="#3182CE", linewidth=2.0)
        if "zone_2_temp_c" in df.columns:
            plt.plot(df["hour"], df["zone_2_temp_c"], label="Zone 2 (Forming & Assembly)", color="#38A169", linewidth=2.0)

    # Shade comfort band (22 - 26 C)
    plt.axhspan(22.0, 26.0, color="#C6F6D5", alpha=0.35, label="Comfort Band (22-26°C)")
    plt.xlabel("Hour of Day (0..24)")
    plt.ylabel("Temperature (°C)")
    plt.title(f"Thermal Zone Dynamics - {controller_name} ({scenario_name.replace('_', ' ').title()})")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()

    fig_plotly = go.Figure()
    if not df.empty and "hour" in df.columns:
        fig_plotly.add_trace(go.Scatter(x=df["hour"], y=df["outside_temp_c"], mode="lines", name="Ambient Temp", line=dict(color="red", dash="dash")))
        if "zone_1_temp_c" in df.columns:
            fig_plotly.add_trace(go.Scatter(x=df["hour"], y=df["zone_1_temp_c"], mode="lines", name="Zone 1", line=dict(color="blue", width=2)))
        if "zone_2_temp_c" in df.columns:
            fig_plotly.add_trace(go.Scatter(x=df["hour"], y=df["zone_2_temp_c"], mode="lines", name="Zone 2", line=dict(color="green", width=2)))

    fig_plotly.add_hrect(y0=22.0, y1=26.0, fillcolor="green", opacity=0.15, layer="below", line_width=0, annotation_text="Comfort Band")
    fig_plotly.update_layout(title=f"Thermal Zone Dynamics - {controller_name}", xaxis_title="Hour", yaxis_title="Temp (°C)")
    fig_plotly.write_html(html_path)

    return str(png_path), str(html_path)

