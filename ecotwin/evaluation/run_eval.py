"""Full benchmark evaluation runner across controllers, scenarios, and random seeds."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
import os
import time
import pandas as pd
import numpy as np

# Apply OpenMP duplicate fix
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ecotwin.env.ecotwin_env import EcoTwinEnv
from ecotwin.controllers.fixed import FixedController
from ecotwin.controllers.rule_based import RuleBasedController
from ecotwin.controllers.ppo_train import PPOAgent, train_ppo
from ecotwin.controllers.ppo_pref import PPOPrefAgent, train_ppo_pref
from ecotwin.preference.simulated_manager import SimulatedManager
from ecotwin.preference.pair_generator import PairGenerator
from ecotwin.preference.reward_model import PreferenceRewardModel
from ecotwin.evaluation.metrics import compute_controller_metrics
from ecotwin.evaluation.plots import plot_cost_comparison, plot_power_profile, plot_temperature_traces
from ecotwin.llm.logger import FloorLogger


def run_single_episode(
    controller,
    env: EcoTwinEnv,
    seed: int,
    logger: Optional[FloorLogger] = None,
    scenario_name: str = "summer_day"
) -> Dict[str, Any]:
    """Runs a single 96-step episode with a controller and returns summary metrics."""
    obs, info = env.reset(seed=seed, options={"scenario": scenario_name})
    term = False
    tot_reward = 0.0

    while not term:
        action = controller.act(obs)
        step_idx = env.sim.current_step
        price = env.tariff_mgr.get_price(step_idx)
        out_t = env.weather_mgr.get_temperature(step_idx)

        obs, r, term, _, info = env.step(action)
        tot_reward += r

        if logger is not None and env.last_result is not None:
            logger.log_step(
                result=env.last_result,
                action=action,
                outside_temp=out_t,
                price_per_kwh=price,
                reward=r,
                scenario_name=scenario_name
            )

    last_res = env.last_result
    return {
        "controller": getattr(controller, "name", str(controller.__class__.__name__)),
        "scenario": scenario_name,
        "seed": seed,
        "cost_inr": last_res.cumulative_cost_inr if last_res else 0.0,
        "peak_kw": last_res.peak_power_kw if last_res else 0.0,
        "kwh": last_res.cumulative_energy_kwh if last_res else 0.0,
        "demand_charge_inr": env.tariff_mgr.compute_demand_charge(last_res.peak_power_kw) if last_res else 0.0,
        "completed_jobs": last_res.completed_jobs_count if last_res else 0,
        "quota": last_res.quota if last_res else 30,
        "late_jobs": last_res.late_jobs_count if last_res else 0,
        "comfort_violations_deg": last_res.total_comfort_violation if last_res else 0.0,
        "total_reward": tot_reward
    }


def ensure_trained_models(
    ppo_hand_path: str = "models/ppo_hand_model.zip",
    ppo_pref_path: str = "models/ppo_pref_model.zip",
    reward_model_path: str = "models/reward_model.pt",
    train_timesteps: int = 15000,
    verbose: int = 1
):
    """Ensures PPO models exist; trains them if missing."""
    Path("models").mkdir(parents=True, exist_ok=True)

    # 1. Hand-reward PPO
    if not Path(ppo_hand_path).exists():
        if verbose > 0:
            print(f"PPO hand model not found. Training for {train_timesteps} timesteps...")
        train_ppo(
            total_timesteps=train_timesteps,
            seed=42,
            save_path=ppo_hand_path,
            verbose=verbose
        )

    # 2. Preference Reward Model
    rm = PreferenceRewardModel(obs_dim=58, act_dim=8)
    if not Path(reward_model_path).exists():
        if verbose > 0:
            print("Training preference reward model from simulated manager pairs...")
        env = EcoTwinEnv(randomize_scenarios=True)
        mgr = SimulatedManager(random_seed=42)
        gen = PairGenerator(env, mgr, segment_length=32)
        pairs = gen.generate_pairs(num_pairs=60)
        rm.train_on_pairs(
            train_pairs=pairs[:50],
            val_pairs=pairs[50:],
            epochs=20,
            save_path=reward_model_path,
            verbose=verbose
        )
    else:
        rm.load(reward_model_path)

    # 3. Preference PPO
    if not Path(ppo_pref_path).exists():
        if verbose > 0:
            print(f"PPO preference model not found. Training for {train_timesteps} timesteps...")
        train_ppo_pref(
            reward_model=rm,
            total_timesteps=train_timesteps,
            seed=42,
            save_path=ppo_pref_path,
            verbose=verbose
        )


def run_full_benchmark(
    seeds: List[int] = [42, 101, 2026],
    scenarios: Optional[List[str]] = None,
    output_dir: str = "evaluation",
    train_timesteps: int = 15000,
    verbose: int = 1
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Runs full evaluation across all four controllers, all scenarios, and multiple seeds."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if scenarios is None:
        scenarios = ["summer_day", "winter_day", "price_spike", "heavy_quota", "rush_order"]

    ppo_hand_path = "models/ppo_hand_model.zip"
    ppo_pref_path = "models/ppo_pref_model.zip"
    rm_path = "models/reward_model.pt"

    ensure_trained_models(
        ppo_hand_path=ppo_hand_path,
        ppo_pref_path=ppo_pref_path,
        reward_model_path=rm_path,
        train_timesteps=train_timesteps,
        verbose=verbose
    )

    # Initialize all four controllers
    controllers = [
        FixedController(),
        RuleBasedController(),
        PPOAgent(model_path=ppo_hand_path),
        PPOPrefAgent(model_path=ppo_pref_path)
    ]

    all_records = []
    telemetry_samples: Dict[str, pd.DataFrame] = {}
    summer_rule_df = pd.DataFrame()

    if verbose > 0:
        print(f"\nRunning Full Benchmark across {len(controllers)} controllers, {len(scenarios)} scenarios, {len(seeds)} seeds...")

    env = EcoTwinEnv()
    for scen in scenarios:
        for seed in seeds:
            for ctrl in controllers:
                log = FloorLogger(log_dir=str(out / "logs"))
                rec = run_single_episode(ctrl, env, seed=seed, logger=log, scenario_name=scen)
                all_records.append(rec)

                # Save sample telemetry on summer_day seed 42 for plotting
                if scen == "summer_day" and seed == seeds[0]:
                    df_run = log.to_dataframe()
                    telemetry_samples[ctrl.name] = df_run
                    if ctrl.name == "Rule-Based Heuristic":
                        summer_rule_df = df_run

    # Compute aggregate metrics
    summary_df = compute_controller_metrics(all_records, baseline_controller_name="Fixed Schedule")
    summary_csv = out / "results_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    # Generate benchmark plots
    plot_cost_comparison(summary_df, output_dir=output_dir)
    plot_power_profile(telemetry_samples, scenario_name="summer_day", output_dir=output_dir)
    if not summer_rule_df.empty:
        plot_temperature_traces(summer_rule_df, controller_name="Rule-Based Heuristic", scenario_name="summer_day", output_dir=output_dir)

    # Run Ablation: PPO with vs without weather forecast on summer_day
    if verbose > 0:
        print("\nRunning Ablation Study: PPO With vs Without Weather Forecast...")
    ablation_records = []
    ppo_agent = PPOAgent(model_path=ppo_hand_path)

    for with_fc in [True, False]:
        lbl = "PPO with Weather Forecast" if with_fc else "PPO without Weather Forecast (Ablation)"
        env_ablation = EcoTwinEnv(scenario_name="summer_day", include_weather_forecast=with_fc)
        for seed in seeds:
            obs, _ = env_ablation.reset(seed=seed)
            term = False
            while not term:
                act = ppo_agent.act(obs)
                obs, _, term, _, info = env_ablation.step(act)
            res = env_ablation.last_result
            ablation_records.append({
                "configuration": lbl,
                "seed": seed,
                "cost_inr": res.cumulative_cost_inr if res else 0.0,
                "peak_kw": res.peak_power_kw if res else 0.0,
                "completed_jobs": res.completed_jobs_count if res else 0,
                "comfort_violations_deg": res.total_comfort_violation if res else 0.0
            })

    ablation_df = pd.DataFrame(ablation_records).groupby("configuration").mean().reset_index()
    ablation_csv = out / "ablation_results.csv"
    ablation_df.to_csv(ablation_csv, index=False)

    if verbose > 0:
        print("\n=== Benchmark Summary Table ===")
        display_cols = ["controller", "scenario", "cost_inr", "cost_saved_pct", "peak_kw", "peak_reduction_pct", "completed_jobs"]
        print(summary_df[display_cols].to_string(index=False))

        print("\n=== Weather Forecast Ablation Table ===")
        print(ablation_df.to_string(index=False))

    return summary_df, ablation_df


if __name__ == "__main__":
    run_full_benchmark(train_timesteps=12000, verbose=1)

