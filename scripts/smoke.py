"""End-to-end smoke test script for EcoTwin."""
import sys
import os

# Ensure project root is in path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from ecotwin.twin.factory import FactoryFloorSim


def smoke_phase1():
    print("=== Smoke Test Phase 1: Digital Twin ===")
    sim = FactoryFloorSim()
    sim.reset(seed=42)
    print(f"Factory initialized: {len(sim.machines)} machines, {len(sim.zones)} zones.")

    # Run full 96-step episode
    for step in range(sim.episode_steps):
        m_actions = {mid: 1 for mid in sim.machines}
        h_actions = {zid: 2 for zid in sim.zones}
        price = 5.0 + 3.0 * np.sin(step * 2 * np.pi / 96)
        temp = 28.0 + 8.0 * np.sin((step - 24) * 2 * np.pi / 96)

        res = sim.step(m_actions, h_actions, price_per_kwh=price, outside_temp_c=temp)

    print(f"Episode completed: Steps={sim.episode_steps}")
    print(f"  Total kWh: {sim.cumulative_energy_kwh:.2f}")
    print(f"  Total Cost: INR {sim.cumulative_cost_inr:.2f}")
    print(f"  Peak kW: {sim.peak_power_kw:.2f}")
    print(f"  Jobs Completed: {sim.job_queue.num_completed}/{sim.quota}")
    assert sim.cumulative_energy_kwh > 0
    assert sim.cumulative_cost_inr > 0
    assert sim.peak_power_kw > 0
    print("[PASS] Phase 1 smoke test passed successfully.\n")


def smoke_phase2():
    print("=== Smoke Test Phase 2: Data Pipeline ===")
    from ecotwin.data.tariff import TariffManager
    from ecotwin.data.weather import WeatherManager
    from ecotwin.data.scenarios import ScenarioManager

    tm = TariffManager()
    wm = WeatherManager()
    sm = ScenarioManager()

    print(f"Loaded {len(sm.list_scenarios())} scenarios: {sm.list_scenarios()}")
    for s_name in sm.list_scenarios():
        scen = sm.get_scenario(s_name)
        tm.set_scenario_tariff(scen)
        wm.set_scenario_weather(scen)
        p0 = tm.get_price(0)
        t0 = wm.get_temperature(0)
        p_fc = tm.get_price_forecast(0, 24)
        w_fc = wm.get_weather_forecast(0, 24)
        print(f"  Scenario '{s_name}': T0={t0:.1f}C, P0={p0:.1f} INR, PriceForecastLen={len(p_fc)}, WeatherForecastLen={len(w_fc)}")

    print("[PASS] Phase 2 smoke test passed successfully.\n")


def smoke_phase3():
    print("=== Smoke Test Phase 3: Gymnasium Environment ===")
    from ecotwin.env.ecotwin_env import EcoTwinEnv
    env = EcoTwinEnv(scenario_name="summer_day")
    obs, info = env.reset(seed=42)
    print(f"Env reset: Obs shape={obs.shape}, Action space={env.action_space}")

    ep_reward = 0.0
    for step in range(96):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        if terminated or truncated:
            break

    print(f"Episode finished at step {info['step']}:")
    print(f"  Total reward: {ep_reward:.2f}")
    print(f"  Cost: INR {info['cumulative_cost_inr']:.2f}")
    print(f"  Peak kW: {info['peak_kw']:.2f}")
    print(f"  Completed jobs: {info['completed_jobs']}/{info['quota']}")
    print("[PASS] Phase 3 smoke test passed successfully.\n")


def smoke_phase4():
    print("=== Smoke Test Phase 4: Baseline Controllers ===")
    from ecotwin.env.ecotwin_env import EcoTwinEnv
    from ecotwin.controllers.fixed import FixedController
    from ecotwin.controllers.rule_based import RuleBasedController

    env = EcoTwinEnv(scenario_name="summer_day")
    fixed = FixedController()
    rule = RuleBasedController()

    for ctrl in [fixed, rule]:
        obs, _ = env.reset(seed=123)
        term = False
        tot_r = 0.0
        while not term:
            act = ctrl.act(obs)
            obs, r, term, _, info = env.step(act)
            tot_r += r
        print(f"  {ctrl.name}: Cost=INR {info['cumulative_cost_inr']:.2f}, Peak={info['peak_kw']:.2f} kW, Completed={info['completed_jobs']}/{info['quota']}")

    print("[PASS] Phase 4 smoke test passed successfully.\n")


def smoke_phase5():
    print("=== Smoke Test Phase 5: PPO Training ===")
    from ecotwin.controllers.ppo_train import train_ppo, PPOAgent
    from ecotwin.env.ecotwin_env import EcoTwinEnv

    print("Training micro PPO model (384 steps)...")
    save_path = "models/smoke_ppo.zip"
    train_ppo(
        total_timesteps=384,
        seed=42,
        save_path=save_path,
        n_steps=384,
        batch_size=64,
        n_epochs=1,
        verbose=0
    )

    agent = PPOAgent(model_path=save_path)
    env = EcoTwinEnv(scenario_name="summer_day")
    obs, _ = env.reset(seed=99)
    act = agent.act(obs)
    print(f"PPO agent loaded and generated action: {act}")
    obs, r, term, _, info = env.step(act)
    print(f"Step executed with reward: {r:.3f}")
    print("[PASS] Phase 5 smoke test passed successfully.\n")


def smoke_phase6():
    print("=== Smoke Test Phase 6: Preference Learning ===")
    from ecotwin.env.ecotwin_env import EcoTwinEnv
    from ecotwin.preference.simulated_manager import SimulatedManager
    from ecotwin.preference.pair_generator import PairGenerator
    from ecotwin.preference.reward_model import PreferenceRewardModel

    env = EcoTwinEnv(scenario_name="summer_day")
    mgr = SimulatedManager(random_seed=42)
    gen = PairGenerator(env, mgr, segment_length=16)

    print("Generating 10 preference pairs...")
    pairs = gen.generate_pairs(num_pairs=10)
    print(f"Generated {len(pairs)} pairs with labels: {[y for _, _, y in pairs]}")

    rm = PreferenceRewardModel(obs_dim=58, act_dim=8, hidden_dim=32)
    acc = rm.train_on_pairs(train_pairs=pairs, epochs=5, lr=1e-3, save_path="models/smoke_reward_model.pt", verbose=0)
    print(f"Reward model trained on pairs. Accuracy: {acc:.1%}")
    print("[PASS] Phase 6 smoke test passed successfully.\n")


def smoke_phase7():
    print("=== Smoke Test Phase 7: LLM Advisor & Instruction Parser ===")
    from ecotwin.llm.client import GeminiClient
    from ecotwin.llm.advisor import LLMAdvisor
    from ecotwin.llm.instruction_parser import InstructionParser
    from ecotwin.llm.logger import FloorLogger
    import pandas as pd

    client = GeminiClient()
    print(f"Gemini Client Status: {client.badge}")

    parser = InstructionParser(client=client)
    weights, reason = parser.parse_instruction("Customer batch is urgent, ensure high priority!")
    print(f"Parsed Weights: {weights}")
    print(f"Reasoning: {reason}")

    advisor = LLMAdvisor(client=client)
    df_sample = pd.DataFrame([
        {"step": 8, "hour": 2.0, "price_inr_per_kwh": 4.5, "outside_temp_c": 22.0,
         "total_power_kw": 40.0, "machines_power_kw": 30.0, "hvac_power_kw": 10.0,
         "zone_1_temp_c": 24.0, "zone_2_temp_c": 24.0, "min_slack": 20.0, "completed_jobs": 5}
    ])

    exp = advisor.explain("Why was Job 14 started at 2 AM?", df_sample)
    print(f"Advisor Explanation: {exp}")

    missing_exp = advisor.explain("Why was Job 999 rejected?", df_sample)
    print(f"Missing Fact Test: '{missing_exp}'")
    assert "not enough information" in missing_exp.lower()

    whatif = advisor.run_what_if_analysis("What if temperature setpoint is relaxed by 1 degree?")
    print(f"What-If Result: Cost Delta={whatif['cost_delta_inr']:+.2f} INR")
    print("[PASS] Phase 7 smoke test passed successfully.\n")


def smoke_phase8():
    print("=== Smoke Test Phase 8: Evaluation & Benchmark ===")
    from ecotwin.evaluation.run_eval import run_single_episode
    from ecotwin.controllers.rule_based import RuleBasedController
    from ecotwin.env.ecotwin_env import EcoTwinEnv
    from ecotwin.evaluation.plots import plot_cost_comparison, plot_power_profile, plot_temperature_traces
    import pandas as pd

    env = EcoTwinEnv(scenario_name="winter_day")
    ctrl = RuleBasedController()
    res = run_single_episode(ctrl, env, seed=42, scenario_name="winter_day")
    print(f"Eval episode finished: Cost={res['cost_inr']:.2f} INR, Peak={res['peak_kw']:.2f} kW, Completed={res['completed_jobs']}/{res['quota']}")

    # Test plot generation
    df_metrics = pd.DataFrame([
        {"controller": "Fixed Schedule", "scenario": "winter_day", "cost_inr": 5000.0},
        {"controller": "Rule-Based Heuristic", "scenario": "winter_day", "cost_inr": 4600.0},
    ])
    p1, _ = plot_cost_comparison(df_metrics, output_dir="evaluation")
    print(f"Sample evaluation plot created at: {p1}")
    print("[PASS] Phase 8 smoke test passed successfully.\n")


def smoke_phase9():
    print("=== Smoke Test Phase 9: Dashboard AppTest ===")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("dashboard/app.py", default_timeout=30)
    at.run()
    assert not at.exception
    assert len(at.tabs) == 4
    print("Dashboard initialized and rendered all 4 tabs without exceptions.")
    print("[PASS] Phase 9 smoke test passed successfully.\n")


if __name__ == "__main__":
    smoke_phase1()
    smoke_phase2()
    smoke_phase3()
    smoke_phase4()
    smoke_phase5()
    smoke_phase6()
    smoke_phase7()
    smoke_phase8()
    smoke_phase9()
    print("=====================================================")
    print("ALL PHASES 1-9 SMOKE TESTS PASSED CLEANLY & OFFLINE!")
    print("=====================================================")








