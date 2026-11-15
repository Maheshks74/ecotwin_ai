"""Tests for Phase 4: Baseline Controllers (Fixed, Rule-Based)."""
import pytest
import numpy as np

from ecotwin.env.ecotwin_env import EcoTwinEnv
from ecotwin.controllers.fixed import FixedController
from ecotwin.controllers.rule_based import RuleBasedController


def test_controller_interfaces():
    """Verify both controllers expose identical act(obs) -> action interface adhering to action space."""
    env = EcoTwinEnv(scenario_name="summer_day")
    obs, _ = env.reset(seed=42)

    fixed = FixedController()
    rule = RuleBasedController()

    act_fixed = fixed.act(obs)
    act_rule = rule.act(obs)

    assert act_fixed.shape == (8,)
    assert act_rule.shape == (8,)
    assert env.action_space.contains(act_fixed)
    assert env.action_space.contains(act_rule)


def test_rule_based_precooling_and_deferral():
    """Verify rule-based logic switches HVAC setpoints and defers heavy machines during peak."""
    rule = RuleBasedController()

    # Synthetic obs at step 65 (16:15 - pre-cooling window)
    angle_precool = (65 / 96.0) * 2 * np.pi
    obs_precool = np.zeros(58, dtype=np.float32)
    obs_precool[0] = np.sin(angle_precool)
    obs_precool[1] = np.cos(angle_precool)
    obs_precool[2] = 0.0 # normal price
    obs_precool[55] = 1.0 # ample slack

    act_precool = rule.act(obs_precool)
    assert act_precool[6] == 0 and act_precool[7] == 0 # Pre-cool to 22 C (index 0)

    # Synthetic obs at step 75 (18:45 - peak electricity slab)
    angle_peak = (75 / 96.0) * 2 * np.pi
    obs_peak = np.zeros(58, dtype=np.float32)
    obs_peak[0] = np.sin(angle_peak)
    obs_peak[1] = np.cos(angle_peak)
    obs_peak[2] = 1.0 # high price
    obs_peak[55] = 1.0 # ample slack

    act_peak = rule.act(obs_peak)
    assert act_peak[6] == 4 and act_peak[7] == 4 # Float HVAC to 26 C (index 4)
    assert act_peak[3] == 0 and act_peak[4] == 0 # Defer heavy machines M4 & M5


def test_controllers_episode_evaluation():
    """Run full episode evaluation with Fixed vs RuleBased on Summer Day."""
    env = EcoTwinEnv(scenario_name="summer_day")

    fixed = FixedController()
    rule = RuleBasedController()

    # Run Fixed
    obs, _ = env.reset(seed=100)
    term = False
    fixed_reward = 0.0
    while not term:
        act = fixed.act(obs)
        obs, r, term, _, info_fixed = env.step(act)
        fixed_reward += r

    # Run Rule-Based
    obs, _ = env.reset(seed=100)
    term = False
    rule_reward = 0.0
    while not term:
        act = rule.act(obs)
        obs, r, term, _, info_rule = env.step(act)
        rule_reward += r

    print(f"\nFixed: Cost=INR {info_fixed['cumulative_cost_inr']:.2f}, Peak={info_fixed['peak_kw']:.2f} kW, Reward={fixed_reward:.2f}")
    print(f"Rule:  Cost=INR {info_rule['cumulative_cost_inr']:.2f}, Peak={info_rule['peak_kw']:.2f} kW, Reward={rule_reward:.2f}")

    assert info_fixed["completed_jobs"] >= info_fixed["quota"]
    assert info_rule["completed_jobs"] >= info_rule["quota"]
    # Rule-based should achieve lower electricity cost or peak kW
    assert info_rule["cumulative_cost_inr"] <= info_fixed["cumulative_cost_inr"] + 10.0

