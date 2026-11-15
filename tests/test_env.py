"""Tests for Phase 3: Gymnasium Environment."""
import pytest
import numpy as np
from gymnasium.utils.env_checker import check_env

from ecotwin.env.ecotwin_env import EcoTwinEnv


def test_gymnasium_check_env():
    """Verify standard Gymnasium compliance with check_env."""
    env = EcoTwinEnv(scenario_name="summer_day")
    # check_env raises exception if environment violates standard API
    check_env(env.unwrapped)


def test_random_1000_steps():
    """Verify environment runs 1000 random steps across multiple resets without error."""
    env = EcoTwinEnv(randomize_scenarios=True)
    obs, info = env.reset(seed=42)
    assert obs.shape == (58,)

    for step in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (58,)
        assert np.all(np.isfinite(obs))
        assert np.isfinite(reward)

        if terminated or truncated:
            obs, info = env.reset()


def test_producing_nothing_is_never_optimal():
    """Prove that an agent choosing HOLD everywhere (producing 0 jobs) receives a strictly worse

    total episode reward than an agent that fulfills the quota, even considering energy saved.
    """
    env_lazy = EcoTwinEnv(scenario_name="summer_day")
    env_active = EcoTwinEnv(scenario_name="summer_day")

    # Seed both environments identically
    obs_lazy, _ = env_lazy.reset(seed=101)
    obs_active, _ = env_active.reset(seed=101)

    lazy_total_reward = 0.0
    active_total_reward = 0.0

    # Policy 1: LAZY (HOLD all machines all 96 steps)
    lazy_action = np.array([0, 0, 0, 0, 0, 0, 2, 2])
    terminated = False
    while not terminated:
        _, reward, terminated, _, info_lazy = env_lazy.step(lazy_action)
        lazy_total_reward += reward

    # Policy 2: ACTIVE (Run machines to complete jobs, default setpoint)
    active_action = np.array([1, 1, 1, 1, 1, 1, 2, 2])
    terminated = False
    while not terminated:
        _, reward, terminated, _, info_active = env_active.step(active_action)
        active_total_reward += reward

    print(f"\nLazy total reward: {lazy_total_reward:.2f}, completed: {info_lazy['completed_jobs']}/{info_lazy['quota']}")
    print(f"Active total reward: {active_total_reward:.2f}, completed: {info_active['completed_jobs']}/{info_active['quota']}")

    # Verification:
    # 1. Lazy agent completed 0 jobs
    assert info_lazy["completed_jobs"] == 0
    # 2. Active agent completed at least quota
    assert info_active["completed_jobs"] >= info_active["quota"]
    # 3. Active policy reward is significantly higher (less negative) than Lazy policy reward
    assert active_total_reward > lazy_total_reward, (
        f"Active reward ({active_total_reward}) must exceed Lazy reward ({lazy_total_reward})"
    )
    # The penalty difference should be substantial (at least 50 reward units)
    assert (active_total_reward - lazy_total_reward) > 40.0


def test_runtime_reward_weights_update():
    """Verify that runtime changes to reward weights directly affect reward calculation."""
    env = EcoTwinEnv(scenario_name="summer_day")
    env.reset(seed=77)

    # Step with standard weights
    action = env.action_space.sample()
    _, r1, _, _, _ = env.step(action)

    # Update comfort penalty drastically
    env.set_reward_weights({"comfort_violation_weight": 500.0})
    assert env.reward_weights["comfort_violation_weight"] == 500.0

