"""Tests for Phase 6: Preference Learning (Simulated Manager, Pairs, Reward Model)."""
from pathlib import Path
import pytest
import numpy as np

from ecotwin.env.ecotwin_env import EcoTwinEnv
from ecotwin.preference.simulated_manager import SimulatedManager, TrajectorySegment
from ecotwin.preference.pair_generator import PairGenerator
from ecotwin.preference.reward_model import PreferenceRewardModel
from ecotwin.controllers.ppo_pref import PreferenceRewardWrapper


def test_simulated_manager_preferences():
    """Verify manager prefers segments with low night power, high quota, and few comfort violations."""
    mgr = SimulatedManager(random_seed=42)

    # Good segment: fulfilled quota, no comfort violations, low night power
    seg_good = TrajectorySegment(
        segment_id="good",
        states=np.zeros((10, 58), dtype=np.float32),
        actions=np.zeros((10, 8), dtype=np.int64),
        cumulative_cost=5000.0,
        total_comfort_violation=0.0,
        peak_kw=100.0,
        night_machine_kwh=10.0,
        completed_jobs=30,
        quota=30
    )

    # Bad segment: heavy night shift, massive comfort violations, missed quota
    seg_bad = TrajectorySegment(
        segment_id="bad",
        states=np.zeros((10, 58), dtype=np.float32),
        actions=np.zeros((10, 8), dtype=np.int64),
        cumulative_cost=6000.0,
        total_comfort_violation=12.0,
        peak_kw=140.0,
        night_machine_kwh=120.0,
        completed_jobs=15,
        quota=30
    )

    u_good = mgr.compute_latent_utility(seg_good)
    u_bad = mgr.compute_latent_utility(seg_bad)
    assert u_good > u_bad, f"Good utility {u_good} must exceed bad utility {u_bad}"

    # Comparison should choose seg_good
    winner, prob = mgr.compare_trajectories(seg_good, seg_bad, add_noise=False)
    assert winner == 1
    assert prob > 0.90


def test_pair_generator_and_reward_model_training(tmp_path):
    """Verify end-to-end pair generation, reward training, accuracy, and model persistence."""
    env = EcoTwinEnv(scenario_name="summer_day")
    mgr = SimulatedManager(random_seed=123)
    gen = PairGenerator(env, mgr, segment_length=24)

    # Generate small dataset of 40 pairs
    pairs = gen.generate_pairs(num_pairs=40)
    assert len(pairs) == 40
    seg1, seg2, y = pairs[0]
    assert seg1.states.shape == (24, 58)
    assert y in (0, 1)

    train_pairs = pairs[:30]
    val_pairs = pairs[30:]

    reward_model = PreferenceRewardModel(obs_dim=58, act_dim=8, hidden_dim=32)
    save_path = str(tmp_path / "test_reward_model.pt")

    val_acc = reward_model.train_on_pairs(
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        epochs=15,
        lr=2e-3,
        batch_size=8,
        save_path=save_path,
        verbose=0
    )

    assert Path(save_path).exists()
    assert val_acc >= 0.60 # Should learn to classify better than random on small set

    # Test inference
    single_r = reward_model.predict_step_reward(seg1.states[0], seg1.actions[0])
    assert np.isfinite(single_r)


def test_preference_reward_wrapper():
    """Verify PreferenceRewardWrapper replaces reward and exposes learned_reward info."""
    raw_env = EcoTwinEnv(scenario_name="summer_day")
    reward_model = PreferenceRewardModel(obs_dim=58, act_dim=8)
    wrapped = PreferenceRewardWrapper(raw_env, reward_model)

    obs, info = wrapped.reset(seed=42)
    act = wrapped.action_space.sample()
    obs, r, term, trunc, info = wrapped.step(act)

    assert np.isfinite(r)
    assert "learned_reward" in info
    assert "pref_reward" in info

