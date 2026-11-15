"""Tests for Phase 5: PPO Training pipeline."""
from pathlib import Path
import pytest
import numpy as np

from ecotwin.env.ecotwin_env import EcoTwinEnv
from ecotwin.controllers.ppo_train import train_ppo, PPOAgent


def test_ppo_short_training_and_saving(tmp_path):
    """Verify PPO can train for a small budget (768 steps = 2 batches) and save model."""
    save_path = str(tmp_path / "test_ppo_model.zip")
    tb_log = str(tmp_path / "tb_logs")

    model = train_ppo(
        total_timesteps=768, # 2 rollouts of 384
        seed=42,
        save_path=save_path,
        tensorboard_log=tb_log,
        checkpoint_freq=5000,
        randomize_scenarios=False,
        n_steps=384,
        batch_size=64,
        n_epochs=2,
        verbose=0
    )

    assert Path(save_path).exists()
    assert Path(save_path).stat().st_size > 1000 # Valid zip file


def test_ppo_agent_act_interface(tmp_path):
    """Verify PPOAgent loads trained checkpoint and produces valid actions."""
    save_path = str(tmp_path / "test_ppo_model.zip")
    train_ppo(
        total_timesteps=384,
        seed=123,
        save_path=save_path,
        n_steps=384,
        batch_size=64,
        n_epochs=1,
        verbose=0
    )

    agent = PPOAgent(model_path=save_path)
    env = EcoTwinEnv(scenario_name="summer_day")
    obs, _ = env.reset(seed=55)

    action = agent.act(obs)
    assert action.shape == (8,)
    assert env.action_space.contains(action)

    # Step in environment
    obs, r, term, trunc, info = env.step(action)
    assert np.isfinite(r)

