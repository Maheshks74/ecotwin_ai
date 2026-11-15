"""PPO training using learned Bradley-Terry preference reward model."""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import os
import gymnasium as gym
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from ecotwin.env.ecotwin_env import EcoTwinEnv
from ecotwin.preference.reward_model import PreferenceRewardModel


class PreferenceRewardWrapper(gym.Wrapper):
    """Wraps EcoTwinEnv to replace the hand-crafted reward with the learned preference reward,

    while preserving a strict quota penalty at episode end so production is not abandoned.
    """

    def __init__(
        self,
        env: EcoTwinEnv,
        reward_model: PreferenceRewardModel,
        reward_scale: float = 0.05,
        quota_penalty_weight: float = 3.0
    ):
        super().__init__(env)
        self.reward_model = reward_model
        self.reward_scale = reward_scale
        self.quota_penalty_weight = quota_penalty_weight

    def step(self, action):
        obs, orig_reward, terminated, truncated, info = self.env.step(action)

        # Predict step reward from learned model
        r_learned = self.reward_model.predict_step_reward(obs, action)

        terminal_quota_penalty = 0.0
        if terminated:
            missed = max(0, info["quota"] - info["completed_jobs"])
            terminal_quota_penalty = missed * self.quota_penalty_weight

        pref_reward = float(
            r_learned * self.reward_scale - terminal_quota_penalty
        )
        info["learned_reward"] = r_learned
        info["pref_reward"] = pref_reward

        return obs, pref_reward, terminated, truncated, info


class PPOPrefAgent:
    """Wrapper around trained PPO model with preference reward."""

    def __init__(self, model_path: Optional[str] = None):
        self.model: Optional[PPO] = None
        self.name = "PPO Preference-Reward"
        if model_path is not None and Path(model_path).exists():
            self.load(model_path)

    def load(self, model_path: str):
        self.model = PPO.load(model_path)

    def act(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        if self.model is None:
            return np.array([1, 1, 1, 1, 1, 1, 2, 2], dtype=np.int64)
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return np.asarray(action, dtype=np.int64)

    def reset(self):
        pass


def train_ppo_pref(
    reward_model: PreferenceRewardModel,
    total_timesteps: int = 15000,
    seed: int = 42,
    save_path: str = "models/ppo_pref_model.zip",
    tensorboard_log: str = "runs/ppo_pref",
    checkpoint_freq: int = 5000,
    learning_rate: float = 3e-4,
    n_steps: int = 384,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    ent_coef: float = 0.01,
    verbose: int = 1
) -> PPO:
    """Train PPO agent using the learned human-preference reward function."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(tensorboard_log).mkdir(parents=True, exist_ok=True)
    checkpoints_dir = Path("checkpoints/ppo_pref")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def make_env():
        raw_env = EcoTwinEnv(randomize_scenarios=True)
        return PreferenceRewardWrapper(raw_env, reward_model)

    vec_env = DummyVecEnv([make_env])

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1000, checkpoint_freq),
        save_path=str(checkpoints_dir),
        name_prefix=f"ppo_pref_seed{seed}"
    )

    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128])
    )

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=ent_coef,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        seed=seed,
        verbose=verbose
    )

    if verbose > 0:
        print(f"Starting Preference PPO training for {total_timesteps} timesteps...")

    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=False
    )

    model.save(save_path)
    if verbose > 0:
        print(f"Preference PPO Model saved successfully to {save_path}")

    return model

