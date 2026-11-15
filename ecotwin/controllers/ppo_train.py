# """PPO training pipeline using Stable-Baselines3."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional
import os
import numpy as np

# Ensure OpenMP runtime fix is applied
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from ecotwin.env.ecotwin_env import EcoTwinEnv


class PPOAgent:
    """Wrapper around trained Stable-Baselines3 PPO model exposing standard act(obs) interface."""

    def __init__(self, model_path: Optional[str] = None, env: Optional[EcoTwinEnv] = None):
        self.model: Optional[PPO] = None
        self.name = "PPO Hand-Reward"
        if model_path is not None and Path(model_path).exists():
            self.load(model_path)
        elif env is not None:
            self.model = PPO("MlpPolicy", env, verbose=0)

    def load(self, model_path: str):
        self.model = PPO.load(model_path)

    def act(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        if self.model is None:
            # Fallback action if model not trained yet
            return np.array([1, 1, 1, 1, 1, 1, 2, 2], dtype=np.int64)
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return np.asarray(action, dtype=np.int64)

    def reset(self):
        pass


def train_ppo(
    total_timesteps: int = 15000,
    seed: int = 42,
    save_path: str = "models/ppo_hand_model.zip",
    tensorboard_log: str = "runs/ppo_hand",
    checkpoint_freq: int = 5000,
    randomize_scenarios: bool = True,
    reward_weights: Optional[Dict[str, float]] = None,
    learning_rate: float = 3e-4,
    n_steps: int = 384, # 4 episodes
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    ent_coef: float = 0.01,
    verbose: int = 1
) -> PPO:
    """Train PPO agent with hand-designed reward on EcoTwinEnv."""
    # Ensure output directories exist
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(tensorboard_log).mkdir(parents=True, exist_ok=True)
    run_name = Path(save_path).stem
    checkpoints_dir = Path("checkpoints") / run_name
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def make_env():
        return EcoTwinEnv(
            randomize_scenarios=randomize_scenarios,
            include_weather_forecast=True,
            reward_weights=reward_weights
        )

    vec_env = DummyVecEnv([make_env])

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1000, checkpoint_freq),
        save_path=str(checkpoints_dir),
        name_prefix=f"{run_name}_seed{seed}"
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
        print(f"Starting PPO training for {total_timesteps} timesteps with seed {seed}...")

    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=False
    )

    model.save(save_path)
    if verbose > 0:
        print(f"Model saved successfully to {save_path}")

    vec_env.close()
    return model


if __name__ == "__main__":
    train_ppo(total_timesteps=5000, verbose=1)

