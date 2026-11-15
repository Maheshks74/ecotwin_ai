"""Neural Reward Model trained on trajectory preferences with Bradley-Terry loss."""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple
import os

# OpenMP duplicate runtime fix
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from ecotwin.preference.simulated_manager import TrajectorySegment


class RewardNet(nn.Module):
    """Small PyTorch MLP estimating step-level reward r_hat(s, a)."""

    def __init__(self, obs_dim: int = 58, act_dim: int = 8, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = obs_dim + act_dim
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        state : torch.Tensor
            Observation tensor, shape (B, obs_dim).
        action : torch.Tensor
            Action tensor, shape (B, act_dim).

        Returns
        -------
        torch.Tensor
            Step reward scalar, shape (B, 1).
        """
        # Normalize actions to roughly [-1, 1]
        act_norm = (action.float() - 1.0) / 2.0
        x = torch.cat([state.float(), act_norm], dim=-1)
        return self.net(x)


class PreferenceRewardModel:
    """Wrapper managing training, evaluation, and inference of the preference reward network."""

    def __init__(
        self,
        obs_dim: int = 58,
        act_dim: int = 8,
        hidden_dim: int = 64,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.model = RewardNet(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=hidden_dim).to(self.device)

    def score_segment(self, segment: TrajectorySegment) -> torch.Tensor:
        """Computes total trajectory score R_hat(tau) = sum_t r_hat(s_t, a_t)."""
        states_t = torch.tensor(segment.states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(segment.actions, dtype=torch.float32, device=self.device)
        step_rewards = self.model(states_t, actions_t)
        return step_rewards.sum()

    def train_on_pairs(
        self,
        train_pairs: List[Tuple[TrajectorySegment, TrajectorySegment, int]],
        val_pairs: Optional[List[Tuple[TrajectorySegment, TrajectorySegment, int]]] = None,
        epochs: int = 25,
        lr: float = 1e-3,
        batch_size: int = 16,
        save_path: Optional[str] = "models/reward_model.pt",
        verbose: int = 1
    ) -> float:
        """Train reward network using Bradley-Terry preference loss."""
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        num_train = len(train_pairs)
        self.model.train()

        for epoch in range(epochs):
            indices = np.random.permutation(num_train)
            total_loss = 0.0

            for i in range(0, num_train, batch_size):
                batch_indices = indices[i:i + batch_size]
                optimizer.zero_grad()

                logits_diff = []
                targets = []

                for idx in batch_indices:
                    seg1, seg2, y = train_pairs[idx]
                    r1 = self.score_segment(seg1)
                    r2 = self.score_segment(seg2)
                    logits_diff.append(r1 - r2)
                    targets.append(float(y))

                diff_t = torch.stack(logits_diff)
                target_t = torch.tensor(targets, dtype=torch.float32, device=self.device)

                loss = criterion(diff_t, target_t)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch_indices)

            avg_loss = total_loss / num_train

        val_acc = self.evaluate_accuracy(val_pairs if val_pairs else train_pairs)
        if verbose > 0:
            print(f"Reward Model Training Complete. Final Val Accuracy: {val_acc:.2%}")

        if save_path:
            torch.save(self.model.state_dict(), save_path)
            if verbose > 0:
                print(f"Saved reward model weights to {save_path}")

        return val_acc

    def evaluate_accuracy(
        self,
        pairs: List[Tuple[TrajectorySegment, TrajectorySegment, int]]
    ) -> float:
        """Computes held-out preference classification accuracy."""
        self.model.eval()
        correct = 0
        with torch.no_grad():
            for seg1, seg2, y in pairs:
                r1 = self.score_segment(seg1).item()
                r2 = self.score_segment(seg2).item()
                pred = 1 if r1 >= r2 else 0
                if pred == y:
                    correct += 1
        return float(correct / max(1, len(pairs)))

    def load(self, model_path: str):
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict_step_reward(self, state: np.ndarray, action: np.ndarray) -> float:
        """Inference for a single step transition (s, a)."""
        self.model.eval()
        with torch.no_grad():
            s_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            a_t = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
            r = self.model(s_t, a_t)
            return float(r.squeeze().item())

