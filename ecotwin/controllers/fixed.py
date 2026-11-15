"""Fixed schedule baseline controller."""
from __future__ import annotations
import numpy as np


class FixedController:
    """Fixed baseline controller: continuous machine operation and constant 24°C setpoint."""

    def __init__(self, name: str = "Fixed Schedule"):
        self.name = name

    def act(self, observation: np.ndarray) -> np.ndarray:
        """Exposes standard act(observation) -> action interface.

        Returns
        -------
        np.ndarray
            Action vector [M1..M6, Z1, Z2].
            Always run machines (1) and maintain constant comfort setpoint 24°C (index 2).
        """
        return np.array([1, 1, 1, 1, 1, 1, 2, 2], dtype=np.int64)

    def reset(self):
        pass

