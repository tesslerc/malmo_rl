import argparse
from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np


class Policy(ABC):
    """A policy abstract base class.
    Defines the methods which are required to be implemented by all policies deriving from this class.
    """

    def __init__(self, params: argparse) -> None:
        self.params: argparse = params

    @abstractmethod
    def get_action(self, state: np.ndarray, is_train: bool) -> Tuple[int, Dict[str, float]]:
        # An agent which wishes to learn from terminal_due_to_timeout, should set this value to False on all
        # occurrences. This way the replay_memory and other utilities will not treat those transitions as different.
        pass

    @abstractmethod
    def update_observation(self, reward: float, terminal: bool, terminal_due_to_timeout: bool, is_train: bool) -> None:
        pass
