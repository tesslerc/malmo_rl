import argparse
from typing import Tuple

import numpy as np

from agents.agent import Agent as BaseAgent


class Agent(BaseAgent):
    """
    Sanity check domain is a simple 2 state markov chain to quickly check that the algorithms are properly connected
    and working.
    The chain works as such:
        State 1: turn right will move to State 2, other actions leave the agent at state 1.
        State 2: move forward will move the agent to the termination state, other actions will cause it to return to
            state 1.

        All actions cause a -1 reward penalty, ensuring that the correct behavior is to quickly end the game.
    """

    def __init__(self, params: argparse, port: int, start_malmo: bool, agent_index: int) -> None:
        del start_malmo  # Not relevant.
        super(Agent, self).__init__(params, port, False, agent_index)
        self.state = 0
        self.steps = 0

        self.supported_actions = [
            '1',
            '2',
            '3',
            'new game'
        ]

    def _restart_world(self, is_train: bool) -> None:
        del is_train

        self.state = 0

    def perform_action(self, action_command: str, is_train) -> Tuple[float, bool, np.ndarray, bool, bool]:
        if self.params.retain_rgb:
            zeros_matrix = np.zeros((3, 84, 84)).astype(float)
            ones_matrix = np.zeros((3, 84, 84)).astype(float)
            ones_matrix[0, :, :] = 1
        else:
            zeros_matrix = np.zeros((84, 84)).astype(float)
            ones_matrix = np.ones((84, 84)).astype(float)

        if action_command == 'new game':
            self._restart_world(is_train)
            return 0, False, zeros_matrix, False, False

        random_reward = min(max(np.random.normal(-1, 0.1), -2), 0)

        self.steps += 1
        if self.steps >= 10:
            self.steps = 0
            self.state = 0
            return -1, True, ones_matrix, True, False

        if self.state == 0:
            if action_command == '3':
                self.state = 1
                return random_reward, False, ones_matrix, False, False
            else:
                return random_reward, False, zeros_matrix, False, False
        else:  # self.state == 1
            self.state = 0
            if action_command == '1':
                self.steps = 0
                return 0, True, zeros_matrix, False, True
            else:
                return random_reward, False, zeros_matrix, False, False
