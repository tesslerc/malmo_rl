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

    def _restart_world(self) -> None:
        self.state = 0

    def perform_action(self, action_command: str) -> Tuple[float, bool, np.ndarray, bool]:
        zeros_matrix = np.zeros((84, 84))
        ones_matrix = np.ones_like(zeros_matrix).astype(float)

        self.steps += 1
        if self.steps >= 100:
            self.steps = 0
            self.state = 0
            return -1, True, ones_matrix, True

        if self.state == 0:
            if action_command == 'turn 1':
                self.state = 1
                return -1, False, ones_matrix, False
            else:
                return -1, False, zeros_matrix, False
        else:  # self.state == 1
            self.state = 0
            if action_command == 'move 1':
                return 0, True, zeros_matrix, False
            else:
                return -1, False, zeros_matrix, False
