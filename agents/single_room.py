import argparse
import logging
import re
from typing import Tuple

import numpy as np

from agents.agent import Agent as BaseAgent


class Agent(BaseAgent):
    def __init__(self, params: argparse, port: int, start_malmo: bool) -> None:
        super(Agent, self).__init__(params, port, start_malmo)
        self.experiment_id: str = 'simple_room'

        self.reward_from_timeout_regex = re.compile(
            '<Reward.*description.*=.*\"command_quota_reached\".*reward.*=.*\"(.*[0-9]*)\".*/>', re.I)
        self.reward_for_sending_command_regex = re.compile('<RewardForSendingCommand.*reward="(.*)"/>', re.I)

        self.reward_from_timeout = -6  # command_quota_reached + RewardForSendingCommand
        self.reward_from_success = -0.5  # RewardForSendingCommand + found_goal

    def _restart_world(self) -> None:
        mission_file = './agents/domains/basic.xml'
        with open(mission_file, 'r') as f:
            logging.debug('Loading mission from %s.', mission_file)
            mission_xml = f.read()

            success = False
            while not success:
                self._load_mission_from_xml(mission_xml)
                success = self._wait_for_mission_to_begin()

    def _manual_reward_and_terminal(self, reward: float, terminal: bool, state: np.ndarray, world_state: object) -> \
            Tuple[float, bool, np.ndarray, bool]:
        del world_state  # Not used in the base implementation.

        # Since basic agents don't have the notion of time, hence death due to timeout breaks the markovian assumption
        # of the problem. By setting terminal_due_to_timeout, different agents can decide if to learn or not from these
        # states, thus ensuring a more robust solution and better chances of convergence.
        if self.reward_from_timeout is not None:
            if reward == self.reward_from_timeout:
                terminal_due_to_timeout = True
                terminal = True
            else:
                terminal_due_to_timeout = False
        else:
            terminal_due_to_timeout = False

        if reward == self.reward_from_success:
            terminal = True

        return reward, terminal, state, terminal_due_to_timeout
