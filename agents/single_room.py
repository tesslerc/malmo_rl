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

        self.reward_from_timeout = None

    def _restart_world(self) -> None:
        mission_file = './agents/domains/basic.xml'
        with open(mission_file, 'r') as f:
            logging.debug('Loading mission from %s.', mission_file)
            mission_xml = f.read()

            reward_from_timeout = self.reward_from_timeout_regex.search(mission_xml)
            if reward_from_timeout is not None:
                self.reward_from_timeout = int(reward_from_timeout.group(1))
                reward_for_sending_command = self.reward_for_sending_command_regex.search(mission_xml)
                if reward_for_sending_command is not None:
                    self.reward_from_timeout += int(reward_for_sending_command.group(1))

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
            terminal_due_to_timeout = reward == self.reward_from_timeout
        else:
            terminal_due_to_timeout = False
        return reward, terminal, state, terminal_due_to_timeout
