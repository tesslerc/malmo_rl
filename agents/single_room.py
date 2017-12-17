import argparse
import json
import logging
import random
import re
import time
from typing import Tuple

import numpy as np

from agents.agent import Agent as BaseAgent


class Agent(BaseAgent):
    def __init__(self, params: argparse, port: int, start_malmo: bool, agent_index: int) -> None:
        super(Agent, self).__init__(params, port, start_malmo, agent_index)
        self.experiment_id: str = 'simple_room'

        self.reward_from_timeout_regex = re.compile(
            '<Reward.*description.*=.*\"command_quota_reached\".*reward.*=.*\"(.*[0-9]*)\".*/>', re.I)
        self.reward_for_sending_command_regex = re.compile('<RewardForSendingCommand.*reward="(.*)"/>', re.I)

        self.reward_from_success = 0
        self.number_of_steps = 0
        self.maximal_number_of_steps = 20
        self.touching_block = False

    def _restart_world(self) -> None:
        if not self.game_running:
            self._initialize_malmo_communication()

            mission_file = './agents/domains/basic.xml'
            with open(mission_file, 'r') as f:
                logging.debug('Agent[' + str(self.agent_index) + ']: Loading mission from %s.', mission_file)
                mission_xml = f.read()

                success = False
                while not success:
                    self._load_mission_from_xml(mission_xml)
                    success = self._wait_for_mission_to_begin()

                self.game_running = True

        while True:
            # Ensure that we don't start right on the block.
            x = random.randint(0, 6) + 0.5
            y = 55.0
            z = random.randint(0, 6) + 0.5
            if x != 2.5 or z != 5.5:
                break
        self.agent_host.sendCommand('chat /tp Cristina ' + str(x) + ' ' + str(y) + ' ' + str(z) + ' -180.0 0.0')

        time.sleep(2)
        # Generate random int between 0 and 3
        turn_direction = random.randint(0, 3)
        self.agent_host.sendCommand('turn ' + str(turn_direction))

        self.number_of_steps = 0

    def _manual_reward_and_terminal(self, action_command: str, reward: float, terminal: bool, state: np.ndarray,
                                    world_state) -> \
            Tuple[float, bool, np.ndarray, bool]:  # returns: reward, terminal, state, terminal due to timeout.
        msg = world_state.observations[-1].text
        observations = json.loads(msg)
        grid = observations.get(u'floor3x3', 0)

        self.number_of_steps += 1

        if action_command == 'new game':
            self.touching_block = False

        if (grid[10] == u'gold_block' or
                    grid[14] == u'gold_block' or
                    grid[16] == u'gold_block' or
                    grid[12] == u'gold_block'):
            if self.touching_block and action_command == 'move 1':
                self.touching_block = False
                return self.reward_from_success, True, state, False
            self.touching_block = True
        else:
            self.touching_block = False

        # Since basic agents don't have the notion of time, hence death due to timeout breaks the markovian assumption
        # of the problem. By setting terminal_due_to_timeout, different agents can decide if to learn or not from these
        # states, thus ensuring a more robust solution and better chances of convergence.
        if self.number_of_steps % self.maximal_number_of_steps == 0:
            return reward, True, state, True

        return reward, False, state, False
