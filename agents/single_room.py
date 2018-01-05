import argparse
import logging
import random
import time
from typing import Tuple

import numpy as np

from agents.agent import Agent as BaseAgent


class Agent(BaseAgent):
    def __init__(self, params: argparse, port: int, start_malmo: bool, agent_index: int) -> None:
        super(Agent, self).__init__(params, port, start_malmo, agent_index)
        self.experiment_id: str = 'simple_room'
        self.reward_from_success = 0

    def _restart_world(self) -> None:
        self._initialize_malmo_communication()

        mission_file = './agents/domains/basic.xml'
        with open(mission_file, 'r') as f:
            logging.debug('Agent[' + str(self.agent_index) + ']: Loading mission from %s.', mission_file)
            mission_xml = f.read()

            success = False
            while not success:
                mission = self._load_mission_from_xml(mission_xml)
                self._load_mission_from_missionspec(mission)
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

    def _manual_reward_and_terminal(self, action_command: str, reward: float, terminal: bool, state: np.ndarray,
                                    world_state) -> \
            Tuple[float, bool, np.ndarray, bool, bool]:  # returns: reward, terminal, state, timeout, success.
        del world_state  # Not in use here.

        if reward > 0:
            # Reached goal successfully.
            return self.reward_from_success, True, state, False, True

        # Since basic agents don't have the notion of time, hence death due to timeout breaks the markovian assumption
        # of the problem. By setting terminal_due_to_timeout, different agents can decide if to learn or not from these
        # states, thus ensuring a more robust solution and better chances of convergence.
        if reward < -5:
            return -1, True, state, True, False

        return -1, False, state, False, False
