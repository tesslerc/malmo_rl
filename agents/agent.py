import argparse
import logging
import os
import re
import subprocess
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple

import agents.malmo_dependencies.MalmoPython as MalmoPython
import numpy as np
from PIL import Image


# TODO: Refactor error checking code to be more secure and robust!


class Agent(ABC):
    """An agent abstract base class.
    Defines the methods which are required to be implemented by all agents deriving from this class.
    This class also defines the base behavior for communication with the environment.
    """

    def __init__(self, params: argparse, port: int, start_malmo: bool) -> None:
        self.params = params

        # Malmo interface.
        self.minecraft = None
        self.MalmoPython = MalmoPython
        self.agent_host = self.MalmoPython.AgentHost()
        self.malmo_port = port

        # If no port is given, start a Malmo instance with a random port number.
        if start_malmo:
            self._start_malmo()

        # These are the commands supported by our simulator, different policies can support different actions as long as
        # they are sub-sets of this action set.
        self.supported_actions = [
            'move 1',  # W
            'move -1',  # S
            'turn -1',  # A
            'turn 1',  # D
            'attack 1',  # Q
            'new game'  # 9
        ]

        # Add the default client - on the local machine:
        self.client = MalmoPython.ClientInfo("127.0.0.1", int(self.malmo_port))
        self.client_pool = MalmoPython.ClientPool()
        self.client_pool.add(self.client)

        # Start the mission.
        self.tick_regex = re.compile('<MsPerTick>([0-9]*)</MsPerTick>', re.I)
        self.tick_time = 200  # Default value, in milliseconds.
        self.experiment_id = str(self.malmo_port)
        self.mission_restart_print_frequency = 10
        self.action_retry_threshold = 10

    def _start_malmo(self) -> None:
        # Minecraft directory is found via environment variable, this is required as a part of the Malmo installation
        # process. The MALMO_XSD_PATH points to the ..../Malmo/Schemas folder whereas Minecraft is a subdirectory of
        # Malmo.
        minecraft_directory = Path(os.environ['MALMO_XSD_PATH']).parent.joinpath('Minecraft')
        if self.params.platform is 'linux':
            minecraft_path = str(minecraft_directory) + '\launchClient.sh'
        elif self.params.platform is 'win':
            minecraft_path = str(minecraft_directory) + '\launchClient.bat'
        else:
            raise NotImplementedError('Only windows and linux are currently supported.')

        # Keep CWD as a pointer of where to return to.
        working_directory = os.getcwd()
        # Loading malmo is required from within the Minecraft folder (otherwise we will encounter issues with gradle).
        os.chdir(str(minecraft_directory))
        self.minecraft = subprocess.Popen([minecraft_path, '-port', str(self.malmo_port)],
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        logging.info('Starting Malmo:')
        current_line = ''
        # Emit Malmo console logs to console up to the point where it is fully loaded.
        for c in iter(lambda: self.minecraft.stdout.read(1), ''):
            if c == b'\n':
                logging.debug(current_line)
                # Ugly workaround to check that Malmo has properly loaded.
                if 'CLIENT enter state: DORMANT' in current_line:
                    break
                current_line = ''
            else:
                current_line += c.decode('utf-8')

        logging.info('Malmo loaded!')

        os.chdir(working_directory)

    @abstractmethod
    def _restart_world(self) -> None:
        # Restart world is called upon new mission (new game, mission stuck, agent dead, etc...).
        pass

    def _load_mission_from_xml(self, mission_xml: str) -> None:
        # Given a string variable (containing the mission XML), will reload the mission itself.
        tick_time = self.tick_regex.search(mission_xml)
        if tick_time is not None:
            self.tick_time = int(tick_time.group(1))

        mission = self.MalmoPython.MissionSpec(mission_xml, True)
        mission.forceWorldReset()
        mission_record = self.MalmoPython.MissionRecordSpec()

        retry = 0
        while not self.agent_host.getWorldState().is_mission_running:
            try:
                logging.debug('Restarting the mission.')
                self.agent_host.startMission(mission, self.client_pool, mission_record, 0, str(self.experiment_id))
                break
            except RuntimeError as e:
                retry += 1
                if retry % self.mission_restart_print_frequency == 0:
                    logging.critical('_load_mission_from_xml, Error starting mission', e)
                time.sleep(2)

    def _wait_for_mission_to_begin(self) -> bool:
        world_state = self.agent_host.getWorldState()
        retry = 0
        while not world_state.has_mission_begun:
            time.sleep(1)
            world_state = self.agent_host.getWorldState()
            for error in world_state.errors:
                logging.error('_wait_for_mission_to_begin, Error: ' + error.text)
            retry += 1
            if retry >= 20:
                return False
        return True

    def perform_action(self, action_command: str) -> Tuple[float, bool, np.ndarray, bool]:
        assert (action_command in self.supported_actions)
        retries = 0
        while True:
            if action_command == 'new game':
                self._restart_world()
                reward, terminal, state, world_state, action_succeeded = self._get_new_state(True)
                reward, terminal, state, terminal_due_to_timeout = self._manual_reward_and_terminal(reward, terminal,
                                                                                                    state,
                                                                                                    world_state)
                if action_succeeded:
                    return reward, terminal, state, terminal_due_to_timeout
            else:
                self.agent_host.sendCommand(action_command)
                reward, terminal, state, world_state, action_succeeded = self._get_new_state(False)
                reward, terminal, state, terminal_due_to_timeout = self._manual_reward_and_terminal(reward, terminal,
                                                                                                    state,
                                                                                                    world_state)
                if action_succeeded:
                    return reward, terminal, state, terminal_due_to_timeout

            retries += 1
            if retries >= 10:
                return 0, True, np.empty(0), True

    def _get_new_state(self, new_game: bool) -> Tuple[float, bool, np.ndarray, MalmoPython.WorldState, bool]:
        # Returns: reward, terminal, state, world_state, was action a success or not.
        current_r = 0
        number_of_attempts = 0
        while True:
            time.sleep(3 * self.tick_time / 1000.0)
            world_state, r = self._get_updated_world_state()
            current_r += r

            # TODO: This is an issue... waiting for non-zero reward if a zero reward scenario is a legit option.
            if current_r != 0 or new_game:
                if world_state.is_mission_running and len(world_state.observations) > 0 \
                        and not (world_state.observations[-1].text == "{}") and len(world_state.video_frames) > 0:
                    frame = world_state.video_frames[-1]
                    state = Image.frombytes('RGB', (frame.width, frame.height), bytes(frame.pixels))
                    preprocessed_state = self._preprocess_state(state, self.params.image_width,
                                                                self.params.image_height, not self.params.retain_rgb)
                    return current_r, False, preprocessed_state, world_state, True
                elif not world_state.is_mission_running and not new_game:
                    return current_r, True, np.empty(0), world_state, True

            number_of_attempts += 1

            if number_of_attempts >= self.action_retry_threshold:
                logging.error('_get_new_state, Unable to retrieve state.')
                return 0, False, np.empty(0), None, False

    def _get_updated_world_state(self) -> Tuple[MalmoPython.WorldState, float]:
        world_state = self.agent_host.getWorldState()
        r = 0
        for error in world_state.errors:
            logging.error('_get_updated_world_state, Error: ' + error.text)
        for reward in world_state.rewards:
            r += reward.getValue()
        return world_state, r

    def _manual_reward_and_terminal(self, reward: float, terminal: bool, state: np.ndarray, world_state: object) -> \
            Tuple[float, bool, np.ndarray, bool]:
        del world_state  # Not used in the base implementation.
        terminal_due_to_timeout = False  # Default behavior is allow training on all states.
        return reward, terminal, state, terminal_due_to_timeout

    @staticmethod
    def _preprocess_state(state: Image.Image, width: int, height: int, gray_scale: bool) -> np.ndarray:
        preprocessed_state = state.resize((width, height))
        if gray_scale:
            preprocessed_state = preprocessed_state.convert('L')  # Grayscale conversion.
        else:
            preprocessed_state = preprocessed_state.convert('RGB')  # Any convert op. is required for numpy parsing.
        return np.array(preprocessed_state)
