import argparse
import logging
import os
import random
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

    def __init__(self, params: argparse, port: int, start_malmo: bool, agent_index: int) -> None:
        self.params = params
        self.agent_index = agent_index
        # Malmo interface.
        self.minecraft = None
        self.MalmoPython = MalmoPython
        self.agent_host = None
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

        self.client = None
        self.client_pool = None

        # Start the mission.
        self.tick_regex = re.compile('<MsPerTick>[0-9]*</MsPerTick>', re.I)
        self.tick_time = self.params.ms_per_tick  # Default value, in milliseconds.
        self.experiment_id = None
        self.mission_restart_print_frequency = 10
        self.action_retry_threshold = 10

        self.game_running = False

    def _initialize_malmo_communication(self):
        logging.debug('Agent[' + str(self.agent_index) + ']: Initializing Malmo communication.')
        # Add the default client - on the local machine:
        self.agent_host = self.MalmoPython.AgentHost()
        self.client = MalmoPython.ClientInfo("127.0.0.1", int(self.malmo_port))
        self.client_pool = MalmoPython.ClientPool()
        self.client_pool.add(self.client)
        self.experiment_id = str(random.randint(0, 10000))

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

        logging.info('Agent[' + str(self.agent_index) + ']: Starting Malmo:')
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

        logging.info('Agent[' + str(self.agent_index) + ']: Malmo loaded!')

        os.chdir(working_directory)

    @abstractmethod
    def _restart_world(self) -> None:
        # Restart world is called upon new mission (new game, mission stuck, agent dead, etc...).
        pass

    def _load_mission_from_xml(self, mission_xml: str) -> None:
        logging.debug('Agent[' + str(self.agent_index) + ']: Loading mission from XML.')
        # Given a string variable (containing the mission XML), will reload the mission itself.

        mission_xml = self.tick_regex.sub('<MsPerTick>' + str(self.params.ms_per_tick) + '</MsPerTick>', mission_xml)

        mission = self.MalmoPython.MissionSpec(mission_xml, True)
        # mission.forceWorldReset()
        mission_record = self.MalmoPython.MissionRecordSpec()

        while not self.agent_host.getWorldState().has_mission_begun:
            try:
                number_of_attempts = 0
                logging.debug('Agent[' + str(self.agent_index) + ']: Restarting the mission.')
                time.sleep(5 * self.tick_time / 1000.0)
                self.agent_host.startMission(mission, self.client_pool, mission_record, 0, str(self.experiment_id))
                while not self.agent_host.getWorldState().has_mission_begun:
                    number_of_attempts += 1
                    time.sleep(1)
                    if number_of_attempts >= 20:
                        break
            except RuntimeError as e:
                logging.critical(
                    'Agent[' + str(self.agent_index) + ']: _load_mission_from_xml, Error starting mission', e)

    def _wait_for_mission_to_begin(self) -> bool:
        logging.debug('Agent[' + str(self.agent_index) + ']: Waiting for mission to begin.')
        world_state = self.agent_host.getWorldState()
        number_of_attempts = 0
        while not world_state.has_mission_begun:
            time.sleep(5)
            world_state = self.agent_host.getWorldState()
            for error in world_state.errors:
                logging.error('Agent[' + str(self.agent_index) + ']: _wait_for_mission_to_begin, Error: ' + error.text)
            number_of_attempts += 1
            if number_of_attempts >= 100:
                return False
        return True

    def perform_action(self, action_command: str) -> Tuple[float, bool, np.ndarray, bool]:
        assert (action_command in self.supported_actions)
        number_of_attempts = 0
        logging.debug('Agent[' + str(self.agent_index) + ']: received command ' + action_command)
        while True:
            if action_command == 'new game':
                self._restart_world()
                reward, terminal, state, world_state, action_succeeded = self._get_new_state(True)
                if action_succeeded:
                    return self._manual_reward_and_terminal(action_command, reward, terminal, state, world_state)
            else:
                self.agent_host.sendCommand(action_command)
                reward, terminal, state, world_state, action_succeeded = self._get_new_state(False)
                if action_succeeded:
                    return self._manual_reward_and_terminal(action_command, reward, terminal, state, world_state)

            number_of_attempts += 1
            time.sleep(3 * self.tick_time / 1000.0)
            if number_of_attempts >= 100:
                logging.error('Agent[' + str(self.agent_index) + ']: Failed to send action.')
                self.game_running = False
                return 0, True, np.empty(0), True

    def _get_new_state(self, new_game: bool) -> Tuple[float, bool, np.ndarray, MalmoPython.WorldState, bool]:
        logging.debug('Agent[' + str(self.agent_index) + ']: _get_new_state.')
        # Returns: reward, terminal, state, world_state, was action a success or not.
        current_r = 0
        number_of_attempts = 0
        while True:
            time.sleep(3 * self.tick_time / 1000.0)
            world_state, r = self._get_updated_world_state()
            current_r += r

            # TODO: This is an issue... waiting for non-zero reward if a zero reward scenario is a legit option.
            if world_state is not None:
                if world_state.is_mission_running and len(world_state.observations) > 0 \
                        and not (world_state.observations[-1].text == "{}") and len(world_state.video_frames) > 0:
                    frame = world_state.video_frames[-1]
                    state = Image.frombytes('RGB', (frame.width, frame.height), bytes(frame.pixels))
                    preprocessed_state = self._preprocess_state(state, self.params.image_width,
                                                                self.params.image_height, not self.params.retain_rgb)
                    return current_r, False, preprocessed_state, world_state, True
                elif not world_state.is_mission_running and not new_game:
                    self.game_running = False
                    return current_r, True, np.empty(0), world_state, True

            number_of_attempts += 1
            if number_of_attempts >= 100:
                logging.error('Agent[' + str(self.agent_index) + ']: _get_new_state, Unable to retrieve state.')
                self.game_running = False
                return 0, False, np.empty(0), None, False

    def _get_updated_world_state(self) -> Tuple[MalmoPython.WorldState, float]:
        logging.debug('Agent[' + str(self.agent_index) + ']: _get_updated_world_state.')
        try:
            world_state = self.agent_host.peekWorldState()
            number_of_attempts = 0
            while world_state.is_mission_running and all(e.text == '{}' for e in world_state.observations):
                time.sleep(self.tick_time / 1000.0)
                world_state = self.agent_host.peekWorldState()
                number_of_attempts += 1
                if number_of_attempts >= 100:
                    logging.error('Agent[' + str(self.agent_index) + ']: _get_updated_world_state error in first loop.')
                    return None, 0
            # wait for a frame to arrive after that
            num_frames_seen = world_state.number_of_video_frames_since_last_state
            number_of_attempts = 0
            while world_state.is_mission_running and \
                    world_state.number_of_video_frames_since_last_state == num_frames_seen:
                time.sleep(self.tick_time / 1000.0)
                world_state = self.agent_host.peekWorldState()
                number_of_attempts += 1
                if number_of_attempts >= 100:
                    logging.error(
                        'Agent[' + str(self.agent_index) + ']: _get_updated_world_state error in second loop.')
                    return None, 0

            world_state = self.agent_host.getWorldState()
            for error in world_state.errors:
                logging.error('Agent[' + str(self.agent_index) + ']: _get_updated_world_state, Error: ' + error.text)
            current_r = sum(r.getValue() for r in world_state.rewards)
            return world_state, current_r
        except Exception as e:
            logging.error(
                'Agent[' + str(self.agent_index) + ']: _get_updated_world_state exception met: ' + str(e.format_exc()))
            return None, 0

    def _manual_reward_and_terminal(self, action_command: str, reward: float, terminal: bool, state: np.ndarray,
                                    world_state: object) -> Tuple[float, bool, np.ndarray, bool]:
        del world_state, action_command  # Not used in the base implementation.
        terminal_due_to_timeout = False  # Default behavior is allow training on all states.
        return reward, terminal, state, terminal_due_to_timeout

    @staticmethod
    def _preprocess_state(state: Image.Image, width: int, height: int, gray_scale: bool) -> np.ndarray:
        preprocessed_state = state.resize((width, height))
        if gray_scale:
            preprocessed_state = preprocessed_state.convert('L')  # Grayscale conversion.
        else:
            preprocessed_state = preprocessed_state.convert('RGB')  # Any convert op. is required for numpy parsing.
        return (np.array(preprocessed_state).astype(float)) / 255.0
