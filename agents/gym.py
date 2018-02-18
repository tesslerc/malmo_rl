import argparse
from typing import Tuple
import gym
import numpy as np
from PIL import Image
from agents.agent import Agent as BaseAgent


class Agent(BaseAgent):
    def __init__(self, params: argparse, port: int, start_malmo: bool, agent_index: int) -> None:
        del start_malmo  # Not relevant.
        super(Agent, self).__init__(params, port, False, agent_index)
        self.env = gym.make(params.gym_env)
        self.env = self.env.unwrapped
        self.n_actions = self.env.action_space.n
        self.supported_actions = []

        for action_idx in range(self.n_actions):
            self.supported_actions.append(str(action_idx))

    def _restart_world(self, is_train: bool):
        observation = np.array(self.env.reset())
        observation = Image.fromarray(observation, 'RGB')
        observation = self._preprocess_state(observation, self.params.image_width,
                                             self.params.image_height, not self.params.retain_rgb)
        return observation

    def perform_action(self, action_command: str, is_train: bool) -> Tuple[float, bool, np.ndarray, bool, bool]:
        if action_command == 'new game':
            return 0, False, self._restart_world(is_train), False, False

        action_command = int(action_command)
        observation, reward, done, info = self.env.step(action_command)

        observation = np.array(observation)
        observation = Image.fromarray(observation, 'RGB')
        observation = self._preprocess_state(observation, self.params.image_width,
                                             self.params.image_height, not self.params.retain_rgb)

        return reward, done, observation, False, False
