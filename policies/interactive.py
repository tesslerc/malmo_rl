import argparse
from typing import Dict, List

import numpy as np

from policies.policy import Policy as AbstractPolicy


class Policy(AbstractPolicy):
    def __init__(self, params: argparse.Namespace) -> None:
        super(Policy, self).__init__(params)

        self.getch = None
        if self.params.platform == 'linux':
            from getch import getch
        elif self.params.platform == 'win':
            from msvcrt import getch
        else:
            print('Unsupported OS: ' + self.params.platform)
            exit(0)

        self.getch = getch

        self.action_mapping: Dict[int, str] = {
            119: 'move 1',  # W
            115: 'move -1',  # S
            97: 'turn -1',  # A
            100: 'turn 1',  # D
            101: 'attack 1',  # E
            113: 'quit',  # Q (quit)
        }

    def get_action(self, states: List[np.ndarray], is_train: bool) -> List[str]:
        del is_train

        if self.params.viz is not None:
            # Send screen of each agent to visdom.
            images = np.zeros((self.params.number_of_agents, 3, self.params.image_width, self.params.image_height))
            for idx in range(self.params.number_of_agents):
                if self.params.retain_rgb:
                    images[idx, :, :, :] = states[idx]
                else:
                    images[idx, 1, :, :] = states[idx]
                self.params.viz.image(images[idx], win='state_agent_' + str(idx),
                                      opts=dict(title='Agent ' + str(idx) + '\'s state'))

        while True:
            try:
                key_code = ord(self.getch())
                if key_code not in self.action_mapping:
                    print('Invalid key pressed, try again...')
                else:
                    break
            except OverflowError:
                print('Bad key pressed.')

        action_command = self.parse_action(key_code)
        if action_command == 'quit':
            print('Ending simulation.')
            exit(0)

        return [action_command for _ in range(self.params.number_of_agents)]

    def parse_action(self, key_code: int) -> str:
        return self.action_mapping[key_code]

    def update_observation(self, reward: float, terminal: bool, terminal_due_to_timeout: bool, success: bool,
                           is_train: bool) -> None:
        pass

    def save_state(self):
        pass

    def load_state(self):
        pass
