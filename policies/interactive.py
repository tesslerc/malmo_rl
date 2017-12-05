import argparse
from typing import Dict, List

import numpy as np

from policies.policy import Policy as AbstractPolicy

try:
    from getch import getch
except:
    from msvcrt import getch


class Policy(AbstractPolicy):
    def __init__(self, params: argparse.Namespace) -> None:
        super(Policy, self).__init__(params)

        self.action_mapping: Dict[int, str] = {
            119: 'move 1',  # W
            115: 'move -1',  # S
            97: 'turn -1',  # A
            100: 'turn 1',  # D
            101: 'attack 1',  # E
            113: 'quit',  # Q (quit)
        }

    def get_action(self, state: List[np.ndarray], is_train: bool) -> List[str]:
        del state
        del is_train

        key_code = ord(getch())
        while key_code not in self.action_mapping:
            print('Invalid key pressed, try again...')
            key_code = ord(getch())

        action_command = self.parse_action(key_code)
        if action_command == 'quit':
            print('Ending simulation.')
            exit(0)

        return [action_command]

    def parse_action(self, key_code: int) -> str:
        return self.action_mapping[key_code]

    def update_observation(self, reward: float, terminal: bool, terminal_due_to_timeout: bool, is_train: bool) -> None:
        pass
