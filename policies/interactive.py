import argparse
from typing import Dict, Tuple

import numpy as np

from policies.policy import Policy as AbstractPolicy

try:
    from getch import getch
except:
    from msvcrt import getch


class Policy(AbstractPolicy):
    def __init__(self, params: argparse.Namespace) -> None:
        super(Policy, self).__init__(params)

        self.action_mapping: Dict[int, int] = {
            119:  0,  # W
            115:  1,  # S
            97:   2,  # A
            100:  3,  # D
            101:  4,  # E
            57:   9,  # 9 (new game)
            113: -1,  # Q (quit)
        }

    def get_action(self, reward: float, terminal: bool, terminal_due_to_timeout: bool, observation: np.ndarray,
                   is_train: bool) -> Tuple[int, Dict[str, float]]:
        del observation
        del reward
        del terminal_due_to_timeout
        if terminal:
            print('Game over! Starting a new one!')
            return 9, {}

        key_code = ord(getch())
        while key_code not in self.action_mapping:
            print('Invalid key pressed, try again...')
            key_code = ord(getch())

        action_index = self.parse_action(key_code)
        if action_index < 0:
            print('Ending simulation.')
            exit(0)

        return action_index, {}

    def parse_action(self, key_code: int) -> int:
        return self.action_mapping[key_code]
