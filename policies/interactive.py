from policies.policy import Policy as AbstractPolicy
try:
    from getch import getch
except:
    from msvcrt import getch


class Policy(AbstractPolicy):
    def __init__(self, params):
        super(Policy, self).__init__(params)

        self.action_mapping = {
            119:  0,  # W
            115:  1,  # S
            97:   2,  # A
            100:  3,  # D
            101:  4,  # E
            57:   9,  # 9 (new game)
            113: -1,  # Q (quit)
        }

    def get_action(self, reward, terminal, state, is_train):
        del state
        del reward
        if terminal:
            print('Game over! Starting a new one!')
            return 9

        key_code = ord(getch())
        while key_code not in self.action_mapping:
            print('Invalid key pressed, try again...')
            key_code = ord(getch())

        action_index = self.parse_action(key_code)
        if action_index < 0:
            print('Ending simulation.')
            exit(0)

        return action_index

    def parse_action(self, key_code):
        return self.action_mapping[key_code]
