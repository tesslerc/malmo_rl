import argparse
from collections import namedtuple

import numpy as np

# Definition of observation:
#   state:    s_t
#   action:   a_t
#   reward:   r_(t+1) (reward received due to being at state s_t and performing action a_t which transitions to state
#                       s_(t+1))
#   terminal: t_(t+1) (whether or not the next state is a terminal state)
slim_observation = namedtuple('slim_observation', 'state, action, reward, terminal, terminal_due_to_timeout')
observation = namedtuple('observation', 'state, action, reward, terminal, next_state')


class ReplayMemory(object):
    def __init__(self, params: argparse) -> None:
        self.params = params
        self.memory_size = params.replay_memory_size
        self.batch_size = params.batch_size
        self.state_size = params.state_size
        self.memory = []
        self.elements_in_memory = 0

    def add_observation(self, state: object, action: int, reward: float, terminal: int,
                        terminal_due_to_timeout: bool) -> None:
        self.memory.append(slim_observation(state=state, action=action, reward=reward, terminal=terminal,
                                            terminal_due_to_timeout=terminal_due_to_timeout))
        if self.elements_in_memory >= self.memory_size:
            self.memory.pop(0)
        else:
            self.elements_in_memory += 1

    def sample(self):
        # Returns: states, actions, rewards, termination values, next states
        mini_batch = []
        # Starting from index 3 (to enable building of a full state), and up to 'self.elements_in_memory - 1' to make
        # sure the next state exists.
        training_samples = np.random.choice(3, self.elements_in_memory - 1, self.batch_size)
        for index in range(self.batch_size):
            while (self.memory[training_samples[index - 1]].terminal is True or
                   self.memory[training_samples[index]].terminal_due_to_timeout is True):
                training_samples[index] = np.random.choice(3, self.elements_in_memory - 1)

            obs = self.memory[training_samples[index]]
            state = self._build_state(training_samples[index])
            if obs.terminal is False:
                next_state = self._build_state(training_samples[index] + 1)
            else:
                # Instead of trying to infer state size, just return a state. The terminal flag denotes to disregard
                # this 'next_state' object.
                next_state = state
            mini_batch.append(observation(state=state, action=obs.action, reward=obs.reward, terminal=obs.terminal,
                                          next_state=next_state))
        # Returns tuple: batch_state, batch_action, batch_reward, batch_terminal, batch_next_state
        return zip(*mini_batch)

    def _build_state(self, final_index: int) -> np.ndarray:
        state = []
        saw_terminal = False
        for i in range(0, self.params.state_size):
            if saw_terminal:
                state.insert(0, self.memory[final_index].state)
            else:
                state.insert(0, self.memory[final_index - i].state)
            if self.memory[final_index - i].terminal:
                saw_terminal = True

        return np.array(state)

    def size(self) -> int:
        return len(self.memory)
