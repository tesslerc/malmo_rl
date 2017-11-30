from collections import namedtuple
import numpy as np

# Definition of observation:
#   state:    s_t
#   action:   a_t
#   reward:   r_(t+1) (reward received due to being at state s_t and performing action a_t which transitions to state
#                       s_(t+1))
#   terminal: t_(t+1) (whether or not the next state is a terminal state)
slim_observation = namedtuple('slim_observation', 'state, action, reward, terminal')
observation = namedtuple('observation', 'state, action, reward, terminal, next_state')


class ReplayMemory(object):
    def __init__(self, params):
        self.params = params
        self.memory_size = params.replay_memory_size
        self.batch_size = params.batch_size
        self.state_size = params.state_size
        self.memory = []
        self.elements_in_memory = 0

    def add_observation(self, state, action, reward, terminal):
        self.memory.append(slim_observation(state=state, action=action, reward=reward, terminal=terminal))
        if self.elements_in_memory >= self.memory_size:
            self.memory.pop(0)
        else:
            self.elements_in_memory += 1

    def sample(self):
        mini_batch = []
        training_samples = np.random.choice(3, self.elements_in_memory - 1, self.batch_size)
        for index in range(self.batch_size):
            obs = self.memory[training_samples[index]]
            obs = obs._replace(state=self._build_state(training_samples[index]))
            if obs.terminal is False:
                next_state = self._build_state(training_samples[index] + 1)
            else:
                # Instead of trying to infer state size, just return a state. The terminal flag denotes to disregard
                # this 'next_state' object.
                next_state = obs.state
            mini_batch.append(observation(state=obs.state, action=obs.action, reward=obs.reward, terminal=obs.terminal,
                                          next_state=next_state))
        # Returns tuple: batch_state, batch_action, batch_reward, batch_terminal, batch_next_state
        return zip(*mini_batch)

    def _build_state(self, final_index):
        state = []
        saw_terminal = False
        state.append(self.memory[final_index].state)
        for i in range(1, self.params.state_size):
            if self.memory[final_index - i].terminal:
                saw_terminal = True
            if saw_terminal:
                state.insert(0, np.zeros(self.memory[final_index - i].state.shape))
            else:
                state.insert(0, self.memory[final_index - i].state)

        return np.array(state)

    def size(self):
        return len(self.memory)
