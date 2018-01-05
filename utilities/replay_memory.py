import argparse
import copy
from collections import namedtuple
from typing import List

import numpy as np

from utilities.segment_tree import SumSegmentTree

# Definition of observation:
#   state:    s_t
#   action:   a_t
#   reward:   r_(t+1) (reward received due to being at state s_t and performing action a_t which transitions to state
#                       s_(t+1))
#   terminal: t_(t+1) (whether or not the next state is a terminal state)
slim_observation = namedtuple('slim_observation', 'state, action, reward, terminal, terminal_due_to_timeout, success')
observation = namedtuple('observation', 'state, action, reward, terminal, next_state, index_in_memory')


class ReplayMemory(object):
    def __init__(self, params: argparse) -> None:
        self.params = params
        self.memory_size = params.replay_memory_size
        self.success_memory_size = int(self.memory_size * 0.1)  # 10% of the size of the main memory.
        self.batch_size = params.batch_size
        self.state_size = params.state_size
        self.memory: List[slim_observation] = [None for _ in range(self.memory_size)]
        self.elements_in_memory = 0
        self.insert_index = 0

        # Success memory will only contain trajectories which lead to a successful finish of the task.
        if self.params.success_replay_memory:
            self.success_memory: List[slim_observation] = [None for _ in range(self.success_memory_size)]
            self.maximal_success_trajectory = 10  # For trajectories longer, we will keep only the last X steps.
            self.success_sample_probability = 0.2  # 20% chance to sample from the success memory.
            self.elements_in_success_memory = 0
            self.success_insert_index = 0

        # Prioritized ER parameters.
        if self.params.prioritized_experience_replay:
            self.epsilon = 0.01
            self.alpha = 0.6
            it_capacity = 1
            while it_capacity < self.memory_size:
                it_capacity *= 2
            self._it_sum = SumSegmentTree(it_capacity)
            self._max_priority = 1.0

    def add_observation(self, state: object, action: int, reward: float, terminal: int,
                        terminal_due_to_timeout: bool, success: bool) -> None:
        self.memory[self.insert_index] = slim_observation(state=state, action=action, reward=reward, terminal=terminal,
                                                          terminal_due_to_timeout=terminal_due_to_timeout,
                                                          success=success)

        if self.params.prioritized_experience_replay:
            # Update values in Sum and Min trees (Prioritized ER). To ensure all observations are sampled at least once,
            # they are initially set to maximal priority.
            priority = (self._max_priority + self.epsilon) ** self.alpha
            if self.insert_index < self.params.state_size:
                # We want to make sure that the minimal sampled index will be state_size to ensure we can build a full
                # state.
                priority = 0.0
            self._it_sum[self.insert_index] = priority

        if success and self.params.success_replay_memory:
            # Find trajectory start index
            trajectory_length = 0
            while trajectory_length < self.maximal_success_trajectory and \
                    (self.insert_index - trajectory_length) > 0 and \
                    self.memory[self.insert_index - trajectory_length - 1].terminal != 1:
                trajectory_length += 1

            for idx in reversed(range(trajectory_length)):
                self.success_memory[self.success_insert_index] = copy.deepcopy(self.memory[self.insert_index - idx])
                self.elements_in_success_memory = min(self.elements_in_success_memory + 1, self.success_memory_size)
                self.success_insert_index = (self.success_insert_index + 1) % self.success_memory_size

        self.elements_in_memory = min(self.elements_in_memory + 1, self.memory_size)
        self.insert_index = (self.insert_index + 1) % self.memory_size

    def _sample_proportional(self, batch_size) -> List[int]:
        res = []
        for _ in range(batch_size):
            mass = np.random.random() * self._it_sum.sum(0, self.elements_in_memory - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self):
        # Returns: Tuple[states, actions, rewards, termination values, next states, indices]
        mini_batch = []

        if self.params.prioritized_experience_replay:
            training_samples = self._sample_proportional(self.batch_size)
        else:
            training_samples = np.random.randint(low=(self.params.state_size - 1), high=(self.elements_in_memory - 1),
                                                 size=self.batch_size)
        for index in range(self.batch_size):
            if not self.params.success_replay_memory or np.random.rand() > self.success_sample_probability or \
                    self.elements_in_success_memory < self.params.state_size:
                memory = self.memory

                while memory[training_samples[index]].terminal_due_to_timeout or \
                        training_samples[index] < self.params.state_size:
                    # We do not learn from termination states due to timeout. Timeout is an artificial addition to make
                    # sure episodes end and the train/test procedure continues.
                    # Also make sure all samples are in the range [self.params.state_size, self.elements_in_memory - 1]
                    # to ensure that we can always build the first state and the next state.
                    if self.params.prioritized_experience_replay:
                        training_samples[index] = self._sample_proportional(1)[0]
                    else:
                        training_samples[index] = np.random.randint(low=(self.params.state_size - 1),
                                                                    high=(self.elements_in_memory - 1), size=1)

                sample_index = training_samples[index]
            else:
                memory = self.success_memory
                training_samples[index] = -1
                sample_index = np.random.randint(low=(self.params.state_size - 1),
                                                 high=(self.elements_in_success_memory - 1))
                while memory[sample_index].terminal_due_to_timeout:
                    sample_index = np.random.randint(low=(self.params.state_size - 1),
                                                     high=(self.elements_in_success_memory - 1))

            obs = memory[sample_index]
            state = self._build_state(sample_index, memory)
            if obs.terminal != 1:  # 1 means True.
                next_state = self._build_state(sample_index + 1, memory)
            else:
                # Instead of trying to infer state size, just return a state. The terminal flag denotes to disregard
                # this 'next_state' object.
                next_state = state

            mini_batch.append(observation(state=state, action=obs.action, reward=obs.reward, terminal=obs.terminal,
                                          next_state=next_state, index_in_memory=training_samples[index]))

        return zip(*mini_batch)

    def _build_state(self, final_index: int, memory) -> np.ndarray:
        state = []
        # Final observation should be added prior to the loop, to ensure proper state buildup.
        state.insert(0, memory[final_index].state)
        saw_terminal = False
        for i in range(1, self.params.state_size):
            # Once we encounter a terminal state, this means we are wrapping around to a previous trajectory.
            # States are start-zero-padded given they are the start of the trajectory.
            if memory[final_index - i].terminal:
                saw_terminal = True

            if saw_terminal:
                state.insert(0, np.zeros_like(memory[final_index].state))
            else:
                state.insert(0, memory[final_index - i].state)

        return np.array(state)

    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        assert len(indices) == len(priorities)
        for idx, priority in zip(indices, priorities):
            if idx >= 0:  # idx = -1 means sampled from the success ER.
                assert priority >= 0
                assert 0 <= idx < self.elements_in_memory
                self._it_sum[self.insert_index] = (priority + self.epsilon) ** self.alpha

                self._max_priority = max(self._max_priority, priority)

    def size(self) -> int:
        return self.elements_in_memory


class ParallelReplayMemory(ReplayMemory):
    def __init__(self, params: argparse) -> None:
        super(ParallelReplayMemory, self).__init__(params)

        self.agents_observations: List[List[slim_observation]] = None
        self._reset_agents_observations()

    def _reset_agents_observations(self):
        self.agents_observations = [[] for _ in range(self.params.number_of_agents)]

    def add_observation(self, state, action: List[int], reward: List[float], terminal: List[bool],
                        terminal_due_to_timeout: List[bool], success: List[bool]) -> None:
        agents_still_playing = False
        for idx, r in enumerate(reward):
            if r is not None and terminal[idx] is not None:
                if not terminal[idx]:
                    agents_still_playing = True
                self.agents_observations[idx].append(
                    slim_observation(state=state[idx], action=action[idx], reward=reward[idx],
                                     terminal=int(terminal[idx]), terminal_due_to_timeout=terminal_due_to_timeout[idx],
                                     success=success[idx]))

        # Once all agents have finished playing, insert all trajectories one after the other into the replay memory.
        # This behavior keeps our observations synced properly whilst allowing for multiple instances at once.
        if not agents_still_playing:
            for agent_idx in range(self.params.number_of_agents):
                for _slim_observation in self.agents_observations[agent_idx]:
                    super(ParallelReplayMemory, self).add_observation(_slim_observation.state, _slim_observation.action,
                                                                      _slim_observation.reward,
                                                                      _slim_observation.terminal,
                                                                      _slim_observation.terminal_due_to_timeout,
                                                                      _slim_observation.success)
            self._reset_agents_observations()