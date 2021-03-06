import argparse
import random
import threading
from queue import Queue
from typing import List

import numpy as np


class ParallelAgentsWrapper(object):
    def __init__(self, agent_class, params: argparse) -> None:
        self.params = params
        if self.params.malmo_ports is not None:
            self.ports = self.params.malmo_ports
            malmo_exists = True
        else:
            self.ports = random.sample(range(10001, 10999), self.params.number_of_agents)
            malmo_exists = False

        self.agents = []
        for agent_index, port in enumerate(self.ports):
            self.agents.append(agent_class(self.params, port, not malmo_exists, agent_index))

        self.agent_running = [True for _ in range(self.params.number_of_agents)]

        if self.params.retain_rgb:
            self.previous_state = [np.zeros(
                (3, self.params.image_width, self.params.image_height)) for _ in range(self.params.number_of_agents)]
        else:
            self.previous_state = [np.zeros(
                (self.params.image_width, self.params.image_height)) for _ in range(self.params.number_of_agents)]

    def perform_actions(self, actions: List[str], is_train: bool):
        rewards = [None for _ in range(self.params.number_of_agents)]
        terminations = [None for _ in range(self.params.number_of_agents)]
        states = [None for _ in range(self.params.number_of_agents)]
        terminations_due_to_timeout = [None for _ in range(self.params.number_of_agents)]
        successful_agents = [None for _ in range(self.params.number_of_agents)]

        if actions[0] == 'new game':
            self.agent_running = [True for _ in range(self.params.number_of_agents)]

        results_queue = Queue()
        threads = []
        for idx, agent in enumerate(self.agents):
            threads.append(threading.Thread(target=self.agent_perform_action,
                                            args=(agent, actions[idx], idx, results_queue, is_train)))
            threads[-1].start()

        for thread in threads:
            thread.join()

            result = results_queue.get()
            idx, reward, terminal, state, terminal_due_to_timeout, success = result
            rewards[idx] = reward
            terminations[idx] = terminal
            # Cosmetics, used to keep the termination screen valid while other agents are not done yet.
            if terminal or terminal is None or state.size == 0:
                state = self.previous_state[idx]
            else:
                self.previous_state[idx] = state
            states[idx] = state
            terminations_due_to_timeout[idx] = terminal_due_to_timeout
            successful_agents[idx] = success

        return rewards, terminations, states, terminations_due_to_timeout, successful_agents

    def agent_perform_action(self, agent, action, idx, results_queue, is_train):
        if self.agent_running[idx]:
            reward, terminal, state, terminal_due_to_timeout, success = agent.perform_action(action, is_train)
            if terminal or terminal_due_to_timeout:
                self.agent_running[idx] = False
        else:
            reward, terminal, state, terminal_due_to_timeout, success = (
                None, None, None, None, False)

        results_queue.put(tuple((idx, reward, terminal, state, terminal_due_to_timeout, success)))
