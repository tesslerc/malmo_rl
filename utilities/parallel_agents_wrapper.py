import argparse
import queue
import random
import threading
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
        for port in self.ports:
            self.agents.append(agent_class(self.params, port, not malmo_exists))

        self.agent_running = [True] * self.params.number_of_agents

    def perform_actions(self, actions: List[str]):
        Queue = queue.Queue()

        rewards = [None] * self.params.number_of_agents
        terminations = [None] * self.params.number_of_agents
        states = [None] * self.params.number_of_agents
        terminations_due_to_timeout = [None] * self.params.number_of_agents

        if actions[0] == 'new game':
            self.agent_running = [True] * self.params.number_of_agents

        threads = []
        for idx, agent in enumerate(self.agents):
            threads.append(threading.Thread(target=self.agent_perform_action, args=(agent, actions[idx], idx, Queue)))
            threads[idx].start()

        for thread in threads:
            idx, reward, terminal, state, terminal_due_to_timeout = Queue.get()
            thread.join()
            rewards[idx] = reward
            terminations[idx] = terminal
            states[idx] = state
            terminations_due_to_timeout[idx] = terminal_due_to_timeout

        return rewards, terminations, states, terminations_due_to_timeout

    def agent_perform_action(self, agent, action, idx, Queue):
        if self.agent_running[idx]:
            reward, terminal, state, terminal_due_to_timeout = agent.perform_action(action)
            if terminal:
                self.agent_running[idx] = False
                state = np.zeros((self.params.image_width, self.params.image_height))
        else:
            reward, terminal, state, terminal_due_to_timeout = (
                None, True, np.zeros((self.params.image_width, self.params.image_height)), True)

        Queue.put((idx, reward, terminal, state, terminal_due_to_timeout))
