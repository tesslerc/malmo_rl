import argparse
from typing import List

from utilities.replay_memory import ReplayMemory, slim_observation


class ParallelReplayMemory(ReplayMemory):
    def __init__(self, params: argparse) -> None:
        super(ParallelReplayMemory, self).__init__(params)

        self.agents_observations: List[List[slim_observation]] = [[]] * self.params.number_of_agents

    def _reset_agents_observations(self):
        self.agents_observations = [[]] * self.params.number_of_agents

    def add_observation(self, state, action: List[int], reward: List[float], terminal: List[int],
                        terminal_due_to_timeout: List[bool]) -> None:
        agents_still_playing = False
        for idx, r in enumerate(reward):
            if r is not None:
                if terminal[idx] == 0:
                    agents_still_playing = True
                self.agents_observations[idx].append(
                    slim_observation(state=state[idx], action=action[idx], reward=reward[idx], terminal=terminal[idx],
                                     terminal_due_to_timeout=terminal_due_to_timeout[idx]))

        # Once all agents have finished playing, insert all trajectories one after the other into the replay memory.
        # This behavior keeps our observations synced properly whilst allowing for multiple instances at once.
        if not agents_still_playing:
            for observations in self.agents_observations:
                for observation in observations:
                    super(ParallelReplayMemory, self).add_observation(observation.state, observation.action,
                                                                      observation.reward, observation.terminal,
                                                                      observation.terminal_due_to_timeout)
            self._reset_agents_observations()
