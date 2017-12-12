import argparse
import copy
import math
from random import random
from typing import Dict, List

import numpy as np
import torch
from torch import optim
from torch.autograd import Variable

from policies.models.dqn import DQN
from policies.policy import Policy as AbstractPolicy
from utilities.parallel_replay_memory import ParallelReplayMemory


# TODO: A2C.


class Policy(AbstractPolicy):
    def __init__(self, params: argparse) -> None:
        super(Policy, self).__init__(params)
        self.step: int = 0
        self.best_score: float = None

        self.action_mapping: List[str] = self.params.available_actions

        self.cuda: bool = torch.cuda.is_available()
        self.model: torch.nn.Module = self.create_model()
        if self.cuda:
            self.model.cuda()

        self.target_model: torch.nn.Module = copy.deepcopy(self.model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.params.lr)
        self.criterion = torch.nn.MSELoss()
        self.replay_memory = ParallelReplayMemory(self.params)

        self.max_reward: int = None

        self.previous_actions: List[int] = None
        self.previous_states: np.ndarray = None

        self.current_state = np.zeros(
            (self.params.number_of_agents, self.params.state_size, self.params.image_width, self.params.image_height),
            dtype=np.float32)

    def create_model(self) -> torch.nn.Module:
        return DQN(len(self.action_mapping))

    def update_observation(self, rewards: List[float], terminations: List[bool],
                           terminations_due_to_timeout: List[bool],
                           is_train: bool) -> None:
        # Normalizing reward forces all rewards to the range of [0, 1]. This tends to help convergence.
        for idx, reward in enumerate(rewards):
            if not terminations_due_to_timeout[idx]:
                # Max reward is set to Rmax / (1 - gamma). This way Q(s, a) will be squashed to [-1, 1].
                self.max_reward = max(self.max_reward,
                                      abs(reward) / (1.0 - self.params.gamma)) if self.max_reward is not None \
                    else abs(reward) / (1.0 - self.params.gamma)
            if self.params.normalize_reward and rewards[idx] is not None:
                rewards[idx] = rewards[idx] * 1.0 / self.max_reward

        if self.previous_actions is not None and is_train:
            self.replay_memory.add_observation(self.previous_states, self.previous_actions, rewards,
                                               [int(terminal) for terminal in terminations],
                                               terminations_due_to_timeout)

        for idx, terminal in enumerate(terminations):
            if terminal:
                self.current_state[idx] = np.zeros(
                    (self.params.state_size, self.params.image_width, self.params.image_height), dtype=np.float32)

    def get_action(self, states: List[np.ndarray], is_train: bool) -> List[str]:
        # Normalize pixel values to [0,1] range.
        states = np.array(states).reshape(self.params.number_of_agents, self.params.image_width,
                                          self.params.image_height) / 255.0

        self.current_state[:, :(self.params.state_size - 1)] = self.current_state[:, 1:]
        self.current_state[:, -1] = states

        if is_train:
            # Decrease epsilon value
            self.step += 1
            if self.step > self.params.learn_start:
                epsilon = self.params.epsilon_end + (self.params.epsilon_start - self.params.epsilon_end) * math.exp(
                    -1. * (self.step - self.params.learn_start) / self.params.epsilon_decay)
            else:
                epsilon = 1
        else:
            epsilon = self.params.epsilon_test

        actions = self.action_epsilon_greedy(epsilon)

        if is_train:
            self.previous_actions, self.previous_states = actions.numpy().tolist(), states

        string_actions = []
        for action in actions:
            string_actions.append(self.action_mapping[action])
        return string_actions

    def action_epsilon_greedy(self, epsilon: float) -> torch.LongTensor:
        if epsilon > random():
            # Random Action
            actions = torch.from_numpy(np.random.randint(0, len(self.action_mapping), self.params.number_of_agents))
        else:
            torch_state = torch.from_numpy(self.current_state)
            if self.cuda:
                torch_state = torch_state.cuda()
            actions = self.model(Variable(torch_state, volatile=True)).data.max(1)[1].cpu()
        return actions

    def update_target_network(self) -> None:
        if self.step % self.params.target_update_interval == 0 or self.params.actively_follow_target:
            if self.params.actively_follow_target:
                params_target = self.target_model.named_parameters()
                params_model = self.model.named_parameters()
                dict_params_model = dict(params_model)

                for name, param in params_target:
                    # When actively following, we update on each step but a small increment towards the real model. This
                    # should allow for 'smoother' behavior (slowly changing policy without drastic changes each
                    # 'target_update_interval'.
                    dict_params_model[name].data.copy_(
                        self.params.target_update_alpha * param.data + (1 - self.params.target_update_alpha) *
                        dict_params_model[name].data)
            else:
                self.model = copy.deepcopy(self.target_model)

    def train(self) -> Dict[str, float]:
        batch_state, batch_action, batch_reward, batch_terminal, batch_next_state, indices = self.replay_memory.sample()
        batch_state = Variable(torch.from_numpy(np.array(batch_state)).type(torch.FloatTensor))
        # batch_action = List[a_1, a_2, ..., a_batch_size].
        # As a tensor it has a single dimension length of batch_size. Performing unsqueeze(-1) will add a dimension at
        # the end, making the dimensions 32x1 -> [[a_1], [a_2], ..., [a_batch_size]].
        batch_action = Variable(torch.from_numpy(np.array(batch_action)).unsqueeze(-1)).type(torch.LongTensor)
        batch_reward = Variable(torch.from_numpy(np.array(batch_reward)).type(torch.FloatTensor))
        batch_next_state = Variable(torch.from_numpy(np.array(batch_next_state)).type(torch.FloatTensor))
        # not_done_mask contains 0 for terminal states and 1 for non-terminal states.
        not_done_mask = Variable(torch.from_numpy(1 - np.array(batch_terminal)).type(torch.FloatTensor))
        if self.cuda:
            batch_state = batch_state.cuda()
            batch_action = batch_action.cuda()
            batch_reward = batch_reward.cuda()
            batch_next_state = batch_next_state.cuda()
            not_done_mask = not_done_mask.cuda()

        # Zero out the gradient buffer.
        self.optimizer.zero_grad()

        loss, td_error = self.get_loss(batch_state, batch_action, batch_reward, not_done_mask, batch_next_state)

        # Update priorities in ER.
        if self.params.prioritized_experience_replay:
            self.replay_memory.update_priorities(indices, np.abs(td_error))
        loss.backward()

        if self.params.gradient_clipping > 0:
            torch.nn.utils.clip_grad_norm(self.target_model.parameters(), self.params.gradient_clipping)
        self.optimizer.step()

        return {'loss': loss.data[0], 'td_error': td_error.mean()}

    def get_loss(self, batch_state, batch_action, batch_reward, not_done_mask, batch_next_state):
        # Calculate expected Q values.
        current_q = self.target_model(batch_state).gather(1, batch_action)

        # Calculate 1 step Q expectation: Q'(s) = r + gamma * max_a {Q(s+1)}.
        # Loss is: 0.5*(current_q_values - target_q_values)^2.
        # TD error is: current_q_values - target_q_values.
        if self.params.double_dqn:
            next_best_actions = self.model(batch_next_state).detach().max(1)[1].unsqueeze(-1)
            next_max_q = self.target_model(batch_next_state).detach().gather(1, next_best_actions).squeeze(-1)
        else:
            next_max_q = self.target_model(batch_next_state).detach().max(1)[0]
        next_q = next_max_q.mul(not_done_mask)
        target_q = batch_reward + (self.params.gamma * next_q)

        td_error = (current_q - target_q).data.cpu().numpy()[0]

        # Calculate the loss and propagate the gradients.
        loss = self.criterion(current_q, target_q)
        return loss, td_error
