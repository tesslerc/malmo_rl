import argparse
import copy
import math
from random import random, randrange
from typing import Dict, Tuple

import numpy as np
import torch
from torch import optim
from torch.autograd import Variable

from policies.models.DQN import DQN
from policies.policy import Policy as AbstractPolicy
from utilities.replay_memory import ReplayMemory


class Policy(AbstractPolicy):
    def __init__(self, params: argparse) -> None:
        super(Policy, self).__init__(params)
        self.step: int = 0
        self.best_score: float = None

        self.cuda: bool = torch.cuda.is_available()
        self.model = DQN(self.params.num_actions)
        if self.cuda:
            self.model.cuda()

        self.target_model: DQN = copy.deepcopy(self.model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.params.lr)
        self.criterion = torch.nn.MSELoss()
        self.replay_memory = ReplayMemory(self.params)

        self.max_reward: int = None

        self.previous_action: int = None
        self.previous_state: np.ndarray = None

        self.current_state = np.zeros((self.params.state_size, self.params.image_width, self.params.image_height),
                                      dtype=np.float32)

    def update_observation(self, reward: float, terminal: bool, terminal_due_to_timeout: bool, is_train: bool) -> None:
        # Normalizing reward forces all rewards to the range of [0, 1]. This tends to help convergence.
        self.max_reward = max(self.max_reward, abs(reward)) if self.max_reward is not None else abs(reward)
        if self.params.normalize_reward:
            reward = reward * 1.0 / self.max_reward
        if self.previous_action is not None and is_train:
            self.replay_memory.add_observation(self.previous_state, self.previous_action, reward,
                                               1 if terminal else 0, terminal_due_to_timeout)
        if terminal:
            self.current_state = np.zeros((self.params.state_size, self.params.image_width, self.params.image_height),
                                          dtype=np.float32)

    def get_action(self, state: np.ndarray, is_train: bool) -> Tuple[int, Dict[str, float]]:
        log_dict = {}
        if self.step > self.params.learn_start and is_train:
            log_dict = self.train()

        # Normalize pixel values to [0,1] range.
        state = np.array(state).reshape(self.params.image_width, self.params.image_height) / 255.0

        self.current_state[:(self.params.state_size - 1)] = self.current_state[1:]
        self.current_state[-1] = state

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

        if epsilon > random():
            # Random Action
            action = torch.IntTensor([randrange(self.params.num_actions)])
        else:
            state = torch.from_numpy(self.current_state).unsqueeze(0)  # Unsqueeze adds the batch dimension.
            if self.cuda:
                state = state.cuda()
            action = self.model(Variable(state, volatile=True)).data.max(1)[1].cpu()

        if is_train:
            self.previous_action, self.previous_state = action, state
        return action[0], log_dict

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
        batch_state, batch_action, batch_reward, batch_terminal, batch_next_state = self.replay_memory.sample()
        batch_state = Variable(torch.from_numpy(np.array(batch_state)).type(torch.FloatTensor))
        batch_action = Variable(torch.stack(batch_action)).type(torch.LongTensor)
        batch_reward = Variable(torch.from_numpy(np.array(batch_reward)).type(torch.FloatTensor))
        batch_next_state = Variable(torch.from_numpy(np.array(batch_next_state)).type(torch.FloatTensor))
        not_done_mask = Variable(torch.from_numpy(1 - np.array(batch_terminal)).type(torch.FloatTensor))
        if self.cuda:
            batch_state = batch_state.cuda()
            batch_action = batch_action.cuda()
            batch_reward = batch_reward.cuda()
            batch_next_state = batch_next_state.cuda()
            not_done_mask = not_done_mask.cuda()

        # Zero out the gradient buffer.
        self.optimizer.zero_grad()

        # Calculate expected Q values.
        current_Q = self.target_model(batch_state).gather(1, batch_action)

        # Calculate 1 step Q expectation: Q'(s) = r + gamma * max_a {Q(s+1)}.
        # Loss is: 0.5*(current_Q_values - target_Q_values)^2.
        # TD error is: current_Q_values - target_Q_values.
        if self.params.double_dqn:
            next_max_Q = self.model(batch_next_state).detach().max(1)[0]
        else:
            next_max_Q = self.target_model(batch_next_state).detach().max(1)[0]
        next_Q = not_done_mask * next_max_Q
        target_Q = batch_reward + (self.params.gamma * next_Q)

        # Calculate the loss and propagate the gradients.
        loss = self.criterion(current_Q, target_Q)
        loss.backward()

        if self.params.gradient_clipping > 0:
            torch.nn.utils.clip_grad_norm(self.target_model.parameters(), self.params.gradient_clipping)
        self.optimizer.step()

        return {'loss': loss.data[0], 'td_error': (current_Q - target_Q).mean().data[0]}
