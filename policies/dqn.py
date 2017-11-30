import copy
import math
from collections import deque
import numpy as np
from random import random, randrange
import torch
from torch import optim
from torch.autograd import Variable

from policies.policy import Policy as AbstractPolicy
from policies.models.DQN import DQN
from utilities.replay_memory import ReplayMemory


class Policy(AbstractPolicy):
    def __init__(self, params):
        super(Policy, self).__init__(params)
        self.step = 0
        self.best_score = None

        self.cuda = torch.cuda.is_available()
        self.model = DQN(self.params.num_actions)
        if self.cuda:
            self.model.cuda()

        self.target_model: DQN = copy.deepcopy(self.model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.params.lr)
        self.criterion = torch.nn.MSELoss()
        self.replay_memory = ReplayMemory(self.params)

        self.previous_action = None
        self.previous_observation = None
        self.reward = None
        self.terminal = None

        self.current_state = np.zeros((self.params.state_size, self.params.image_width, self.params.image_height),
                                      dtype=np.float32)

    def get_action(self, reward, terminal, observation, is_train):
        if self.step > self.params.learn_start and is_train:
            self.train()

        if self.previous_action is not None and is_train:
            self.replay_memory.add_observation(self.previous_observation, self.previous_action, reward,
                                               1 if terminal else 0)

        if self.terminal:
            self.current_state = np.zeros((self.params.state_size, self.params.image_width, self.params.image_height),
                                          dtype=np.float32)

        # Normalize pixel values to [0,1] range.
        observation = np.array(observation).reshape(self.params.image_width, self.params.image_height) / 255.0

        self.current_state[:(self.params.state_size - 1)] = self.current_state[1:]
        self.current_state[-1] = observation

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
            state = torch.from_numpy(self.current_state).unsqueeze(0)
            if self.cuda:
                state = state.cuda()
            action = self.model(Variable(state, volatile=True)).data.max(1)[1].cpu()

        if is_train:
            self.previous_action, self.previous_observation = action, observation
        return action[0]

    def update_target_network(self):
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

    def train(self):
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
        current_Q_values = self.target_model(batch_state).gather(1, batch_action)

        # Calculate 1 step Q expectation: Q'(s) = r + max_a {Q(s+1)}.
        # TD error is current_Q_values - target_Q_values.
        next_max_Q = self.target_model(batch_next_state).detach().max(1)[0]
        next_Q_values = not_done_mask * next_max_Q

        target_Q_values = batch_reward + (self.params.gamma * next_Q_values)

        # Calculate the loss and propagate the gradients.
        loss = self.criterion(current_Q_values, target_Q_values)
        loss.backward()
        if self.params.gradient_clipping > 0:
            torch.nn.utils.clip_grad_norm(self.target_model.parameters(), self.params.gradient_clipping)
        self.optimizer.step()

# TODO: Double Q Learning.
