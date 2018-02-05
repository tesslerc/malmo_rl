import argparse
import math
import os
from random import random
from typing import Dict, List

import numpy as np
import torch
from torch import optim
from torch.autograd import Variable

from policies.models.dqn import DQN
from policies.policy import Policy as AbstractPolicy
from utilities import helpers
from utilities.replay_memory import ParallelReplayMemory
from utilities.adamw_optimizer import AdamW


def exp_lr_scheduler(optimizer, epoch, lr_decay=0.1, lr_decay_epoch=7):
    """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""
    if epoch % lr_decay_epoch:
        return optimizer

    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    return optimizer


class Policy(AbstractPolicy):
    def __init__(self, params: argparse) -> None:
        super(Policy, self).__init__(params)
        self.step: int = 0

        self.action_mapping: List[str] = self.params.available_actions

        self.cuda: bool = torch.cuda.is_available() and not self.params.no_cuda
        self.model: torch.nn.Module = self.create_model()
        self.target_model: torch.nn.Module = self.create_model()
        self.target_model.apply(helpers.weights_init)
        self.dtype = torch.FloatTensor
        self.dtype_int = torch.LongTensor

        if self.cuda:
            self.model = self.model.cuda()
            self.target_model = self.target_model.cuda()
            self.dtype = torch.cuda.FloatTensor
            self.dtype_int = torch.cuda.LongTensor

        self.update_target_network()

        self.optimizer = optim.RMSprop(self.target_model.parameters(), lr=self.params.lr, eps=0.01, alpha=0.95)
        self.criterion = torch.nn.SmoothL1Loss()
        self.replay_memory = ParallelReplayMemory(self.params)

        self.min_reward: int = None
        self.max_reward: int = None

        self.previous_actions: List[int] = None
        self.previous_states: np.ndarray = None

        self.current_state: np.ndarray = np.zeros(
            (self.params.number_of_agents, self.params.state_size * (3 if self.params.retain_rgb else 1),
             self.params.image_width, self.params.image_height), dtype=np.float32)

        if params.resume:
            self.load_state()

    def create_model(self) -> torch.nn.Module:
        return DQN(len(self.action_mapping), self.params.state_size * (3 if self.params.retain_rgb else 1))

    def update_observation(self, rewards: List[float], terminations: List[bool],
                           terminations_due_to_timeout: List[bool], success: List[bool],
                           is_train: bool) -> None:
        # Normalizing reward forces all Q values to the range of [-1, 1]. This tends to help convergence.
        for idx, reward in enumerate(rewards):
            if not terminations_due_to_timeout[idx] and reward is not None and reward != 0:
                self.max_reward = max(self.max_reward, reward) if self.max_reward is not None else reward
                self.min_reward = min(self.min_reward, reward) if self.min_reward is not None else reward

            if self.params.normalize_reward and reward is not None and \
                    self.max_reward is not None and self.min_reward is not None:
                    # Q values are limited to the range of [-1, 1]
                    rewards[idx] = reward * (1.0 - self.params.gamma) / max(abs(self.min_reward), abs(self.max_reward))

        if self.previous_actions is not None and is_train:
            self.replay_memory.add_observation(self.previous_states, self.previous_actions, rewards,
                                               terminations, terminations_due_to_timeout, success)

        for idx, terminal in enumerate(terminations):
            if terminal or terminal is None:
                self.current_state[idx] = np.zeros(
                    (self.params.state_size * (3 if self.params.retain_rgb else 1), self.params.image_width,
                     self.params.image_height), dtype=np.float32)

    def get_action(self, states: List[np.ndarray], is_train: bool) -> List[str]:
        if self.params.viz is not None:
            # Send screen of each agent to visdom.
            images = np.zeros((self.params.number_of_agents, 3, 84, 84))
            for idx in range(self.params.number_of_agents):
                if self.params.retain_rgb:
                    images[idx, :, :, :] = states[idx]
                else:
                    images[idx, 1, :, :] = states[idx]
                self.params.viz.image(images[idx], win='state_agent_' + str(idx),
                                      opts=dict(title='Agent ' + str(idx) + '\'s state'))

        states = np.array(states)
        if not self.params.retain_rgb:
            states = np.expand_dims(states, axis=1)

        self.current_state[:, :(self.params.state_size - 1) * (3 if self.params.retain_rgb else 1)] = \
            self.current_state[:, 1 * (3 if self.params.retain_rgb else 1):]
        self.current_state[:, -1 * (3 if self.params.retain_rgb else 1):] = states

        if is_train:
            # Decrease epsilon value
            self.step += 1
            if self.step > self.params.learn_start:
                epsilon = self.params.epsilon_end + (self.params.epsilon_start - self.params.epsilon_end) * math.exp(
                    -1. * (self.step - self.params.learn_start) / self.params.epsilon_decay)
            else:
                epsilon = 1

            self.update_target_network()
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
        torch_state = torch.from_numpy(self.current_state).type(self.dtype)

        q_values = self.model(Variable(torch_state, volatile=True)).data.cpu()

        if epsilon > random():
            # Random Action
            actions = torch.from_numpy(np.random.randint(0, len(self.action_mapping), self.params.number_of_agents))
        else:
            actions = q_values.max(1)[1]

        if self.params.viz is not None:
            # Send Q distribution of each agent to visdom.
            for idx in range(self.params.number_of_agents):
                values = np.eye(len(self.action_mapping))
                self.params.viz.bar(X=values, win='plot_agent_' + str(idx),
                                    Y=q_values[idx].numpy(),
                                    opts=dict(
                                        title='Agent ' + str(idx) + '\'s expected Q values',
                                        xlabel='Value',
                                        stacked=False,
                                        legend=self.action_mapping
                                    ))

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
                self.model.load_state_dict(self.target_model.state_dict())

    def train(self, next_state) -> Dict[str, float]:
        del next_state  # not required.
        if self.replay_memory.size() < self.params.batch_size or \
                self.replay_memory.size() < self.params.learn_start or \
                self.step % self.params.learn_frequency != 0 or \
                self.step < self.params.learn_start:
            return {}

        batch_state, batch_action, batch_reward, batch_terminal, batch_next_state, indices = self.replay_memory.sample()
        batch_state = np.reshape(np.array(batch_state), (self.params.batch_size,
                                                         self.params.state_size * (3 if self.params.retain_rgb else 1),
                                                         self.params.image_width, self.params.image_height))

        batch_state = Variable(torch.from_numpy(batch_state)).type(self.dtype)
        # batch_action = List[a_1, a_2, ..., a_batch_size].
        # As a tensor it has a single dimension length of batch_size. Performing unsqueeze(-1) will add a dimension at
        # the end, making the dimensions 32x1 -> [[a_1], [a_2], ..., [a_batch_size]].
        batch_action = Variable(torch.from_numpy(np.array(batch_action)).unsqueeze(-1)).type(self.dtype_int)
        batch_reward = Variable(torch.from_numpy(np.array(batch_reward))).type(self.dtype)
        batch_next_state = np.reshape(np.array(batch_next_state),
                                      (self.params.batch_size,
                                       self.params.state_size * (3 if self.params.retain_rgb else 1),
                                       self.params.image_width, self.params.image_height))
        batch_next_state = Variable(torch.from_numpy(batch_next_state)).type(self.dtype)
        # not_done_mask contains 0 for terminal states and 1 for non-terminal states.
        not_done_mask = Variable(torch.from_numpy(1 - np.array(batch_terminal))).type(self.dtype)

        # Zero out the gradient buffer.
        self.optimizer.zero_grad()

        loss, td_error = self.get_loss(batch_state, batch_action, batch_reward, not_done_mask, batch_next_state)
        # Update priorities in ER.
        if self.params.prioritized_experience_replay:
            self.replay_memory.update_priorities(indices, np.abs(td_error))
        loss.backward()

        if self.params.gradient_clipping > 0:
            for param in self.target_model.parameters():
                param.grad.data.clamp_(-self.params.gradient_clipping, self.params.gradient_clipping)

        self.optimizer.step()
        self.optimizer = exp_lr_scheduler(self.optimizer, epoch=self.step, lr_decay=0.99999, lr_decay_epoch=1)

        return {'loss': loss.data[0], 'td_error': td_error.mean()}

    def get_loss(self, batch_state, batch_action, batch_reward, not_done_mask, batch_next_state):
        # Calculate expected Q values.
        current_q = self.target_model(batch_state).gather(1, batch_action)

        # Calculate 1 step Q expectation: Q'(s) = r + gamma * max_a {Q(s+1)}.
        # Loss is: 0.5*(current_q_values - target_q_values)^2.
        # TD error is: current_q_values - target_q_values.
        if self.params.double_dqn:
            next_best_actions = self.model(batch_next_state).max(1)[1].unsqueeze(-1)
            next_max_q = self.target_model(batch_next_state).gather(1, next_best_actions).squeeze(-1)
        else:
            next_max_q = self.target_model(batch_next_state).max(1)[0]
        next_q = next_max_q.mul(not_done_mask)

        target_q = batch_reward + (self.params.gamma * next_q)
        td_error = (current_q - target_q).data.cpu().numpy()[0]

        # Calculate the loss and propagate the gradients.
        loss = self.criterion(current_q, target_q.detach())
        return loss, td_error

    def save_state(self) -> None:
        checkpoint = {
            'model': self.model.state_dict(),
            'target_model': self.target_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': self.step,
            'min_reward': self.min_reward,
            'max_reward': self.max_reward,

        }
        torch.save(checkpoint, 'saves/' + self.params.save_name + '.pth.tar')

    def load_state(self) -> None:
        if os.path.isfile('saves/' + self.params.save_name + '.pth.tar'):
            checkpoint = torch.load('saves/' + self.params.save_name + '.pth.tar')

            self.model.load_state_dict(checkpoint['model'])
            self.target_model.load_state_dict(checkpoint['target_model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.step = checkpoint['step']
            self.min_reward = checkpoint['min_reward']
            self.max_reward = checkpoint['max_reward']
        else:
            raise FileNotFoundError('Unable to resume saves/' + self.params.save_name + '.pth.tar')
