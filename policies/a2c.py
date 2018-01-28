import argparse
import os
from typing import Dict, List, Any

import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
from torch.distributions import Categorical

from policies.models.actor_critic import ACTOR_CRITIC
from policies.policy import Policy as AbstractPolicy
from utilities import helpers


class Policy(AbstractPolicy):
    def __init__(self, params: argparse) -> None:
        super(Policy, self).__init__(params)
        self.step: int = 0

        self.action_mapping: List[str] = self.params.available_actions

        self.cuda: bool = torch.cuda.is_available() and not self.params.no_cuda
        self.model: torch.nn.Module = self.create_model()
        self.model.apply(helpers.weights_init)

        self.cx = Variable(torch.zeros(self.params.number_of_agents, self.model.lstm_size))
        self.hx = Variable(torch.zeros(self.params.number_of_agents, self.model.lstm_size))

        self.dtype = torch.FloatTensor
        if self.cuda:
            self.dtype = torch.cuda.FloatTensor
            self.model = self.model.cuda()
            self.cx = self.cx.cuda()
            self.hx = self.hx.cuda()

        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.params.lr, eps=0.01, alpha=0.95)
        self.criterion = torch.nn.SmoothL1Loss()

        self.min_reward: int = None
        self.max_reward: int = None

        self.previous_states: np.ndarray = None
        self.entropies = [[] for _ in range(self.params.number_of_agents)]
        self.log_probs = [[] for _ in range(self.params.number_of_agents)]
        self.rewards = [[] for _ in range(self.params.number_of_agents)]
        self.values = [[] for _ in range(self.params.number_of_agents)]
        self.timeouts = [[] for _ in range(self.params.number_of_agents)]
        self.terminals = [[] for _ in range(self.params.number_of_agents)]

        self.current_state: np.ndarray = np.zeros(
            (self.params.number_of_agents, self.params.state_size * (3 if self.params.retain_rgb else 1),
             self.params.image_width, self.params.image_height), dtype=np.float32)

        self.all_finished = False

        if params.resume:
            self.load_state()

    def create_model(self) -> torch.nn.Module:
        return ACTOR_CRITIC(len(self.action_mapping), self.params.state_size * (3 if self.params.retain_rgb else 1))

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

        any_running = False
        for termination in terminations:
            if termination is False:
                any_running = True

        if is_train:
            if not any_running:
                self.all_finished = True

            for idx in range(self.params.number_of_agents):
                if len(self.terminals[idx]) == 0 or not self.terminals[idx][-1]:
                    self.terminals[idx].append(terminations[idx])
                    self.timeouts[idx].append(terminations_due_to_timeout[idx])
                    self.rewards[idx].append(
                        torch.FloatTensor([rewards[idx] if rewards[idx] is not None else 0]).type(self.dtype)
                    )

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

        actions, log_probs, entropies, values = self.sample_action()

        if is_train:
            self.previous_states = states
            for idx in range(self.params.number_of_agents):
                if len(self.terminals[idx]) == 0 or not self.terminals[idx][-1]:
                    self.log_probs[idx].append(log_probs[idx])
                    self.values[idx].append(values[idx])
                    self.entropies[idx].append(entropies[idx])

        string_actions = []
        for action in actions:
            string_actions.append(self.action_mapping[action])
        return string_actions

    def sample_action(self):
        torch_state = torch.from_numpy(self.current_state).float()
        if self.cuda:
            torch_state = torch_state.cuda()
        probs, log_probs, values, (self.hx, self.cx) = self.model((Variable(torch_state), (self.hx, self.cx)))

        entropies = -(log_probs * probs).sum(1)

        m = Categorical(probs)
        actions = m.sample()
        log_probs = m.log_prob(actions)

        probs = probs.data.cpu()
        if self.params.viz is not None:
            # Send Q distribution of each agent to visdom.
            for idx in range(self.params.number_of_agents):
                self.params.viz.bar(X=np.diag(probs[idx].numpy()), win='distribution_agent_' + str(idx),
                                    Y=np.ones_like(probs[idx].numpy()),
                                    opts=dict(
                                        title='Agent ' + str(idx) + '\'s distribution',
                                        stacked=False,
                                        legend=self.action_mapping
                                    ))

        return actions.data.cpu(), log_probs, entropies, values.squeeze()

    def train(self, next_states) -> Dict[str, float]:
        if not (self.all_finished or self.step % self.params.learn_frequency == 0):
            return {}

        policy_loss = 0
        value_loss = 0
        td_error = 0

        current_state = np.copy(self.current_state)
        current_state[:, :(self.params.state_size - 1) * (3 if self.params.retain_rgb else 1)] = \
            self.current_state[:, 1 * (3 if self.params.retain_rgb else 1):]
        current_state[:, -1 * (3 if self.params.retain_rgb else 1):] = next_states

        current_state_torch = Variable(torch.from_numpy(np.array(current_state)).float())
        if self.cuda:
            current_state_torch = current_state_torch.cuda()
        _, _, value, (self.hx, self.cx) = self.model((current_state_torch, (self.hx, self.cx)))

        for agent_idx in range(self.params.number_of_agents):
            R = value[agent_idx].squeeze().data
            if self.all_finished:
                R *= 0

            self.values[agent_idx].append(Variable(R))
            gae = 0

            for i in reversed(range(len(self.rewards[agent_idx]))):
                R = self.params.gamma * R + self.rewards[agent_idx][i]

                advantage = Variable(R) - self.values[agent_idx][i]
                value_loss = value_loss + 0.5 * advantage.pow(2)

                # Generalized Advantage Estimataion
                delta_t = self.rewards[agent_idx][i] \
                          + self.params.gamma * self.values[agent_idx][i + 1].data \
                          - self.values[agent_idx][i].data

                gae = gae * self.params.gamma * self.params.tau + delta_t

                var_gae = Variable(gae)
                if self.cuda:
                    var_gae = var_gae.cuda()

                policy_loss = policy_loss - \
                              self.log_probs[agent_idx][i] * var_gae - 0.01 * self.entropies[agent_idx][i]

                td_error += delta_t

        self.optimizer.zero_grad()

        total_loss = policy_loss.sum() + 0.5 * value_loss.sum()
        total_loss.backward()

        if self.params.gradient_clipping > 0:
            for param in self.model.parameters():
                param.grad.data.clamp_(-self.params.gradient_clipping, self.params.gradient_clipping)

        self.optimizer.step()

        if self.all_finished:
            self.current_state: np.ndarray = np.zeros(
                (self.params.number_of_agents, self.params.state_size * (3 if self.params.retain_rgb else 1),
                 self.params.image_width, self.params.image_height), dtype=np.float32)
            self.cx = Variable(torch.zeros(self.params.number_of_agents, self.model.lstm_size))
            self.hx = Variable(torch.zeros(self.params.number_of_agents, self.model.lstm_size))
        else:
            self.cx = Variable(self.cx.data)
            self.hx = Variable(self.hx.data)

        if self.cuda:
            self.cx = self.cx.cuda()
            self.hx = self.hx.cuda()

        self.all_finished = False

        del self.rewards[:]
        del self.log_probs[:]
        del self.timeouts[:]
        del self.values[:]
        del self.entropies[:]
        del self.terminals[:]

        self.entropies = [[] for _ in range(self.params.number_of_agents)]
        self.log_probs = [[] for _ in range(self.params.number_of_agents)]
        self.rewards = [[] for _ in range(self.params.number_of_agents)]
        self.values = [[] for _ in range(self.params.number_of_agents)]
        self.timeouts = [[] for _ in range(self.params.number_of_agents)]
        self.terminals = [[] for _ in range(self.params.number_of_agents)]

        return {'loss': total_loss.data[0], 'td_error': td_error.mean()}

    def save_state(self) -> None:
        checkpoint = {
            'model': self.model.state_dict(),
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
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.step = checkpoint['step']
            self.min_reward = checkpoint['min_reward']
            self.max_reward = checkpoint['max_reward']
        else:
            raise FileNotFoundError('Unable to resume saves/' + self.params.save_name + '.pth.tar')
