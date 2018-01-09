# Thanks to Felix Yu and Kai Arulkumaran for the helpful references.
# Blog: https://flyyufelix.github.io/2017/10/24/distributional-bellman.html
# Code: https://github.com/Kaixhin/Rainbow

import argparse
from random import random

import numpy as np
import torch
from torch.autograd import Variable

from policies.dqn import Policy as DQN_Policy
from policies.models.distributional_dqn import DISTRIBUTIONAL_DQN


def zeros_like(x):
    assert x.__class__.__name__.find('Variable') != -1 or x.__class__.__name__.find(
        'Tensor') != -1, "Object is neither a Tensor nor a Variable"

    y = torch.zeros(x.size())
    if x.is_cuda:
        y = y.cuda()

    if x.__class__.__name__ == 'Variable':
        return torch.autograd.Variable(y, requires_grad=x.requires_grad)
    elif x.__class__.__name__.find('Tensor') != -1:
        return torch.zeros(y)


def ones_like(x):
    assert x.__class__.__name__.find('Variable') != -1 or x.__class__.__name__.find(
        'Tensor') != -1, "Object is neither a Tensor nor a Variable"

    y = torch.ones(x.size())
    if x.is_cuda:
        y = y.cuda()

    if x.__class__.__name__ == 'Variable':
        return torch.autograd.Variable(y, requires_grad=x.requires_grad)
    elif x.__class__.__name__.find('Tensor') != -1:
        return torch.ones(y)


class Policy(DQN_Policy):
    def __init__(self, params: argparse, quantile_regression=False):
        self.quantile_regression = quantile_regression
        super(Policy, self).__init__(params)

        self.delta_z = (self.params.max_q_value - self.params.min_q_value) * 1.0 / (self.params.number_of_atoms - 1)
        self.atom_values = torch.linspace(self.params.min_q_value, self.params.max_q_value, self.params.number_of_atoms)
        if self.cuda:
            self.atom_values = self.atom_values.cuda()

    def create_model(self) -> torch.nn.Module:
        return DISTRIBUTIONAL_DQN(len(self.action_mapping), self.params.number_of_atoms,
                                  self.params.state_size * (3 if self.params.retain_rgb else 1),
                                  not self.quantile_regression)

    def action_epsilon_greedy(self, epsilon: float) -> torch.LongTensor:
        torch_state = torch.from_numpy(self.current_state).float()
        if self.cuda:
            torch_state = torch_state.cuda()
        distributions = self.model(Variable(torch_state, volatile=True)).data
        q_values = torch.mul(distributions, self.atom_values.expand_as(distributions)).sum(2).cpu()

        if epsilon > random():
            # Random Action
            actions = torch.from_numpy(np.random.randint(0, len(self.action_mapping), self.params.number_of_agents))
        else:
            actions = q_values.max(1)[1]

        if self.params.viz is not None:
            # Send Q distribution of each agent to visdom.
            for idx in range(self.params.number_of_agents):
                self.params.viz.bar(X=distributions.cpu().numpy()[idx, :, :].T, win='distribution_agent_' + str(idx),
                                    Y=self.atom_values.cpu().numpy(),
                                    opts=dict(
                                        title='Agent ' + str(idx) + '\'s distribution',
                                        stacked=False,
                                        legend=self.action_mapping
                                    ))

        return actions

    def get_loss(self, batch_state, batch_action, batch_reward, not_done_mask, batch_next_state):
        # Calculate expected Q values.
        not_done_mask = not_done_mask.data.unsqueeze(1)

        batch_action = batch_action.view(self.params.batch_size, 1, 1)
        action_mask = batch_action.expand(self.params.batch_size, 1, self.params.number_of_atoms)

        current_distributions = self.target_model(batch_state)
        current_distributions_gathered = current_distributions.gather(1, action_mask).squeeze()

        current_q = torch.mul(current_distributions_gathered.data, self.atom_values).sum(1)

        # Update rule: Z'(s, a) = r(s, a) + gamma * Z(s', argmax_a(Q(s' ,a))
        # Loss is: cross entropy(Z(s, a), Z'(s, a))
        if self.params.double_dqn:
            next_distributions = self.model(batch_next_state).data
            q_values = (self.atom_values.expand_as(next_distributions) * next_distributions).sum(2)
        else:
            next_distributions = self.target_model(batch_next_state).data
            q_values = (self.atom_values.expand_as(next_distributions) * next_distributions).sum(2)

        next_best_actions = q_values.max(1)[1]
        next_best_actions = next_best_actions.view(self.params.batch_size, 1, 1)
        next_best_actions_mask = next_best_actions.expand(self.params.batch_size, 1, self.params.number_of_atoms)

        next_distributions = self.target_model(batch_next_state).data
        next_distributions = next_distributions.gather(1, next_best_actions_mask).squeeze()
        next_distributions *= not_done_mask

        # Compute Tz (Bellman operator T applied to z)
        # Tz = R + γ*z (accounting for terminal states)
        Tz = batch_reward.data.unsqueeze(1) + not_done_mask * self.params.gamma * self.atom_values.unsqueeze(0)
        # Clamp between supported values.
        # Supported values decreased by epsilon value to avoid scenario where discrete value falls directly on the
        # support.
        Tz = Tz.clamp(min=self.params.min_q_value, max=self.params.max_q_value)
        # Compute L2 projection of Tz onto fixed support z
        b = (Tz - self.params.min_q_value) / self.delta_z  # b = (Tz - Vmin) / Δz
        l, u = b.floor().long(), b.ceil().long()

        # Deltas denotes when u != l. This catches the times where the distribution falls directly on the support.
        # In this case (u - b) + (b - l) = 0
        deltas = (1 - (u - l)).float()
        deltas = deltas.mul(torch.pow(deltas.sum(1).unsqueeze(1).clamp(min=1), -1))

        # Distribute probability of Tz
        m = batch_state.data.new(self.params.batch_size, self.params.number_of_atoms).zero_()
        offset = torch.linspace(0, ((self.params.batch_size - 1) * self.params.number_of_atoms),
                                self.params.batch_size).long().unsqueeze(1).expand(self.params.batch_size,
                                                                                   self.params.number_of_atoms)
        if self.cuda:
            offset = offset.cuda()

        m.view(-1).index_add_(0, (l + offset).view(-1),
                              (next_distributions * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
        m.view(-1).index_add_(0, (u + offset).view(-1),
                              (next_distributions * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)
        m.view(-1).index_add_(0, (l + offset).view(-1), deltas.float().view(-1))  # When m_u == m_l, add delta.

        target_q = (self.atom_values.expand_as(m) * m).sum(1)
        td_error = (current_q - target_q).cpu().numpy()
        loss = -torch.sum(Variable(m) * current_distributions_gathered.log())

        return loss, td_error
