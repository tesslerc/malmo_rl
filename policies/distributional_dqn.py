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
    def __init__(self, params: argparse):
        super(Policy, self).__init__(params)
        self.atom_values = Variable(torch.from_numpy(
            np.linspace(-self.params.min_max_q_values, self.params.min_max_q_values, self.params.number_of_atoms)).type(
            torch.FloatTensor))
        self.delta_z = self.params.min_max_q_values * 2.0 / self.params.number_of_atoms

        self.criterion = torch.nn.BCELoss()

    def create_model(self) -> torch.nn.Module:
        return DISTRIBUTIONAL_DQN(len(self.action_mapping), self.params.number_of_atoms)

    def action_epsilon_greedy(self, epsilon: float) -> torch.LongTensor:
        torch_state = torch.from_numpy(self.current_state)
        if self.cuda:
            torch_state = torch_state.cuda()
        distributions, q_values = self.model(Variable(torch_state, volatile=True), self.atom_values)

        if epsilon > random():
            # Random Action
            actions = torch.from_numpy(np.random.randint(0, len(self.action_mapping), self.params.number_of_agents))
        else:
            actions = q_values.data.max(1)[1].cpu()

        if self.params.viz is not None:
            # Send Q distribution of each agent to visdom.
            for idx in range(self.params.number_of_agents):
                self.params.viz.bar(X=distributions.data.numpy()[idx, :, :].T, win='distribution_agent_' + str(idx),
                                    opts=dict(
                                        title='Agent ' + str(idx) + '\'s distribution',
                                        stacked=False,
                                        legend=self.action_mapping
                                    ))

        return actions

    def get_loss(self, batch_state, batch_action, batch_reward, not_done_mask, batch_next_state):
        # Calculate expected Q values.
        expanded_batch_action = batch_action.view(-1, 1, 1).expand(self.params.batch_size, 1,
                                                                   self.params.number_of_atoms)
        batch_reward = batch_reward.view(-1, 1).expand(self.params.batch_size, self.params.number_of_atoms)

        current_distributions, current_q = self.target_model(batch_state, self.atom_values)
        current_distributions = current_distributions.gather(1, expanded_batch_action).squeeze(1)
        current_q = current_q.gather(1, batch_action)

        # Update rule: Z'(s, a) = r(s, a) + gamma * Z(s', argmax_a(Q(s' ,a))
        # Loss is: cross entropy(Z(s, a), Z'(s, a))
        if self.params.double_dqn:
            _, q_values = self.model(batch_next_state, self.atom_values)
            next_best_actions = Variable(q_values.data.max(1)[1].unsqueeze(-1)).detach().view(-1, 1, 1).expand(
                self.params.batch_size, 1, self.params.number_of_atoms)
            next_distributions, _ = self.target_model(batch_next_state, self.atom_values)
            next_distributions = next_distributions.detach().gather(1, next_best_actions).squeeze(-1).squeeze(1)
        else:
            _, q_values = self.target_model(batch_next_state, self.atom_values)
            q_values = q_values.detatch()
            next_best_actions = Variable(q_values.data.max(1)[1].unsqueeze(-1)).detach().view(-1, 1, 1).expand(
                self.params.batch_size, 1, self.params.number_of_atoms)
            next_distributions, _ = self.target_model(batch_next_state, self.atom_values)
            next_distributions = next_distributions.detach().gather(1, next_best_actions).squeeze(-1).squeeze(1)

        target_distribution = zeros_like(next_distributions).detach()

        # Step A. is calculating the new indices.
        # Step B. calculate the new loss.
        # Calculate r + gamma * z(s', a)
        max_q = ones_like(self.atom_values).float() * self.params.min_max_q_values
        min_q = max_q * -1

        Tz = torch.max(torch.min(self.atom_values * self.params.gamma + batch_reward, max_q), min_q)
        b = ((Tz - min_q) / self.delta_z).type(torch.FloatTensor)  # bj in [0, number_of_atoms - 1]
        l, u = b.floor().data, b.ceil().data
        l_numpy, u_numpy = l.numpy().astype(int), u.numpy().astype(int)

        for batch_idx in range(self.params.batch_size):
            for atom_idx in range(self.params.number_of_agents):
                if (not_done_mask[batch_idx].data == 1).numpy():
                    target_distribution[batch_idx, l_numpy[batch_idx, atom_idx]] = \
                        target_distribution[batch_idx, l_numpy[batch_idx, atom_idx]] + \
                        next_distributions[batch_idx, atom_idx] * (
                                u[batch_idx, atom_idx] - b[batch_idx, atom_idx])
                    target_distribution[batch_idx, u_numpy[batch_idx, atom_idx]] = \
                        target_distribution[batch_idx, u_numpy[batch_idx, atom_idx]] + \
                        next_distributions[batch_idx, atom_idx] * (
                                b[batch_idx, atom_idx] - l[batch_idx, atom_idx])
                else:
                    target_distribution[batch_idx, l_numpy[batch_idx, atom_idx]] = \
                        target_distribution[batch_idx, l_numpy[batch_idx, atom_idx]] + \
                        u[batch_idx, atom_idx] - b[batch_idx, atom_idx]
                    target_distribution[batch_idx, u_numpy[batch_idx, atom_idx]] = \
                        target_distribution[batch_idx, u_numpy[batch_idx, atom_idx]] + \
                        b[batch_idx, atom_idx] - u[batch_idx, atom_idx]

        td_error = (current_q - target_distribution @ self.atom_values).data.cpu().numpy()[0]

        # Calculate the loss and propagate the gradients.
        loss = self.criterion(current_distributions, target_distribution)
        return loss, td_error
