# Thanks to Felix Yu for the helpful reference.
# Blog: https://flyyufelix.github.io/2017/10/24/distributional-bellman.html
# Keras code: https://github.com/flyyufelix/C51-DDQN-Keras

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
        # Success replay memory and prioritized ER cause non uniform sampling from experience, which in turn causes
        # convergence to the wrong distribution.
        params.success_replay_memory = False
        params.prioritized_experience_replay = False
        super(Policy, self).__init__(params)

        self.delta_z = (self.params.max_q_value - self.params.min_q_value) / float(self.params.number_of_atoms - 1)
        self.atom_values = Variable(torch.from_numpy(
            np.array([self.params.min_q_value + i * self.delta_z for i in range(self.params.number_of_atoms)]))).type(
            torch.FloatTensor)

        self.criterion = torch.nn.KLDivLoss()

    def create_model(self) -> torch.nn.Module:
        return DISTRIBUTIONAL_DQN(len(self.action_mapping), self.params.number_of_atoms, self.params.state_size)

    def action_epsilon_greedy(self, epsilon: float) -> torch.LongTensor:
        torch_state = torch.from_numpy(self.current_state)
        if self.cuda:
            torch_state = torch_state.cuda()
        distributions, q_values = self.model(Variable(torch_state, volatile=True).type(torch.FloatTensor),
                                             self.atom_values)

        if epsilon > random():
            # Random Action
            actions = torch.from_numpy(np.random.randint(0, len(self.action_mapping), self.params.number_of_agents))
        else:
            actions = q_values.data.max(1)[1].cpu()

        if self.params.viz is not None:
            # Send Q distribution of each agent to visdom.
            for idx in range(self.params.number_of_agents):
                self.params.viz.bar(X=distributions.data.numpy()[idx, :, :].T, win='distribution_agent_' + str(idx),
                                    Y=self.atom_values.data.numpy(),
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
        expanded_batch_reward = batch_reward.view(-1, 1).expand(self.params.batch_size, self.params.number_of_atoms)

        current_distributions, current_q = self.target_model(batch_state, self.atom_values, True)

        current_distributions = current_distributions.gather(1, expanded_batch_action).squeeze(1)
        current_q = current_q.gather(1, batch_action)

        # Update rule: Z'(s, a) = r(s, a) + gamma * Z(s', argmax_a(Q(s' ,a))
        # Loss is: cross entropy(Z(s, a), Z'(s, a))
        if self.params.double_dqn:
            _, q_values = self.model(batch_next_state, self.atom_values)
        else:
            _, q_values = self.target_model(batch_next_state, self.atom_values)

        next_best_actions = Variable(q_values.data.max(1)[1]).view(-1, 1, 1).expand(self.params.batch_size, 1,
                                                                                    self.params.number_of_atoms)
        next_distributions, _ = self.target_model(batch_next_state, self.atom_values)
        next_distributions = next_distributions.gather(1, next_best_actions).squeeze(1)

        target_distribution = zeros_like(current_distributions).detach()
        # Step A. is calculating the new indices.
        # Step B. calculate the new loss.
        # Calculate r + gamma * z(s', a)
        Tz = torch.clamp(self.atom_values * self.params.gamma + expanded_batch_reward, self.params.min_q_value,
                         self.params.max_q_value)
        bj = ((Tz - self.params.min_q_value) / self.delta_z).type(torch.FloatTensor)  # bj in [0, number_of_atoms - 1]
        m_l, m_u = bj.floor().data, bj.ceil().data
        m_l_numpy, m_u_numpy = m_l.numpy().astype(int), m_u.numpy().astype(int)

        for batch_idx in range(self.params.batch_size):
            if (not_done_mask[batch_idx].data == 1).numpy():
                for atom_idx in range(self.params.number_of_atoms):
                    if m_l_numpy[batch_idx, atom_idx] != m_u_numpy[batch_idx, atom_idx]:
                        target_distribution[batch_idx, m_l_numpy[batch_idx, atom_idx]] = \
                            target_distribution[batch_idx, m_l_numpy[batch_idx, atom_idx]] + \
                            next_distributions[batch_idx, atom_idx] * (
                                    m_u[batch_idx, atom_idx] - bj[batch_idx, atom_idx])
                        target_distribution[batch_idx, m_u_numpy[batch_idx, atom_idx]] = \
                            target_distribution[batch_idx, m_u_numpy[batch_idx, atom_idx]] + \
                            next_distributions[batch_idx, atom_idx] * (
                                    bj[batch_idx, atom_idx] - m_l[batch_idx, atom_idx])
                    else:
                        target_distribution[batch_idx, m_u_numpy[batch_idx, atom_idx]] = \
                            target_distribution[batch_idx, m_u_numpy[batch_idx, atom_idx]] + \
                            next_distributions[batch_idx, atom_idx]
            else:
                Tz_single = torch.clamp(batch_reward[batch_idx], self.params.min_q_value, self.params.max_q_value)
                bj_single = ((Tz_single - self.params.min_q_value) / self.delta_z).type(torch.FloatTensor)
                m_l_single, m_u_single = np.floor(bj_single.data.numpy())[0], np.ceil(bj_single.data.numpy())[0]

                if m_l_single != m_u_single:
                    target_distribution[batch_idx, m_l_single] = \
                        target_distribution[batch_idx, m_l_single] + \
                        float(m_u_single) - bj_single
                    target_distribution[batch_idx, m_u_single] = \
                        target_distribution[batch_idx, m_u_single] + \
                        bj_single - float(m_l_single)
                else:
                    target_distribution[batch_idx, m_u_single] = \
                        target_distribution[batch_idx, m_u_single] + 1

        td_error = (current_q - target_distribution @ self.atom_values).data.cpu().numpy()[0]

        # Calculate the loss (one sided cross entropy) and propagate the gradients.
        # loss = -torch.sum(target_distribution.mul(current_distributions))
        loss = self.criterion(current_distributions, target_distribution.detach())
        return loss, td_error
