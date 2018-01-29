import argparse
from random import random

import numpy as np
import torch
from torch.autograd import Variable

from policies.distributional_dqn import Policy as DISTRIBUTIONAL_POLICY


class Policy(DISTRIBUTIONAL_POLICY):
    # Implements Quantile Regression DQN as proposed by Dabney et al. (https://arxiv.org/abs/1710.10044).
    def __init__(self, params: argparse):
        super(Policy, self).__init__(params, quantile_regression=True)

        self.support_weight = 1.0 / self.params.number_of_atoms
        self.cdf = np.cumsum(np.ones(self.params.number_of_atoms, np.float) * self.support_weight, -1)
        self.mid_supports = self.cdf - self.support_weight / 2

        self.criterion = torch.nn.SmoothL1Loss(reduce=False)

    def action_epsilon_greedy(self, epsilon: float) -> torch.LongTensor:
        torch_state = torch.from_numpy(self.current_state).float()
        if self.cuda:
            torch_state = torch_state.cuda()
        quantiles = self.model(Variable(torch_state, volatile=True)).data
        q_values = (quantiles * self.support_weight).sum(2).cpu()

        if epsilon > random():
            # Random Action
            actions = torch.from_numpy(np.random.randint(0, len(self.action_mapping), self.params.number_of_agents))
        else:
            actions = q_values.max(1)[1]

        if self.params.viz is not None:
            supports = quantiles.sort()[0].cpu().numpy()
            cdf = np.tile(self.cdf, (len(self.action_mapping), 1))

            # Send Q distribution of each agent to visdom.
            for idx in range(self.params.number_of_agents):
                self.params.viz.line(X=supports[idx, :, :].T, win='cdf_agent_' + str(idx),
                                    Y=cdf.T,
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

        current_quantiles = self.target_model(batch_state)
        current_quantiles_gathered = current_quantiles.gather(1, action_mask).squeeze()

        current_q = (current_quantiles_gathered.data * self.support_weight).sum(1)

        # Update rule: Z'(s, a) = r(s, a) + gamma * Z(s', argmax_a(Q(s' ,a))
        if self.params.double_dqn:
            next_quantiles = self.model(batch_next_state).data
        else:
            next_quantiles = self.target_model(batch_next_state).data

        q_values = (next_quantiles * self.support_weight).sum(2)

        next_best_actions = q_values.max(1)[1]
        next_best_actions = next_best_actions.view(self.params.batch_size, 1, 1)
        next_best_actions_mask = next_best_actions.expand(self.params.batch_size, 1, self.params.number_of_atoms)

        next_quantiles = self.target_model(batch_next_state).data
        next_quantiles = next_quantiles.gather(1, next_best_actions_mask).squeeze()
        next_quantiles *= not_done_mask

        # Compute Tθ (Bellman operator T applied to θ)
        # Tθ = R + γ*θ (accounting for terminal states)
        Ttheta = batch_reward.data.unsqueeze(1) + self.params.gamma * next_quantiles

        sorted_target = Ttheta.sort()[0]

        # for batch_idx in range(self.params.batch_size):
        #     for atom_idx in range(self.params.number_of_atoms):
        #         u = current_quantiles_gathered[batch_idx, atom_idx] - sorted_target[batch_idx, atom_idx]
        #         if (u.data.cpu() < 0).numpy():
        #             multiplier = np.abs(self.mid_supports[0, atom_idx] - 1)
        #         else:
        #             multiplier = self.mid_supports[0, atom_idx]
        #
        #         loss += per_element_loss[batch_idx, atom_idx] * multiplier
        loss = 0
        for idx in range(self.params.number_of_atoms):
            sorted_target_idx = sorted_target[:, idx].unsqueeze(-1).expand(-1, self.params.number_of_atoms)
            per_element_loss = self.criterion(current_quantiles_gathered, Variable(sorted_target_idx).detach())
            u = current_quantiles_gathered.data.cpu().numpy() - sorted_target_idx.cpu().numpy()
            # if (u < 0)...
            u = np.minimum(0, u)
            u[u != 0] = 1.0

            cdf = np.tile(self.mid_supports, (self.params.batch_size, 1))
            multiplier = Variable(torch.from_numpy(np.abs(cdf - u)).float())

            if self.cuda:
                multiplier = multiplier.cuda()

            loss += torch.sum(torch.mul(multiplier, per_element_loss)) / self.params.number_of_atoms

        target_q = (Ttheta * self.support_weight).sum(1)
        td_error = (current_q - target_q).cpu().numpy()

        return loss, td_error
