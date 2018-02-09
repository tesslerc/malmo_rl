import argparse
import os
from typing import Dict, List, Any

import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
from torch.distributions import Categorical

from policies.models.actor_critic import ACTOR_CRITIC
from policies.dqn import Policy as DQN_POLICY


class Policy(DQN_POLICY):
    def __init__(self, params: argparse) -> None:
        super(Policy, self).__init__(params)

    def create_model(self) -> torch.nn.Module:
        return ACTOR_CRITIC(len(self.action_mapping), self.params.state_size * (3 if self.params.retain_rgb else 1))

    def get_action(self, states: List[np.ndarray], is_train: bool) -> List[str]:
        self.step += 1

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

        actions = self.sample_action()

        if is_train:
            self.previous_actions, self.previous_states = actions, states

        string_actions = []
        for action in actions:
            string_actions.append(self.action_mapping[action])
        return string_actions

    def sample_action(self):
        torch_state = torch.from_numpy(self.current_state).type(self.dtype)

        probs, _, _ = self.target_model((Variable(torch_state)))

        actions = probs.multinomial()

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

        return [action[0] for action in actions.data.cpu().numpy()]

    def get_loss(self, batch_state, batch_action, batch_reward, not_done_mask, batch_next_state):
        # Calculate expected Q values.
        probs, log_probs, current_v = self.target_model(batch_state)

        probs = probs.gather(1, batch_action)
        log_probs = log_probs.gather(1, batch_action)

        # Critic loss
        # Calculate 1 step V expectation: V'(s) = r + gamma * V(s+1).
        # Loss is: 0.5*(current_v_values - target_v_values)^2.
        # TD error is: current_q_values - target_q_values.
        _, _, next_v = self.target_model(batch_next_state)
        next_v = next_v.mul(not_done_mask.unsqueeze(-1))

        target_v = batch_reward.unsqueeze(-1) + (self.params.gamma * next_v)
        advantage = (target_v - current_v)

        critic_loss = self.criterion(current_v, target_v.detach())

        # Policy loss
        policy_loss = -Variable(advantage.data) * log_probs
        policy_loss = torch.mean(policy_loss)

        # Entropy loss
        entropy = log_probs * probs
        entropy = -entropy.sum(-1).mean()

        # Calculate the loss and propagate the gradients.
        loss = critic_loss * 0.5 + policy_loss - entropy * 0.001
        return loss, advantage.data.cpu().numpy()

    def save_state(self) -> None:
        checkpoint = {
            'model': self.target_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': self.step,
            'min_reward': self.min_reward,
            'max_reward': self.max_reward,

        }
        torch.save(checkpoint, 'saves/' + self.params.save_name + '.pth.tar')

    def load_state(self) -> None:
        if os.path.isfile('saves/' + self.params.save_name + '.pth.tar'):
            checkpoint = torch.load('saves/' + self.params.save_name + '.pth.tar')

            self.target_model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.step = checkpoint['step']
            self.min_reward = checkpoint['min_reward']
            self.max_reward = checkpoint['max_reward']
        else:
            raise FileNotFoundError('Unable to resume saves/' + self.params.save_name + '.pth.tar')
