import argparse
import logging
import sys
from typing import Tuple, Dict, List

import numpy as np
from torch import nn

from policies.policy import Policy
from utilities.parallel_agents_wrapper import ParallelAgentsWrapper


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def weights_init(m: nn.Module) -> None:
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:  # -1 means not found, i.e isn't a Conv layer.
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_os() -> str:
    if sys.platform in ['linux', 'linux2']:
        return 'linux'
    elif sys.platform == 'darwin':  # osx
        raise NotImplementedError('OSX interactive agent is not supported yet.')
    elif sys.platform in ['win32', 'cygwin']:
        return 'win'
    else:
        raise Exception('Unidentified operating system.')


def play_full_episode(agents: ParallelAgentsWrapper, policy: Policy, step: int, params: argparse, is_train: bool) \
        -> Tuple[ParallelAgentsWrapper, int, bool, bool, float, int, Dict[str, float]]:
    eval_required = False
    checkpoint_reached = False
    epoch_reward = 0
    rewards, terminals, states, terminals_due_to_timeout, success = agents.perform_actions(
        ['new game' for _ in range(params.number_of_agents)])  # Restart all the agents.

    log_dict = {}
    start_step = step
    successful_agents = [0 for _ in range(params.number_of_agents)]
    while not all([t or t is None for t in terminals]):  # Loop ends only when all agents have terminated.
        action = policy.get_action(states, is_train)
        rewards, terminals, states, terminals_due_to_timeout, success = agents.perform_actions(action)

        # reward is a list. Passing it to update_observation changes its values hence all references should be
        # performed prior to calling update_observation.
        for idx, reward in enumerate(rewards):
            if reward is not None:
                epoch_reward += reward
                if success[idx]:
                    successful_agents[idx] = 1
        logging.debug('step: %s, reward: %s, terminal: %s, terminal_due_to_timeout: %s, sucess: %s', step, rewards,
                      terminals, terminals_due_to_timeout, success)

        policy.update_observation(rewards, terminals, terminals_due_to_timeout, success, is_train)

        if is_train:
            single_log_dict = policy.train()
        else:
            single_log_dict = {}

        step += 1

        if step % params.eval_frequency == 0:
            eval_required = True
        if step % params.checkpoint_interval == 0:
            checkpoint_reached = True

        for item in single_log_dict:
            if item in log_dict:
                log_dict[item] = log_dict[item] + single_log_dict[item]
            else:
                log_dict[item] = single_log_dict[item]

    for item in log_dict:
        log_dict[item] = log_dict[item] * 1.0 / (step - start_step)
    return agents, step, eval_required, checkpoint_reached, epoch_reward, sum(successful_agents), log_dict


def vis_plot(viz, log_dict: Dict[str, List[Tuple[int, float]]]):
    for field in log_dict:
        if '_mva' not in field:
            _, values = zip(*log_dict[field])
            values = np.array(values)
            median_value = np.abs(np.median(values))
            min_value = min(0, max(np.min(values), (-median_value * 10))) if field in ['td_error', 'loss'] else None
            max_value = max(0, min(np.max(values), median_value * 10)) if field in ['td_error', 'loss'] else None

            plot_data = np.array(log_dict[field])
            viz.line(X=plot_data[:, 0], Y=plot_data[:, 1], win=field,
                     opts=dict(title=field, legend=[field], ytickmax=max_value,
                               ytickmin=min_value))
            '''if (field + '_mva') in log_dict:

                plot_data = np.array(log_dict[field + '_mva'])
                viz.line(X=plot_data[:, 0], Y=plot_data[:, 1], win=field, name=field + '_mva',
                         opts=dict(showlegend=True, legend=[field + '_mva'],
                                   ytickmax=max_value,
                                   ytickmin=min_value),
                         update='append')'''
