import argparse
import logging
import sys
from typing import Tuple, Dict, List, Any

import numpy as np

from policies.policy import Policy
from utilities.parallel_agents_wrapper import ParallelAgentsWrapper


def get_os() -> str:
    if sys.platform in ['linux', 'linux2']:
        return 'linux'
    elif sys.platform == 'darwin':  # osx
        raise NotImplementedError('OSX interactive agent is not supported yet.')
    elif sys.platform in ['win32', 'cygwin']:
        return 'win'
    else:
        raise Exception('Unidentified operating system.')


def play_full_episode(agents: ParallelAgentsWrapper, policy: Policy, step: int, params: argparse, is_train: bool,
                      viz: Any) -> Tuple[ParallelAgentsWrapper, int, bool, float, Dict[str, float]]:
    eval_required = False
    epoch_reward = 0
    reward, terminal, state, terminal_due_to_timeout = agents.perform_actions(
        ['new game'] * params.number_of_agents)  # Restart all the agents.

    log_dict = {}
    start_step = step
    while not all(terminal):  # Loop ends only when all agents have terminated.
        action = policy.get_action(state, is_train)

        # Send screen of each agent to visdom.
        images = np.zeros((params.number_of_agents, 3, 84, 84))
        for idx in range(params.number_of_agents):
            images[idx, 1, :, :] = state[idx]
            viz.image(images[idx], win='state_agent_' + str(idx), opts=dict(title='Agent ' + str(idx) + '\'s state'))
        reward, terminal, state, terminal_due_to_timeout = agents.perform_actions(action)

        # reward is a list. Passing it to update_observation changes its values hence all references should be
        # performed prior to calling update_observation.
        for r in reward:
            if r is not None:
                epoch_reward += r
        logging.debug('step: %s, reward: %s, terminal: %s, terminal_due_to_timeout: %s', step, reward, terminal,
                      terminal_due_to_timeout)

        policy.update_observation(reward, terminal, terminal_due_to_timeout, is_train)

        if step > params.learn_start and is_train:
            single_log_dict = policy.train()
        else:
            single_log_dict = {}

        step += 1

        if step % params.eval_frequency == 0:
            eval_required = True
        for item in single_log_dict:
            if item in log_dict:
                log_dict[item] = log_dict[item] + single_log_dict[item]
            else:
                log_dict[item] = single_log_dict[item]

    for item in log_dict:
        single_log_dict[item] = single_log_dict[item] * 1.0 / (step - start_step)
    return agents, step, eval_required, epoch_reward, log_dict


def vis_plot(viz, log_dict: Dict[str, List[Tuple[int, float]]]):
    for field in log_dict:
        if '_mva' not in field:
            plot_data = np.array(log_dict[field])
            viz.line(X=plot_data[:, 0], Y=plot_data[:, 1], win=field, opts=dict(title=field, legend=[field]))
            if (field + '_mva') in log_dict:
                plot_data = np.array(log_dict[field + '_mva'])
                viz.line(X=plot_data[:, 0], Y=plot_data[:, 1], win=field, name=field + '_mva',
                         opts=dict(showlegend=True, legend=[field + '_mva']), update='append')
