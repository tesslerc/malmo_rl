# command line example:
# python main.py <policy> <domain> --param...

import argparse
import logging
import sys
from typing import Tuple, Dict, List

import numpy as np

try:
    parser: argparse = (__import__("parameters.%s" % sys.argv[1], fromlist=["parameters"])).parser
    params = parser.parse_args()
    logging.basicConfig(format='%(levelname)s: %(message)s',
                        level=(logging.DEBUG if params.verbose_prints else logging.INFO))
except ImportError:
    raise FileNotFoundError('Parameter file for policy ' + sys.argv[1] + ' not found!')

try:
    Policy = (__import__("policies.%s" % sys.argv[1], fromlist=["policies"])).Policy
except ImportError:
    raise FileNotFoundError('Policy file ' + sys.argv[1] + ' not found!')

try:
    Agent = (__import__("agents.%s" % sys.argv[2], fromlist=["agents"])).Agent
except ImportError:
    raise FileNotFoundError('Agent file ' + sys.argv[2] + ' not found!')


def get_os() -> str:
    if sys.platform in ['linux', 'linux2']:
        return 'linux'
    elif sys.platform == 'darwin':  # osx
        raise NotImplementedError('OSX interactive agent is not supported yet.')
    elif sys.platform in ['win32', 'cygwin']:
        return 'win'
    else:
        raise Exception('Unidentified operating system.')


def play_full_episode(_agent: Agent, _policy: Policy, _step: int, _params: argparse, _is_train: bool) -> \
        Tuple[Agent, int, bool, float, Dict[str, float]]:
    _eval_required = False
    _epoch_reward = 0
    _reward, _terminal, _state, _terminal_due_to_timeout = _agent.perform_action(9)
    _log_dict = {}
    _start_step = _step
    while not _terminal:
        _action, _single_log_dict = _policy.get_action(_state, _is_train)
        _reward, _terminal, _state, _terminal_due_to_timeout = _agent.perform_action(_action)
        _policy.update_observation(_reward, _terminal, _terminal_due_to_timeout, _is_train)
        logging.debug('step: %s, reward: %s, terminal: %s, terminal_due_to_timeout: %s', _step, _reward, _terminal,
                      _terminal_due_to_timeout)
        _step += 1
        _epoch_reward += _reward
        if _step % _params.eval_frequency == 0:
            _eval_required = True
        for _item in _single_log_dict:
            if _item in _log_dict:
                _log_dict[_item] = _log_dict[_item] + _single_log_dict[_item]
            else:
                _log_dict[_item] = _single_log_dict[_item]
    for _item in _log_dict:
        _single_log_dict[_item] = _single_log_dict[_item] * 1.0 / (_step - _start_step)
    return _agent, _step, _eval_required, _epoch_reward, _log_dict


def vis_plot(_viz, log_dict: Dict[str, List[Tuple[int, float]]]):
    for _field in log_dict:
        if '_mva' not in _field:
            plot_data = np.array(log_dict[_field])
            _viz.line(X=plot_data[:, 0], Y=plot_data[:, 1], win=_field, opts=dict(title=_field, legend=[_field]))
            if (_field + '_mva') in log_dict:
                plot_data = np.array(log_dict[_field + '_mva'])
                viz.line(X=plot_data[:, 0], Y=plot_data[:, 1], win=_field, name=_field + '_mva',
                         opts=dict(showlegend=True, legend=[_field + '_mva']), update='append')


params.platform = get_os()
if params.no_visualization:
    viz = None
else:
    from visdom import Visdom

    viz = Visdom()
    logging.info('To view results, run \'python -m visdom.server\'')  # activate visdom server on bash
    logging.info('then head over to http://localhost:8097')  # open this address on browser

agent = Agent(params)
policy = Policy(params)
step = 0
train_log_dict: Dict[str, List[Tuple[int, float]]] = {}
eval_log_dict: Dict[str, List[Tuple[int, float]]] = {'episodes': [], 'avg_reward': [], 'max_episode_reward': [],
                                                     'episodes_mva': [], 'avg_reward_mva': [],
                                                     'max_episode_reward_mva': []}

while step < params.max_steps:
    prev_step = step
    agent, step, eval_required, _, episode_log_dict = play_full_episode(agent, policy, step, params, True)
    for field in episode_log_dict:
        if field not in train_log_dict:
            train_log_dict[field] = []
            train_log_dict[field + '_mva'] = []
        train_log_dict[field].append((step, episode_log_dict[field]))
        all_values = np.array(train_log_dict[field])[:, 1]
        mva_length = min(params.graph_moving_average_length, len(all_values))
        moving_average = np.sum(
            all_values[-mva_length:]) * 1.0 / mva_length
        train_log_dict[field + '_mva'].append((step, moving_average))

    if viz is not None:
        vis_plot(viz, train_log_dict)

    if eval_required:
        logging.info('Eval started after %s training steps.', step)
        eval_step = 0
        total_eval_reward = 0
        eval_epochs = 0
        max_eval_epoch_reward = None
        while eval_step < params.eval_steps:
            eval_epochs += 1
            agent, eval_step, _, eval_epoch_reward, _ = play_full_episode(agent, policy, eval_step, params, False)
            total_eval_reward += eval_epoch_reward
            max_eval_epoch_reward = eval_epoch_reward if max_eval_epoch_reward is None else max(max_eval_epoch_reward,
                                                                                                eval_epoch_reward)

        eval_log_dict['episodes'].append((step, eval_epochs))
        eval_log_dict['avg_reward'].append((step, total_eval_reward * 1.0 / eval_epochs))
        eval_log_dict['max_episode_reward'].append((step, max_eval_epoch_reward))
        for field in ['episodes', 'avg_reward', 'max_episode_reward']:
            all_values = np.array(eval_log_dict[field])[:, 1]
            mva_length = min(params.graph_moving_average_length, len(all_values))
            moving_average = np.sum(
                all_values[-mva_length:]) * 1.0 / mva_length
            eval_log_dict[field + '_mva'].append((step, moving_average))

        logging.info('Eval ran for %s steps and a total of %s epochs.', eval_step, eval_epochs)
        logging.info('Average reward during eval (per epoch) is: %s.', total_eval_reward * 1.0 / eval_epochs)
        logging.info('Maximal reward during eval (accumulated over an epoch) is: %s.', max_eval_epoch_reward)

        if viz is not None:
            vis_plot(viz, eval_log_dict)
