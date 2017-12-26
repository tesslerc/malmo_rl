import argparse
import copy
import logging
import os
import pickle
import sys
import time
from typing import Tuple, Dict, List

import numpy as np

from utilities import helpers
from utilities.parallel_agents_wrapper import ParallelAgentsWrapper

try:
    parser: argparse = (__import__("parameters.%s" % sys.argv[1], fromlist=["parameters"])).parser
    params = parser.parse_args()
    if params.malmo_ports is not None:
        assert (len(params.malmo_ports) == params.number_of_agents)

    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                        level=(logging.DEBUG if params.verbose_prints else logging.INFO),
                        datefmt='%Y-%m-%d %H:%M:%S')
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

if params.resume:
    with open('saves/' + params.save_name + '.pkl', 'rb') as f:
        checkpoint = pickle.load(f)

    step: int = checkpoint['step']
    train_log_dict: Dict[str, List[Tuple[int, float]]] = checkpoint['train_log_dict']
    eval_log_dict: Dict[str, List[Tuple[int, float]]] = checkpoint['eval_log_dict']
    params = helpers.DotDict(checkpoint['params'])
    params.resume = True

    logging.info('Loaded checkpoint from: saves/' + params.save_name)
else:
    step = 0
    train_log_dict: Dict[str, List[Tuple[int, float]]] = {}
    eval_log_dict: Dict[str, List[Tuple[int, float]]] = {'success_rate': [], 'avg_reward': [], 'max_episode_reward': [],
                                                         'success_rate_mva': [], 'avg_reward_mva': [],
                                                         'max_episode_reward_mva': []}

params.platform = helpers.get_os()
if params.no_visualization:
    viz = None
else:
    from visdom import Visdom
    viz = Visdom()
    logging.info('To view results, run \'python -m visdom.server\'')  # activate visdom server on bash
    logging.info('then head over to http://localhost:8097')  # open this address on browser

    if params.resume:
        helpers.vis_plot(viz, train_log_dict)
        helpers.vis_plot(viz, eval_log_dict)

params.viz = viz

agents = ParallelAgentsWrapper(Agent, params)
policy = Policy(params)

start_time = time.clock()
start_step = step  # Not equal to zero when resuming from checkpoint.
while step < params.max_steps:
    prev_step = step
    agents, step, eval_required, checkpoint_reached, _, _, episode_log_dict = helpers.play_full_episode(agents, policy,
                                                                                                        step, params,
                                                                                                        True)
    for field in episode_log_dict:
        if field not in train_log_dict:
            train_log_dict[field] = []
            train_log_dict[field + '_mva'] = []
        train_log_dict[field].append((step, episode_log_dict[field]))
        all_values = np.array(train_log_dict[field])[:, 1]
        # Length of moving average, relevant for the initial first data points.
        mva_length = min(params.graph_moving_average_length, len(all_values))
        moving_average = float(np.sum(all_values[-mva_length:]) * 1.0 / mva_length)
        train_log_dict[field + '_mva'].append((step, moving_average))

    if params.viz is not None:
        helpers.vis_plot(params.viz, train_log_dict)

    if eval_required:
        eval_clock = time.clock()
        logging.info('Eval started after %s training steps.', step)
        eval_step = 0
        total_eval_reward = 0
        eval_epochs = 0
        total_successful_runs = 0
        max_eval_epoch_reward = None
        while eval_step < params.eval_steps:
            eval_epochs += 1
            agents, eval_step, _, _, eval_epoch_reward, successful_agents, _ = helpers.play_full_episode(agents, policy,
                                                                                                         eval_step,
                                                                                                         params, False)
            total_eval_reward += eval_epoch_reward
            max_eval_epoch_reward = eval_epoch_reward if max_eval_epoch_reward is None else max(
                max_eval_epoch_reward, eval_epoch_reward)
            total_successful_runs += successful_agents

        eval_log_dict['success_rate'].append(
            (step, total_successful_runs * 1.0 / (eval_epochs * params.number_of_agents)))
        eval_log_dict['avg_reward'].append((step, total_eval_reward * 1.0 / (eval_epochs * params.number_of_agents)))
        eval_log_dict['max_episode_reward'].append((step, max_eval_epoch_reward * 1.0 / params.number_of_agents))
        for field in ['success_rate', 'avg_reward', 'max_episode_reward']:
            all_values = np.array(eval_log_dict[field])[:, 1]
            mva_length = min(params.graph_moving_average_length, len(all_values))
            moving_average = float(np.sum(all_values[-mva_length:]) * 1.0 / mva_length)
            eval_log_dict[field + '_mva'].append((step, moving_average))

        logging.info('Eval ran for %s steps and a total of %s epochs.', eval_step, eval_epochs)
        logging.info('Average reward during eval (per epoch) is: %s.', total_eval_reward * 1.0 / eval_epochs)
        logging.info('Maximal reward during eval (accumulated over an epoch) is: %s.', max_eval_epoch_reward)
        logging.info('Train speed: %s steps/second. Test speed: %s steps/second',
                     (step - start_step) * 1.0 / (time.clock() - start_time),
                     eval_step * 1.0 / (time.clock() - eval_clock))

        if params.viz is not None:
            helpers.vis_plot(params.viz, eval_log_dict)

    if checkpoint_reached and params.save_name is not None:
        params.viz = None
        if type(params) == type(helpers.DotDict()):
            save_params = copy.deepcopy(dict(params))
        else:
            save_params = copy.deepcopy(vars(params))

        checkpoint = {
            'params': save_params,
            'step': step,
            'eval_log_dict': eval_log_dict,
            'train_log_dict': train_log_dict
        }
        filename = 'saves/' + params.save_name + '.pkl'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f, pickle.HIGHEST_PROTOCOL)
        policy.save_state()

        params.viz = viz
        logging.info('Saved checkpoint to: saves/' + params.save_name)
