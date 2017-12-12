import argparse
import logging
import numpy as np
import sys
import time
from typing import Tuple, Dict, List

from utilities import helpers
from utilities.parallel_agents_wrapper import ParallelAgentsWrapper

try:
    parser: argparse = (__import__("parameters.%s" % sys.argv[1], fromlist=["parameters"])).parser
    params = parser.parse_args()
    if params.malmo_ports is not None:
        assert (len(params.malmo_ports) == params.number_of_agents)

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

params.platform = helpers.get_os()
if params.no_visualization:
    viz = None
else:
    from visdom import Visdom

    viz = Visdom()
    logging.info('To view results, run \'python -m visdom.server\'')  # activate visdom server on bash
    logging.info('then head over to http://localhost:8097')  # open this address on browser

agents = ParallelAgentsWrapper(Agent, params)
policy = Policy(params)
step = 0
train_log_dict: Dict[str, List[Tuple[int, float]]] = {}
eval_log_dict: Dict[str, List[Tuple[int, float]]] = {'episodes': [], 'avg_reward': [], 'max_episode_reward': [],
                                                     'episodes_mva': [], 'avg_reward_mva': [],
                                                     'max_episode_reward_mva': []}

start_time = time.clock()
while step < params.max_steps:
    prev_step = step
    agents, step, eval_required, _, episode_log_dict = helpers.play_full_episode(agents, policy, step, params, True,
                                                                                 viz)
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

    if viz is not None:
        helpers.vis_plot(viz, train_log_dict)

    if eval_required:
        eval_clock = time.clock()
        logging.info('Eval started after %s training steps.', step)
        eval_step = 0
        total_eval_reward = 0
        eval_epochs = 0
        max_eval_epoch_reward = None
        while eval_step < params.eval_steps:
            eval_epochs += 1
            agents, eval_step, _, eval_epoch_reward, _ = helpers.play_full_episode(agents, policy, eval_step, params,
                                                                                   False, viz)
            total_eval_reward += eval_epoch_reward
            max_eval_epoch_reward = eval_epoch_reward if max_eval_epoch_reward is None else max(
                max_eval_epoch_reward, eval_epoch_reward)

        eval_log_dict['episodes'].append((step, eval_epochs))
        eval_log_dict['avg_reward'].append((step, total_eval_reward * 1.0 / (eval_epochs * params.number_of_agents)))
        eval_log_dict['max_episode_reward'].append((step, max_eval_epoch_reward * 1.0 / params.number_of_agents))
        for field in ['episodes', 'avg_reward', 'max_episode_reward']:
            all_values = np.array(eval_log_dict[field])[:, 1]
            mva_length = min(params.graph_moving_average_length, len(all_values))
            moving_average = float(np.sum(all_values[-mva_length:]) * 1.0 / mva_length)
            eval_log_dict[field + '_mva'].append((step, moving_average))

        logging.info('Eval ran for %s steps and a total of %s epochs.', eval_step, eval_epochs)
        logging.info('Average reward during eval (per epoch) is: %s.', total_eval_reward * 1.0 / eval_epochs)
        logging.info('Maximal reward during eval (accumulated over an epoch) is: %s.', max_eval_epoch_reward)
        logging.info('Train speed: %s steps/second. Test speed: %s steps/second',
                     step * 1.0 / (time.clock() - start_time),
                     eval_step * 1.0 / (time.clock() - eval_clock))

        if viz is not None:
            helpers.vis_plot(viz, eval_log_dict)
