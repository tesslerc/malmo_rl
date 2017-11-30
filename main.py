# command line example:
# python main.py <policy> <domain> --param...

import sys
try:
    parser = (__import__("parameters.%s" % sys.argv[1], fromlist=["parameters"])).parser
    params = parser.parse_args()
except ImportError:
    raise FileNotFoundError('Parameter file for policy ' + sys.argv[1] + ' not found!')

try:
    Policy = (__import__("policies.%s" % sys.argv[1], fromlist=["policies"])).Policy
except ImportError:
    raise FileNotFoundError('Policy file ' + sys.argv[1] + ' not found!')

try:
    Agent = (__import__("malmo.%s" % sys.argv[2], fromlist=["malmo"])).Agent
except ImportError:
    raise FileNotFoundError('Agent file ' + sys.argv[2] + ' not found!')


def get_os():
    if sys.platform in ['linux', 'linux2']:
        return 'linux'
    elif sys.platform == 'darwin':  # osx
        raise NotImplementedError('OSX interactive agent is not supported yet.')
    elif sys.platform in ['win32', 'cygwin']:
        return 'win'
    else:
        raise Exception('Unidentified operating system.')


def play_until_termination(_agent, _step, _params, _is_train):
    _eval_required = False
    _epoch_reward = 0
    _reward, _terminal, _state = _agent.perform_action(9)
    while not _terminal:
        _action = policy.get_action(_reward, _terminal, _state, _is_train)
        _reward, _terminal, _state = _agent.perform_action(_action)
        _step += 1
        _epoch_reward += _reward
        if _step % _params.eval_frequency == 0:
            _eval_required = True
    return _agent, _step, _eval_required, _epoch_reward


params.platform = get_os()
agent = Agent(params)
policy = Policy(params)
step = 0

while step < params.max_steps:
    agent, step, eval_required, epoch_reward = play_until_termination(agent, step, params, True)

    if eval_required:
        eval_step = 0
        total_eval_reward = 0
        eval_epochs = 0
        max_eval_epoch_reward = None
        while eval_step < params.eval_steps:
            eval_epochs += 1
            agent, eval_step, _, eval_epoch_reward = play_until_termination(agent, eval_step, params, False)
            total_eval_reward += eval_epoch_reward
            max_eval_epoch_reward = eval_epoch_reward if max_eval_epoch_reward is None else max(max_eval_epoch_reward,
                                                                                                eval_epoch_reward)
        print('Eval ended after ' + str(eval_step) + ' steps and a total of ' + str(eval_epochs) + ' epochs.')
        print('Average reward during eval (per epoch) is: ' + str(total_eval_reward * 1.0 / eval_epochs))
        print('Maximal reward during eval (accumulated over an epoch) is: ' + str(max_eval_epoch_reward))
