from parameters.base_parameters import parser as base_parser

parser = base_parser
parser.add_argument('--epsilon_start', type=float, default=1.0)
parser.add_argument('--epsilon_end', type=float, default=0.1)
parser.add_argument('--epsilon_test', type=float, default=0.05)
parser.add_argument('--epsilon_decay', type=int, default=100000)
parser.add_argument('--target_update_interval', type=int, default=10000)
parser.add_argument('--actively_follow_target', default=False, action='store_true',
                    help='When true, instead of updating the network to the target once every N steps, it will be updated each step by the rule W = (1-alpha)*W\'+alpha*W.')
parser.add_argument('--target_update_alpha', type=float, default=0.01)
parser.add_argument('--checkpoint_interval', type=int, default=5000)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--replay_memory_size', type=int, default=50000)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--num_actions', type=int, default=4, help='W,S,A,D as default keys.')
parser.add_argument('--state_size', type=int, default=4, help='Number of observations that create a "state".')
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--gradient_clipping', type=float, default=1.0,
                    help='For no gradient clipping set to 0. Any other value will clip all gradients to +\- gradient_clipping.')
parser.add_argument('--learn_start', type=int, default=50)
parser.add_argument('--double_dqn', default=False, action='store_true')
parser.add_argument('--normalize_reward', default=False, action='store_true')