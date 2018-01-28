from parameters.base_parameters import parser as base_parser

parser = base_parser
parser.add_argument('--state_size', type=int, default=4, help='Number of observations that create a "state".')
parser.add_argument('--lr', type=float, default=0.0000625)
parser.add_argument('--normalize_reward', default=False, action='store_true')
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--gradient_clipping', type=float, default=1.0,
                    help='For no gradient clipping set to 0. Any other value will clip all gradients to +\- gradient_clipping.')
parser.add_argument('--tau', type=float, default=1.0)
parser.add_argument('--learn_frequency', type=int, default=4, help='Number of steps between calls to policy train.')
