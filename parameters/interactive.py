from parameters.base_parameters import parser as base_parser

parser = base_parser
parser.add_argument('--learn_start', type=int, default=100000)  # Need some value, not really relevant.
