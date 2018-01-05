import argparse

from parameters.dqn import parser as dqn_parser

parser: argparse = dqn_parser
parser.add_argument('--number_of_atoms', type=int, default=51)
parser.add_argument('--min_q_value', default=-10.0, type=float,
                    help='All Q values will be normalized to the range of [min, max].')
parser.add_argument('--max_q_value', default=0.0, type=float,
                    help='All Q values will be normalized to the range of [min, max].')