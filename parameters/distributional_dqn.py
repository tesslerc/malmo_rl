import argparse

from parameters.dqn import parser as dqn_parser

parser: argparse = dqn_parser
parser.add_argument('--number_of_atoms', type=int, default=51)
