import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('policy', nargs=1)
parser.add_argument('agent', nargs=1)
parser.add_argument('--malmo_ports', type=int, nargs='+', default=None,
                    help='If an existing instance of Malmo exists, this provides the port to communicate with it. If not sure - leave flag unused and a Malmo instance will be brought up automatically. Multiple ports are space separated.')
parser.add_argument('--number_of_agents', type=int, default=1,
                    help='Number of agents to run in parallel, all use the same policy. This is used to increase the number of observation samples obtained over time.')
parser.add_argument('--max_steps', type=int, default=2000000)
parser.add_argument('--eval_frequency', type=int, default=1000)
parser.add_argument('--eval_steps', type=int, default=150)
parser.add_argument('--image_width', type=int, default=84)
parser.add_argument('--image_height', type=int, default=84)
# TODO: Add support in models for RGB format.
parser.add_argument('--retain_rgb', default=False, action='store_true')
parser.add_argument('--no_visualization', default=False, action='store_true',
                    help='When flag exists, will not plot visualizations.')
parser.add_argument('--graph_moving_average_length', default=5, type=int,
                    help='Graphs plotted using a moving average over the last N points.')
parser.add_argument('--verbose_prints', default=False, action='store_true',
                    help='Enable verbose debug prints for more informative details.')
parser.add_argument('--available_actions', default=['move 1', 'turn -1', 'turn 1'], nargs='+',
                    help='Space separated list of available actions. E.g. "\'move 1\' \'turn -1\'..."')
