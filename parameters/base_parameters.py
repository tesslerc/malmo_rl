import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('policy', nargs=1)
parser.add_argument('agent', nargs=1)
parser.add_argument('--gym_env', type=str, default='Breakout-v0',
                    help='Gym env to run. Only relevant if agent equals \'gym\'.')
parser.add_argument('--malmo_ports', type=int, nargs='+', default=None,
                    help='If an existing instance of Malmo exists, this provides the port to communicate with it. If not sure - leave flag unused and a Malmo instance will be brought up automatically. Multiple ports are space separated.')
parser.add_argument('--number_of_agents', type=int, default=1,
                    help='Number of agents to run in parallel, all use the same policy. This is used to increase the number of observation samples obtained over time.')
parser.add_argument('--max_steps', type=int, default=2000000)
parser.add_argument('--eval_frequency', type=int, default=1000)
parser.add_argument('--eval_steps', type=int, default=100)
parser.add_argument('--image_width', type=int, default=84)
parser.add_argument('--image_height', type=int, default=84)
parser.add_argument('--retain_rgb', default=False, action='store_true')
parser.add_argument('--no_visualization', default=False, action='store_true',
                    help='When flag exists, will not plot visualizations.')
parser.add_argument('--visualization_frequency', default=1, type=int,
                    help='How often to plot visualizations.')
parser.add_argument('--graph_moving_average_length', default=5, type=int,
                    help='Graphs plotted using a moving average over the last N points.')
parser.add_argument('--verbose_prints', default=False, action='store_true',
                    help='Enable verbose debug prints for more informative details.')
parser.add_argument('--ms_per_tick', default=100, type=int,
                    help='Delay between ticks, this is a setting for the Malmo simulator.')
parser.add_argument('--checkpoint_interval', type=int, default=5000)
parser.add_argument('--save_name', type=str, help='The filename given to the saved checkpoint.')
parser.add_argument('--no_cuda', default=False, action='store_true', help='When flag exists, cuda will not be used.')
parser.add_argument('--resume', default=False, action='store_true',
                    help='Resume training of previous model based on save_name.')
