import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('policy', nargs=1)
parser.add_argument('agent', nargs=1)
parser.add_argument('--malmo_port', type=int, default=None,
                    help='If an existing instance of Malmo exists, this provides the port to communicate with it. If not sure - leave flag unused and a Malmo instance will be brought up automatically.')
parser.add_argument('--max_steps', type=int, default=2000000)
parser.add_argument('--eval_frequency', type=int, default=10000)
parser.add_argument('--eval_steps', type=int, default=1000)
parser.add_argument('--image_width', type=int, default=84)
parser.add_argument('--image_height', type=int, default=84)
# TODO: Add support in models for RGB format.
parser.add_argument('--retain_rgb', default=False, action='store_true')
