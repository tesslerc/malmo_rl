from torch import nn


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


class DQN(nn.Module):
    """Implements a 3 layer convolutional network with 2 fully connected layers at the end as explained by:
    Mnih et al. (2013) - https://arxiv.org/abs/1312.5602
    """

    def __init__(self, n_action: int, state_size: int):
        super(DQN, self).__init__()
        self.n_action = n_action

        self.sequential_model = nn.Sequential(
            nn.Conv2d(state_size, 32, kernel_size=8, stride=4, padding=0),  # (In Channel, Out Channel, ...)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, self.n_action)
        )

    def forward(self, x):
        return self.sequential_model(x)
