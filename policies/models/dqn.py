from torch import nn
from torch.nn import functional as F


class DQN(nn.Module):
    """Implements a 3 layer convolutional network with 2 fully connected layers at the end as explained by:
    Mnih et al. (2013) - https://arxiv.org/abs/1312.5602
    """

    def __init__(self, n_action: int):
        super(DQN, self).__init__()
        self.n_action = n_action

        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0)  # (In Channel, Out Channel, ...)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)

        self.affine1 = nn.Linear(3136, 512)
        self.affine2 = nn.Linear(512, self.n_action)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))

        h = F.relu(self.affine1(h.view(h.size(0), -1)))
        h = self.affine2(h)
        return h
