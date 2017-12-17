import torch
from torch import nn
from torch.nn import functional as F


class DISTRIBUTIONAL_DQN(nn.Module):
    """Implements a 3 layer convolutional network with 2 fully connected layers at the end as explained by:
    Bellamare et al. (2017) - https://arxiv.org/abs/1707.06887
    """

    def __init__(self, n_action: int, n_atoms: int) -> None:
        super(DISTRIBUTIONAL_DQN, self).__init__()
        self.n_action = n_action
        self.n_atoms = n_atoms

        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0)  # (In Channel, Out Channel, ...)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)

        self.affine1 = nn.Linear(3136, 512)

        self.affine2 = []
        for _ in range(self.n_action):
            self.affine2.append(nn.Linear(512, self.n_atoms))

    def forward(self, x, atom_values):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))

        h = F.relu(self.affine1(h.view(h.size(0), -1)))

        distributions = []
        for idx in range(self.n_action):
            # Each action 'a' outputs a discrete n_atoms distribution over Q(s,a).
            atoms = self.affine2[idx](h).unsqueeze(1)
            distributions.append(F.softmax(atoms, dim=2))
        distributions = torch.cat(distributions, 1)

        # Returns Q(s, a) probabilities and E[Q(s, a)]
        return distributions, distributions @ atom_values
