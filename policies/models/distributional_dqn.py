import torch
from torch import nn
from torch.nn import functional as F


class DISTRIBUTIONAL_DQN(nn.Module):
    """Implements a 3 layer convolutional network with 2 fully connected layers at the end as explained by:
    Bellamare et al. (2017) - https://arxiv.org/abs/1707.06887
    """

    def __init__(self, n_action: int, n_atoms: int, state_size: int) -> None:
        super(DISTRIBUTIONAL_DQN, self).__init__()
        self.n_action = n_action
        self.n_atoms = n_atoms

        self.conv1 = nn.Conv2d(state_size, 32, kernel_size=8, stride=4, padding=0)  # (In Channel, Out Channel, ...)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)

        self.affine1 = nn.Linear(3136, 512)
        self.affine2 = [nn.Linear(512, self.n_atoms) for _ in range(self.n_action)]

        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, atom_values, log_softmax=False):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))

        h = F.relu(self.affine1(h.view(h.size(0), -1)))

        distribution_list = []
        log_distribution_list = []
        for idx in range(self.n_action):
            # Each action 'a' outputs a discrete n_atoms distribution over Q(s,a).
            atoms = self.affine2[idx](h).unsqueeze(1)  # Add the action dimension.
            distribution_list.append(F.softmax(atoms, dim=-1))
            if log_softmax:
                log_distribution_list.append(self.log_softmax(atoms))
        distributions = torch.cat(distribution_list, 1)

        # Returns Q(s, a) probabilities and E[Q(s, a)]
        if log_softmax:
            log_distributions = torch.cat(log_distribution_list, 1)
            return log_distributions, distributions @ atom_values
        return distributions, distributions @ atom_values
