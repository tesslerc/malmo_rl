import torch
from torch.nn import functional as F
from torch import nn

from policies.models.dqn import Flatten


class DISTRIBUTIONAL_DQN(nn.Module):
    """Implements a 3 layer convolutional network with 2 fully connected layers at the end as explained by:
    Bellamare et al. (2017) - https://arxiv.org/abs/1707.06887

    The final layer projects the features onto (number of actions * number of atoms) supports.
    For each action we receive a probability distribution of future Q values.
    """

    def __init__(self, n_action: int, n_atoms: int, hist_len: int, use_softmax: bool) -> None:
        super(DISTRIBUTIONAL_DQN, self).__init__()
        self.n_action = n_action
        self.n_atoms = n_atoms
        self.use_softmax = use_softmax

        self.sequential_model = nn.Sequential(
            nn.Conv2d(hist_len, 32, kernel_size=8, stride=4, padding=0),  # (In Channel, Out Channel, ...)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, self.n_action * self.n_atoms)
        )

    def forward(self, x):
        quantiles = self.sequential_model(x)

        if self.use_softmax:
            # Returns Q(s, a) probabilities.
            # Probabilities with action over second dimension
            probs = torch.stack([F.softmax(p, dim=1) for p in quantiles.chunk(self.n_action, 1)], 1)

            return probs.clamp(min=1e-8, max=1 - 1e-8)  # Use clipping to prevent NaNs

        else:
            # Returns quantiles, either pre-softmax representation or supports for each quanta.
            quantiles = torch.stack([p for p in quantiles.chunk(self.n_action, 1)], 1)

            return quantiles
