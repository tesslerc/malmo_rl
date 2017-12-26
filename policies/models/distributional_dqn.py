import torch
from torch.nn import functional as F

from policies.models.dqn import DQN


class DISTRIBUTIONAL_DQN(DQN):
    """Implements a 3 layer convolutional network with 2 fully connected layers at the end as explained by:
    Bellamare et al. (2017) - https://arxiv.org/abs/1707.06887
    """

    def __init__(self, n_action: int, n_atoms: int, state_size: int) -> None:
        super(DISTRIBUTIONAL_DQN, self).__init__(n_atoms * n_action, state_size)
        self.n_action = n_action
        self.n_atoms = n_atoms

    def forward(self, x):
        output = super(DISTRIBUTIONAL_DQN, self).forward(x)

        # Returns Q(s, a) probabilities and E[Q(s, a)]
        # Probabilities with action over second dimension
        probs = torch.stack([F.softmax(p, dim=1) for p in output.chunk(self.n_action, 1)], 1)

        return probs.clamp(min=1e-8, max=1 - 1e-8)  # Use clipping to prevent NaNs
