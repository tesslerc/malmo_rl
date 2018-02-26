from torch import nn
import torch


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


class ACTOR_CRITIC(nn.Module):
    def __init__(self, n_action: int, hist_len: int):
        super(ACTOR_CRITIC, self).__init__()
        self.n_action = n_action

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(hist_len, 32, kernel_size=8, stride=4, padding=0),  # (In Channel, Out Channel, ...)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            Flatten()
        )

        self.v_projection = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.policy_projection = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, self.n_action)
        )

        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs):
        x = self.feature_extractor(inputs)

        logits = self.policy_projection(x)
        policy = self.softmax(logits)
        log_policy = self.log_softmax(logits)

        V = self.v_projection(x)
        return policy, log_policy, V
