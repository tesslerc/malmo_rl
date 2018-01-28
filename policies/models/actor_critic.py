from torch import nn
import torch


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


class ACTOR_CRITIC(nn.Module):
    def __init__(self, n_action: int, state_size: int, value_projection_outputs: int =1):
        super(ACTOR_CRITIC, self).__init__()
        self.n_action = n_action
        self.lstm_size = 512

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(state_size, 32, kernel_size=8, stride=4, padding=0),  # (In Channel, Out Channel, ...)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU()
        )

        self.lstm = nn.LSTMCell(512, self.lstm_size)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.v_projection = nn.Linear(self.lstm_size, value_projection_outputs)
        self.policy_projection = nn.Linear(self.lstm_size, self.n_action)

        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)

        self.train()

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = self.feature_extractor(inputs)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        logits = self.policy_projection(x)
        policy = self.softmax(logits)
        log_policy = self.log_softmax(logits)

        V = self.v_projection(x)
        return policy, log_policy, V, (hx, cx)


class DISTRIBUTIONAL_ACTOR_CRITIC(ACTOR_CRITIC):
    def __init__(self, n_action: int, n_atoms: int, state_size: int):
        super(DISTRIBUTIONAL_ACTOR_CRITIC, self).__init__(n_action, state_size, n_atoms * n_action)
        self.n_atoms = n_atoms

    def forward(self, x):
        policy, log_policy, flat_quantiles, lstm_state = super(DISTRIBUTIONAL_ACTOR_CRITIC, self).forward(x)
        quantiles = torch.stack([p for p in flat_quantiles.chunk(self.n_action, 1)], 1)

        expected_quantiles = quantiles * policy.unsqueeze(-1)
        return policy, log_policy, expected_quantiles.sum(1), lstm_state
