import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F
import numpy as np
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=4,
                            out_channels=32,
                            kernel_size=8,
                            stride=4,
                            padding=0)
        nn.init.orthogonal_(self.conv1.weight, np.sqrt(2))
        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=4,
                               stride=2,
                               padding=0)
        nn.init.orthogonal_(self.conv2.weight, np.sqrt(2))
        self.conv3 = nn.Conv2d(in_channels=64,
                               out_channels=64,
                               kernel_size=3,
                               stride=1,
                               padding=0)
        nn.init.orthogonal_(self.conv3.weight, np.sqrt(2))
        self.lin = nn.Linear(in_features=7 * 7 * 64,
                             out_features=512)
        nn.init.orthogonal_(self.lin.weight, np.sqrt(2))
        self.pi_logits = nn.Linear(in_features=512,
                                   out_features=4)
        nn.init.orthogonal_(self.pi_logits.weight, np.sqrt(0.01))
        self.value = nn.Linear(in_features=512,
                               out_features=1)
        nn.init.orthogonal_(self.value.weight, 1)
        self.load()

    def forward(self, obs: np.ndarray):
        h: torch.Tensor

        h = F.relu(self.conv1(obs))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = h.reshape((-1, 7 * 7 * 64))

        h = F.relu(self.lin(h))

        pi = Categorical(logits=self.pi_logits(h))
        value = self.value(h).reshape(-1)

        return pi, value
    def save(self):
        torch.save(self.cpu().state_dict(), 'game-cnn.pk')
    def load(self):
        self.load_state_dict(torch.load('game-cnn.pk')) 

