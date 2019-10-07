import torch as th
from torch import nn
from torch.functional import F
import numpy as np


class DAE(nn.Module):
    def __init__(self,input_dim,output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim,1500)
        self.layer2 = nn.Linear(1500,1500)
        self.layer3 = nn.Linear(1500,1500)
        self.layer4 = nn.Linear(1500,output_dim)
    def forward(self,x, predict):
        h1 = F.relu(self.layer1(x))
        h2 = F.relu(self.layer2(h1))
        h3 = F.relu(self.layer3(h2))
        if predict:
            return np.c_[h1.detach().numpy(),h2.detach().numpy(),h3.detach().numpy()]
        return self.layer4(h3)
