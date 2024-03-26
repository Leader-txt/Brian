import torch
import torch.nn as nn
from torch.distributions import Normal
import math
import time

# input : motor position and six axises acclurates
class Policy(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.base = nn.Sequential(
            nn.Linear(6+8,1024),
            nn.ReLU(),
        )
        self.mean = nn.Sequential(
            nn.Linear(1024,8),
            nn.Tanh()
        )
        self.std = nn.Sequential(
            nn.Linear(1024,8),
            nn.Softplus()
        )
    def forward(self,state):
        # state = state.permute(0, 3, 1, 2)
        # state = state.view(-1,240,135,1).permute(0,3,1,2)
        base = self.base(state)
        mean = self.mean(base)
        std = self.std(base)
        # print(f'mean:{mean} std:{std}')
        dist = Normal(mean,std)
        normal_sample = dist.rsample()
        action = torch.tanh(normal_sample)
        log_pi = dist.log_prob(normal_sample).sum(dim=1,keepdim=True)
        log_pi -= torch.log(1-action.pow(2)+1e-7).sum(dim=1, keepdim=True)
        # print(action)
        action = action*math.pi*2
        return action , log_pi
    
class Critic(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.base = nn.Sequential(
            nn.Linear(6+8+8,1024),
            nn.ReLU(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(1024,1)
        )
    def forward(self,state,action):
        # state = state.permute(0, 3, 1, 2)
        # state = state.view(-1,240,135,1).permute(0,3,1,2)
        base = self.base(torch.cat([state,action/math.pi/2],dim=1))
        return self.mlp(base)