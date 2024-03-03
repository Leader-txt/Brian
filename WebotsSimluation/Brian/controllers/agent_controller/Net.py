import torch
import torch.nn as nn
from torch.distributions import Normal
import math

# input : picture 1920x1080
class Policy(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.base = nn.Sequential(
            nn.AvgPool2d(kernel_size=8,stride=8),
            nn.ReLU(),
            nn.Conv2d(in_channels=1,out_channels=32,kernel_size=5,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5,stride=5),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,stride=1,padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3,stride=3),
            nn.Flatten(),
            nn.Linear(64*16*9,1024),
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
        state = state.permute(0, 3, 1, 2)
        base = self.base(state)
        mean = self.mean(base)
        std = self.std(base)
        dist = Normal(mean,std)
        normal_sample = dist.rsample()
        action = torch.tanh(normal_sample)
        log_pi = dist.log_prob(normal_sample).sum(dim=1,keepdim=True)
        log_pi -= torch.log(1-action.pow(2)+1e-7).sum(dim=1, keepdim=True)
        action = action*math.pi
        return action , log_pi
    
class Critic(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.base = nn.Sequential(
            nn.AvgPool2d(kernel_size=8,stride=8),
            nn.ReLU(),
            nn.Conv2d(in_channels=1,out_channels=32,kernel_size=5,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5,stride=5),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,stride=1,padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3,stride=3),
            nn.Flatten(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(64*16*9+8,1024),
            nn.ReLU(),
            nn.Linear(1024,1)
        )
    def forward(self,state,action):
        state = state.permute(0, 3, 1, 2)
        base = self.base(state)
        base = torch.cat([base,action],dim=1)
        return self.mlp(base)