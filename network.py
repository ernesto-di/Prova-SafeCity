import torch
import torch.nn as nn

class DualHeadDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DualHeadDQN, self).__init__()
        # Input: 32 (16 features * 2 incroci)
        
        self.common = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Head per Incrocio 0
        self.head0 = nn.Linear(64, output_dim)
        # Head per Incrocio 1
        self.head1 = nn.Linear(64, output_dim)

    def forward(self, x):
        features = self.common(x)
        out0 = self.head0(features)
        out1 = self.head1(features)
        return out0, out1