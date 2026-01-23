import torch.nn as nn

class DualHeadDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        # ============================
        # ENCODER CONDIVISO
        # ============================
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # ============================
        # HEAD INCROCIO 0
        # ============================
        self.head0 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

        # ============================
        # HEAD INCROCIO 1
        # ============================
        self.head1 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        shared = self.encoder(x)

        q0 = self.head0(shared)
        q1 = self.head1(shared)

        return q0, q1
