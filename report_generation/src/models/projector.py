# src/models/projector.py
import torch.nn as nn

class MLPProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = None):
        super().__init__()
        h = hidden_dim or out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.GELU(),
            nn.Linear(h, out_dim),
        )

    def forward(self, x):
        return self.net(x)