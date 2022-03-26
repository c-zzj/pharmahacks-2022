from classifier import *
from torch import nn
from typing import Tuple

class SimpleMLP(Module):
    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1094, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(100, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(100, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(100, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(100, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(100, 3),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)