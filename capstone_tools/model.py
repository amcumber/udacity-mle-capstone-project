from dataclasses import dataclass
from itertools import zip_longest
from typing import Tuple
import torch
import torch.nn as nn


class ModelError(Exception):
    """Raised when dimensions fail validity checks"""


@dataclass
class ModelDimensions:
    input_size: int
    hidden_layers: Tuple[int]
    output_size: int

    def __post_init__(self):
        n = len(self.hidden_layers)
        if n > 2:
            raise ModelError(f"Only 1 or 2 hidden layers are allowed: {n}")


class RecommendationModel(nn.Module):
    """Recommendation Model for Starbucks Metric Data"""

    def __init__(
        self,
        dimensions: ModelDimensions,
        act_fun: callable = torch.relu,
        has_dropout: bool = False,
        p_drop=0.2,
    ):
        super().__init__()
        hidden_layers = iter(dimensions.hidden_layers)
        h1 = next(hidden_layers)
        self.bn1 = nn.BatchNorm1d(dimensions.input_size)
        self.fc1 = nn.Linear(dimensions.input_size, h1)

        self.drop = None
        if has_dropout:
            self.drop = nn.Dropout(p_drop)

        self.fc2 = None
        self.bn2 = None

        try:
            h2 = next(hidden_layers)
            self.bn2 = nn.BatchNorm1d(h1)
            self.fc2 = nn.Linear(h1, h2)
        except StopIteration:
            h2 = h1

        self.fc3 = nn.Linear(h2, dimensions.output_size)

        self.act_fun = act_fun
        self.out_fun = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.bn1(x)
        x = self.fc1(x)
        x = self.act_fun(x)
        if self.drop:
            x = self.drop(x)
        if self.fc2:
            x = self.bn2(x)
            x = self.fc2(x)
            x = self.act_fun(x)
        x = self.fc3(x)
        return self.out_fun(x)
