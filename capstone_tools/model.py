from dataclasses import dataclass
from typing import Tuple
import torch.nn as nn


class ModelError(Exception):
    """Raised when dimensions fail validity checks"""


@dataclass
class ModelDimensions:
    input_size: int
    hidden_layers: Tuple[int]
    output_size: int


class RecommendationModel(nn.Module):
    """Recommendation Model for Starbucks Metric Data"""

    def __init__(
        self,
        dimensions: ModelDimensions,
        has_dropout: bool = False,
        has_batch_norm: bool = False,
        p_drop=0.2,
    ):
        super().__init__()
        layers = []
        in_layer = dimensions.input_size
        n_hidden = len(dimensions.hidden_layers)
        for i, out_layer in enumerate(dimensions.hidden_layers):
            layers.extend(
                [
                    nn.Linear(in_layer, out_layer),
                    nn.ReLU(),
                ]
            )
            if has_dropout and i + 1 < n_hidden:
                layers.append(nn.Dropout(p_drop))

            if has_batch_norm and i + i < n_hidden:
                layers.append(nn.BatchNorm1d(out_layer))

            in_layer = out_layer
        layers.append(nn.Linear(out_layer, dimensions.output_size))
        self.linear_stack = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.linear_stack(x)
        return logits
