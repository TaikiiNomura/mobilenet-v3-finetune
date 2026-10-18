from pathlib import Path

import torch
from torch import nn
from torchvision import models


def network(
        weights_path,
        output_dim=2,
):
    model = models.mobilenet_v3_small(weights=None)
    state = torch.load(Path(weights_path), map_location="cpu", weights_only=True)
    model.load_state_dict(state)

    # ImageNetの特徴抽出部とBatchNorm統計は固定し、分類部だけを学習する。
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, output_dim)

    return model
