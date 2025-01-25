from typing import List, Tuple, Dict, Any

import numpy as np

import torch
import torch.nn.functional as F
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils



def detect_remote_nodes(edge_index: torch.Tensor, edge_weight: torch.Tensor, labels: torch.Tensor, propagation, k: int = 1) -> torch.Tensor:
    assert callable(propagation)

    labels = labels.float().view(-1, 1)
    distance_score: torch.Tensor = propagation(labels, edge_index, edge_weight)

    for _ in range(k - 1):
        distance_score = propagation(distance_score, edge_index, edge_weight)

    distance_score = distance_score.view(-1)
    return distance_score


def soft_label_propagation(edge_index: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, propagation, k: int = 1) -> torch.Tensor:
    assert callable(propagation)

    soft_label: torch.Tensor = F.one_hot(y)
    soft_label[~mask] = 0

    for _ in range(k):
        soft_label = propagation(soft_label, edge_index)

    return soft_label
