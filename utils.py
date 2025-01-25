import random
from typing import List, Dict, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

import torch_geometric.transforms as T



def load_data(dataset_str: str, path: str, device="cuda:0") -> Dict:
    '''
        :param dataset_str:
        :param path:
        :return: {x, edge_index, y, train_mask, val_mask, test_val, num_nodes, num_features, num_classes}
    '''

    dataset_str = dataset_str.lower()
    if dataset_str == "cora":
        dataset = torch_geometric.datasets.Planetoid(path, 'Cora', transform=T.NormalizeFeatures())
        data = dataset[0].to(device)
        dataset_package = {
            "name": "cora",
            "x": data.x,
            "edge_index": data.edge_index,
            "y": data.y,
            "num_classes": dataset.num_classes,
            "num_features": data.num_features,
            "train_mask": data.train_mask,
            "val_mask": data.val_mask,
            "test_mask": data.test_mask,
        }

        return dataset_package
    elif dataset_str == "citeseer":
        dataset = torch_geometric.datasets.Planetoid(path, 'Citeseer', transform=T.NormalizeFeatures())
        data = dataset[0].to(device)

        dataset_package = {
            "name": "citeseer",
            "x": data.x,
            "edge_index": data.edge_index,
            "y": data.y,
            "num_features": data.num_features,
            "num_classes": dataset.num_classes,
            "train_mask": data.train_mask,
            "val_mask": data.val_mask,
            "test_mask": data.test_mask,
        }

        return dataset_package
    elif dataset_str == "pubmed":
        dataset = torch_geometric.datasets.Planetoid(path, 'Pubmed', transform=T.NormalizeFeatures())
        data = dataset[0].to(device)

        dataset_package = {
            "name": "pubmed",
            "x": data.x,
            "edge_index": data.edge_index,
            "y": data.y,
            "num_features": data.num_features,
            "num_classes": dataset.num_classes,
            "train_mask": data.train_mask,
            "val_mask": data.val_mask,
            "test_mask": data.test_mask,
        }
        return dataset_package
    else:
        raise KeyError("Load failed...")



def random_walk_norm(edge_index: torch.Tensor, num_nodes=None) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
        edge_index: with self-loops
    '''
    edge_weight: torch.Tensor = torch.ones(edge_index.shape[1], device=edge_index.device)
    edge_weight = pyg_utils.softmax(src=edge_weight, index=edge_index[1])
    return edge_index, edge_weight



def gcn_conv_norm(edge_index: torch.Tensor, num_nodes=None) -> Tuple[torch.Tensor, torch.Tensor]:
    if not pyg_utils.contains_self_loops(edge_index=edge_index):
        edge_index, _ = pyg_utils.add_remaining_self_loops(edge_index)

    edge_index, edge_weight = pyg_nn.conv.gcn_conv.gcn_norm(edge_index)
    return edge_index, edge_weight
