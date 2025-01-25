from typing import Dict, List, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric as pyg
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils



class Encoder_fc(nn.Module):
    '''
        Linear encoder
    '''
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_channels, out_channels)

        self.act_fn = F.leaky_relu
        # self.act_fn = F.relu
        # self.layer_norm = nn.LayerNorm(hidden_channels, eps=1e-6)

        self.reset_parameters()


    def reset_parameters(self) -> None:
        self.fc.reset_parameters()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.fc(x)
        return x



class Encoder(nn.Module):
    '''
        MLP encoder
    '''
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)

        # self.act_fn = F.leaky_relu
        self.act_fn = F.relu
        self.layer_norm = nn.LayerNorm(hidden_channels, eps=1e-6)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.layer_norm.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.layer_norm(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x



# Message-Passing (GCN: $D^{-1/2} A D^{-1/2}$)
class GCNAggr(pyg_nn.MessagePassing):
    '''
        source_2_target
    '''
    def __init__(self, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None) -> torch.Tensor:
        # add self-loops
        if not pyg_utils.contains_self_loops(edge_index):
            # edge_index, edge_weight = pyg_utils.remove_self_loops(edge_index, edge_weight)
            edge_index, _ = pyg_utils.add_remaining_self_loops(edge_index)

        # calculate edge_weight
        if edge_weight is None:
            edge_index, edge_weight = pyg_nn.conv.gcn_conv.gcn_norm(edge_index)

        x = self.propagate(edge_index, edge_weight=edge_weight, x=x, size=None)

        return x

    def message(self, x_j: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return pyg_utils.spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}')



class NormProp(nn.Module):
    '''
        Metric learning based on prototype
    '''
    def __init__(self, in_channels: int, embedding_dim: int, n_prototype: int, polars: torch.Tensor, dropout: float = 0.5):
        super().__init__()
        self.in_channels: int = in_channels
        self.n_hidden: int = 32
        self.embedding_dim: int = embedding_dim
        self.n_prototype: int = n_prototype
        self.drop_p: float = dropout
        self.dropout = nn.Dropout(p = dropout)
        self.K: int = 2

        # self.encoder = Encoder(in_channels, embedding_dim, dropout=dropout)
        self.encoder = Encoder(in_channels, self.n_hidden, embedding_dim, dropout=dropout)
        self.propagation = GCNAggr()

        assert tuple(polars.shape) == (n_prototype, embedding_dim)
        self.prototypes: torch.Tensor = polars

        self.reset_parameters()


    def reset_parameters(self) -> None:
        # reset parameters of encoder
        self.encoder.reset_parameters()
        # reset parameters of propagation
        self.propagation.reset_parameters()


    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None):
        h0: torch.Tensor = self.encoder(x)
        h0 = F.normalize(h0, p=2, dim=-1)

        hs: List[torch.Tensor] = list()
        h = h0
        hs.append(h)

        for _ in range(self.K):
            h = self.propagation(h, edge_index, edge_weight)
            hs.append(h)

        return hs


    def classification_based_on_cosine(self, h: torch.Tensor) -> torch.Tensor:
        assert h.shape[1] == self.prototypes.shape[1]
        h = F.normalize(h, p=2, dim=-1)
        distance: torch.Tensor = h @ self.prototypes.T
        return distance
