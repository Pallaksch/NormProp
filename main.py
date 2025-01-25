import argparse
from typing import List, Tuple, Dict

import numpy as np

import torch
import torch.nn.functional as F
import torch_geometric
import torch_geometric.utils
import torch_geometric.transforms

import models
from models import NormProp


import utils


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora', help='Choose dataset.')
parser.add_argument('--seed', type=int, default=42, help='Set random seed.')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
parser.add_argument('--weight-decay', type=float, default=5e-3, help='Weight decay (L2).')
parser.add_argument('--nhid', type=int, default=16, help='Dim of hidden layer.')
parser.add_argument('--epochs', type=int, default=300, help='Epochs.')
parser.add_argument('--path', type=str, default='./datasets/pyg/', help='Path of dataset cache')
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout.')
parser.add_argument('--polars-path', type=str, default='./polars', help='The path of polars\'s numpy file.')
parser.add_argument('--mu', type=float, default=0.0, help='Weight of L2 norm loss.')
parser.add_argument('--conf-threshold', type=float, default=0.9, help='Threshold of confidence.')
parser.add_argument('--batch-size', type=int, default=5000, help='Batch size.')
parser.add_argument('--warmup', type=int, default=10, help='Warmup epoch.')
parser.add_argument('--split', type=int, default=0, help='Train/Val/Test split id.')
parser.add_argument('--K', type=int, default=2, help='Aggregate K-hop neighborhoods.')
args = parser.parse_args()


# settings of training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch_geometric.seed_everything(args.seed)


# load semi-supervised dataset
dataset_package = utils.load_data(args.dataset, args.path)


x: torch.Tensor             = dataset_package['x']
y: torch.Tensor             = dataset_package['y']
edge_index: torch.Tensor    = dataset_package['edge_index']
edge_index, _               = torch_geometric.utils.add_remaining_self_loops(edge_index=edge_index, num_nodes=x.shape[0])
train_mask: torch.Tensor    = dataset_package['train_mask']
val_mask: torch.Tensor      = dataset_package['val_mask']
test_mask: torch.Tensor     = dataset_package['test_mask']
num_features: int           = dataset_package['num_features']
num_labels: int             = dataset_package['num_classes']

# _, edge_weight              = utils.random_walk_norm(edge_index, x.shape[0])
_, edge_weight              = utils.gcn_conv_norm(edge_index, x.shape[0])


polars                      = np.load(f"{args.polars_path}/polars-{num_labels}-{args.nhid}.npy")
polars                      = torch.tensor(polars).to(device)


# settings of model
model = NormProp(num_features, args.nhid, num_labels, polars, dropout=args.dropout).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
norm_loss_flag: bool = False


# TODO: solve upper bound of Euclidean Norm
upper_bound_list: List[torch.Tensor] = list()
norm: torch.Tensor = torch.ones(x.shape[0], 1, device=x.device)
upper_bound_list.append(norm.view(-1))
for _ in range(model.K):
    norm = model.propagation(norm, edge_index, edge_weight)
    upper_bound_list.append(norm.view(-1))


# TODO: detect remote nodes
# from bias_utils import detect_remote_nodes
# distance_score: torch.Tensor = detect_remote_nodes(edge_index, edge_weight, train_mask, model.propagation, args.K)
# remote_mask: torch.Tensor = distance_score <= 0
# print(f"Remote nodes: {remote_mask.sum().item()}, min = {distance_score.min().item()}, max = {distance_score.max().item()}")


def random_sample(h: torch.Tensor, size: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    random_perm: torch.Tensor = torch.randperm(h.shape[0], device=h.device)
    mask_idx: torch.Tensor = random_perm[: size]
    return h[mask_idx], mask_idx


def train():
    model.train()
    optimizer.zero_grad()

    embedding = model(x, edge_index, edge_weight)

    out = model.classification_based_on_cosine(embedding[-1])
    confidence = torch.max(out, dim=-1)[0]
    train_out = torch.gather(out[train_mask], dim=-1, index=y[train_mask].view(-1, 1))

    embedding_norms = list()
    for emb in embedding:
        emb_norm = (emb * emb).sum(dim=-1).sqrt()
        embedding_norms.append(emb_norm)

    conf_mask = confidence > args.conf_threshold
    # conf_mask = conf_mask & remote_mask
    conf_mask = conf_mask & (~train_mask)
    conf_mask = conf_mask & (~val_mask)

    global norm_loss_flag
    if norm_loss_flag:
        if conf_mask.sum().item() > 0:
            norm_loss = embedding_norms[-1] / upper_bound_list[-1]
            # filter using confidence, e.g. max similarity
            norm_loss = norm_loss[conf_mask]
            # random sample
            random_idx: torch.Tensor = torch.randperm(norm_loss.shape[0], device=x.device)[ : args.batch_size]
            norm_loss = 1 - norm_loss[random_idx].mean()
            # print(f"debug: {confidence_mask.sum().item()}")
        else:
            norm_loss = 0
    else:
        norm_loss = 0
        conf_loss = 0

    cos_diff = 1 - train_out
    # loss = (cos_diff * cos_diff).mean() + (norm_loss * norm_loss) * args.mu
    loss = (cos_diff).mean() + (norm_loss) * args.mu
    loss.backward()

    optimizer.step()
    return float(loss)



@torch.no_grad()
def test():
    model.eval()
    out = model(x, edge_index, edge_weight)
    # out_norm = [(elem * elem).sum(dim=-1) for elem in out]

    predictions: List[torch.Tensor] = list()
    confidence: List[torch.Tensor] = list()
    norms: List[torch.Tensor] = list()
    for i in range(len(out)):
        distance_of_polars: torch.Tensor = model.classification_based_on_cosine(out[i])
        pred_dist, pred = torch.max(distance_of_polars, dim=-1)
        norm: torch.Tensor = (out[i] * out[i]).sum(dim=-1).sqrt()
        predictions.append(pred)
        confidence.append(pred_dist)
        norms.append(norm)

    accs = []
    for mask in [train_mask, val_mask, test_mask]:
        accs.append(int((pred[mask] == y[mask]).sum()) / mask.sum().item())
    return accs


if __name__ == '__main__':
    best_val_acc = test_acc = 0
    for epoch in range(1, args.epochs + 1):
        if epoch > args.warmup:
            norm_loss_flag = True

        loss = train()
        train_acc, val_acc, tmp_test_acc = test()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc

        print(f"---Epoch {epoch}, loss = {loss:.5f}, train-acc = {train_acc:.3f}, val = {val_acc:.3f}, test = {tmp_test_acc:.3f}")

    print(f"\nBest val test = {best_val_acc:.3f}\nTest acc = {test_acc:.4f}")

    print()
    print(test_acc)
