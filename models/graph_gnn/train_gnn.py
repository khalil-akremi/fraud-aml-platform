"""
Trains a 2-layer GraphSAGE GNN for node classification: is this customer
node a ring member, given both its own behavioral features AND the graph
structure connecting it to merchants and other customers?

Why GraphSAGE specifically: it aggregates each node's neighbors' features
via mean-pooling and combines that with the node's own features at each
layer - simple, well-understood, and works well on bipartite-style graphs
like ours (customer-merchant) without needing attention mechanisms (GAT)
that add complexity without a clear benefit at this graph size.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.metrics import classification_report
import numpy as np

from build_gnn_graph import build_graph_data


class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.out = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return self.out(x).squeeze(-1)


def main():
    data = build_graph_data(
        "data/raw/transactions.csv", "data/raw/merchants.csv", "data/raw/ring_ground_truth.csv"
    )
    x, edge_index, y = data["x"], data["edge_index"], data["y"]
    customer_mask = data["customer_mask"]

    # split customer nodes into train/val, stratified so both sets have ring examples
    customer_indices = torch.where(customer_mask)[0]
    ring_indices = customer_indices[y[customer_indices] == 1]
    normal_indices = customer_indices[y[customer_indices] == 0]

    rng = np.random.default_rng(42)
    ring_perm = ring_indices[rng.permutation(len(ring_indices))]
    normal_perm = normal_indices[rng.permutation(len(normal_indices))]

    n_ring_train = int(len(ring_perm) * 0.7)
    n_normal_train = int(len(normal_perm) * 0.7)

    train_idx = torch.cat([ring_perm[:n_ring_train], normal_perm[:n_normal_train]])
    val_idx = torch.cat([ring_perm[n_ring_train:], normal_perm[n_normal_train:]])

    train_mask = torch.zeros(len(y), dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask = torch.zeros(len(y), dtype=torch.bool)
    val_mask[val_idx] = True

    print(f"Train nodes: {train_mask.sum().item()} ({int(y[train_mask].sum())} ring)")
    print(f"Val nodes: {val_mask.sum().item()} ({int(y[val_mask].sum())} ring)")

    model = GraphSAGE(in_channels=x.shape[1], hidden_channels=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # class weighting for severe imbalance, same principle as the XGBoost fraud model
    n_neg = (y[train_mask] == 0).sum()
    n_pos = (y[train_mask] == 1).sum()
    pos_weight = (n_neg / n_pos).clone().detach()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = loss_fn(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 40 == 0:
            print(f"Epoch {epoch}, loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
        probs = torch.sigmoid(out)
        preds = (probs > 0.5).float()

    print("\n--- Validation results ---")
    print(classification_report(
        y[val_mask].numpy(), preds[val_mask].numpy(), digits=3, zero_division=0
    ))

    # show which ring members were caught vs missed
    val_ring_idx = val_idx[y[val_idx] == 1]
    customer_ids_arr = np.array(data["customer_ids"])
    print("Ring members in validation set:")
    for idx in val_ring_idx:
        cid = customer_ids_arr[idx.item()]
        print(f"  {cid}: predicted_prob={probs[idx].item():.3f}, correct={bool(preds[idx].item()==1)}")


if __name__ == "__main__":
    main()