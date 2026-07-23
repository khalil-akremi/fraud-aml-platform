"""
Visualizes a readable slice of the full GNN graph (5,300 nodes total -
too dense to draw as one image). Shows: all ring customer nodes, the
merchants they connect to, and a sample of normal customers who share
those same merchants - enough to see the actual structure the GNN
message-passes over, without an unreadable hairball.
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

transactions = pd.read_csv("data/raw/transactions.csv")
ground_truth = pd.read_csv("data/raw/ring_ground_truth.csv")

ring_customer_ids = set(ground_truth["customer_id"])
ring_id_map = dict(zip(ground_truth["customer_id"], ground_truth["ring_id"]))

# merchants touched by ring customers
ring_tx = transactions[transactions["customer_id"].isin(ring_customer_ids)]
ring_merchant_ids = set(ring_tx["merchant_id"].unique())

# sample of normal customers who ALSO use those same merchants (for contrast)
normal_tx_at_ring_merchants = transactions[
    (transactions["merchant_id"].isin(ring_merchant_ids)) &
    (~transactions["customer_id"].isin(ring_customer_ids))
]
sample_normal_customers = set(
    normal_tx_at_ring_merchants["customer_id"].drop_duplicates().sample(
        n=min(25, normal_tx_at_ring_merchants["customer_id"].nunique()), random_state=1
    )
)

relevant_customers = ring_customer_ids | sample_normal_customers
relevant_tx = transactions[
    transactions["customer_id"].isin(relevant_customers) &
    transactions["merchant_id"].isin(ring_merchant_ids)
]

G = nx.Graph()
for _, row in relevant_tx.iterrows():
    G.add_edge(row["customer_id"], row["merchant_id"])

# color scheme: ring members by ring_id, merchants gray, normal customers light blue
ring_colors = {"A_ip_sharing": "#e74c3c", "B_merchant_concentration": "#f39c12", "C_subtle": "#9b59b6"}
node_colors, node_sizes, labels = [], [], {}
for node in G.nodes():
    if node in ring_customer_ids:
        node_colors.append(ring_colors[ring_id_map[node]])
        node_sizes.append(300)
        labels[node] = node.replace("CUST_", "")
    elif node in ring_merchant_ids:
        node_colors.append("#7f8c8d")
        node_sizes.append(500)
        labels[node] = node.replace("MERCH_", "M")
    else:
        node_colors.append("#aed6f1")
        node_sizes.append(150)

plt.figure(figsize=(14, 10))
pos = nx.spring_layout(G, seed=7, k=0.5)

nx.draw_networkx_edges(G, pos, edge_color="#dcdcdc", width=0.6)
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)
nx.draw_networkx_labels(G, pos, labels=labels, font_size=6)

legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label='Ring A (IP sharing)', markerfacecolor='#e74c3c', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='Ring B (merchant concentration)', markerfacecolor='#f39c12', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='Ring C (subtle)', markerfacecolor='#9b59b6', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='Shared merchants', markerfacecolor='#7f8c8d', markersize=12),
    plt.Line2D([0], [0], marker='o', color='w', label='Normal customers (sample)', markerfacecolor='#aed6f1', markersize=8),
]
plt.legend(handles=legend_elements, loc='upper left', fontsize=9)
plt.title("GNN input graph (ego-network view): ring customers, shared merchants,\nand a sample of normal customers who use the same merchants")
plt.axis("off")
plt.tight_layout()
plt.savefig("models/graph_gnn/gnn_graph_view.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"Nodes shown: {G.number_of_nodes()}, edges: {G.number_of_edges()}")
print("Saved to models/graph_gnn/gnn_graph_view.png")