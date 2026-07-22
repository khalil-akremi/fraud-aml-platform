"""
Draws the shared-IP customer graph and saves it as a PNG.
Ring members (community size >= 3) are highlighted in red;
coincidental 2-node IP collisions are shown in gray for contrast.
"""

import pandas as pd
import networkx as nx
import community as community_louvain
from collections import Counter
import matplotlib.pyplot as plt

transactions = pd.read_csv("data/raw/transactions.csv")

ip_to_customers = transactions.groupby("ip_address")["customer_id"].unique()
shared_ips = ip_to_customers[ip_to_customers.apply(len) > 1]

G = nx.Graph()
for ip, customers in shared_ips.items():
    customers = list(customers)
    for i in range(len(customers)):
        for j in range(i + 1, len(customers)):
            G.add_edge(customers[i], customers[j])

partition = community_louvain.best_partition(G)
community_sizes = Counter(partition.values())

# color nodes: red if in a suspected ring (community size >= 3), gray otherwise
node_colors = []
for node in G.nodes():
    comm_size = community_sizes[partition[node]]
    node_colors.append("#e74c3c" if comm_size >= 3 else "#95a5a6")

plt.figure(figsize=(10, 7))
pos = nx.spring_layout(G, seed=42, k=0.8)  # seed for reproducible layout

nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1400)
nx.draw_networkx_edges(G, pos, edge_color="#bdc3c7", width=1.5)
nx.draw_networkx_labels(G, pos, font_size=8, font_color="white", font_weight="bold")

plt.title("Customer graph via shared IP addresses\n(red = suspected laundering ring, gray = coincidental IP overlap)")
plt.axis("off")
plt.tight_layout()
plt.savefig("models/graph_gnn/ring_detection_graph.png", dpi=150, bbox_inches="tight")
plt.close()

print("Saved graph visualization to models/graph_gnn/ring_detection_graph.png")
print(f"Total nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")