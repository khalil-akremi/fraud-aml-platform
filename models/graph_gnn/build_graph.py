"""
Money-laundering ring detection via graph community detection.

Key design lesson (found through testing, not assumed upfront):
A graph built from customer->merchant AND customer->IP edges technically
works, but buries the ring inside a huge, useless community - because
merchants are commonly shared by thousands of unrelated customers, while
IP addresses are rare and meaningful. The fix is not a different algorithm,
it's choosing the right edges: build the graph directly from shared IPs
(customer-to-customer), which isolates the ring cleanly.
"""

import pandas as pd
import networkx as nx
import community as community_louvain  # pip install python-louvain
from collections import Counter

transactions = pd.read_csv("data/raw/transactions.csv")

# Step 1: find IPs used by more than one distinct customer.
# A legitimate customer's own IP(s) won't be shared with strangers -
# any IP shared by 2+ customers is inherently suspicious.
ip_to_customers = transactions.groupby("ip_address")["customer_id"].unique()
shared_ips = ip_to_customers[ip_to_customers.apply(len) > 1]

print(f"Found {len(shared_ips)} IP addresses shared by multiple customers.")

# Step 2: build a customer-to-customer graph - an edge exists between
# two customers if they were ever seen sharing the same IP address.
G = nx.Graph()
for ip, customers in shared_ips.items():
    customers = list(customers)
    for i in range(len(customers)):
        for j in range(i + 1, len(customers)):
            G.add_edge(customers[i], customers[j])

print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Step 3: run Louvain community detection to find tightly-connected clusters.
partition = community_louvain.best_partition(G)
community_sizes = Counter(partition.values())

# Step 4: flag any community of size >= 3 as a suspected ring.
# (2-node communities are more likely coincidental single IP collisions;
# 3+ customers repeatedly sharing an IP is a much stronger signal.)
suspected_rings = [
    community_id for community_id, size in community_sizes.items() if size >= 3
]

print(f"\nSuspected rings found: {len(suspected_rings)}")
for community_id in suspected_rings:
    members = [node for node, cid in partition.items() if cid == community_id]
    print(f"  Community {community_id}: {members}")