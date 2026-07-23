"""
Builds a bipartite customer-merchant graph (both node types in one graph)
plus customer-customer edges from shared IPs, with unified node features,
ready for GNN training.

Design choice: instead of projecting a customer-to-customer graph by
manually connecting every pair that shares a merchant (which explodes
combinatorially - a popular merchant with 1000 customers would need
~500,000 pairwise edges), we keep customers AND merchants as nodes in
one graph, with simple customer->merchant edges. The GNN's message
passing naturally lets information flow customer -> merchant -> other
customers who use that merchant, without ever materializing those
pairwise edges directly. This is both cheaper and closer to how
real-world bipartite transaction graphs are modeled.
"""

import pandas as pd
import numpy as np
import torch


def build_graph_data(transactions_path, merchants_path, ground_truth_path):
    transactions = pd.read_csv(transactions_path)
    merchants = pd.read_csv(merchants_path)
    ground_truth = pd.read_csv(ground_truth_path)

    tx_merch = transactions.merge(merchants[["merchant_id", "risk_score", "avg_ticket_size"]],
                                    on="merchant_id", how="left")

    # --- customer behavioral features ---
    cust_agg = tx_merch.groupby("customer_id").agg(
        total_transactions=("transaction_id", "count"),
        avg_amount=("amount", "mean"),
        distinct_merchants=("merchant_id", "nunique"),
        distinct_countries=("country", "nunique"),
        avg_merchant_risk_score=("risk_score", "mean"),
    ).reset_index()

    def max_concentration(group):
        return group["merchant_id"].value_counts(normalize=True).max()

    concentration = tx_merch.groupby("customer_id").apply(max_concentration).reset_index()
    concentration.columns = ["customer_id", "max_merchant_concentration"]
    cust_features = cust_agg.merge(concentration, on="customer_id")

    # --- node index mapping: customers first, then merchants ---
    customer_ids = cust_features["customer_id"].tolist()
    merchant_ids = merchants["merchant_id"].tolist()

    node_index = {cid: i for i, cid in enumerate(customer_ids)}
    offset = len(customer_ids)
    node_index.update({mid: offset + i for i, mid in enumerate(merchant_ids)})

    n_nodes = len(customer_ids) + len(merchant_ids)

    # --- unified feature matrix ---
    # columns: [is_customer, total_transactions, avg_amount, distinct_merchants,
    #           distinct_countries, risk_score, max_concentration, avg_ticket_size]
    X = np.zeros((n_nodes, 8), dtype=np.float32)

    cust_features_indexed = cust_features.set_index("customer_id")
    for cid, idx in node_index.items():
        if cid in cust_features_indexed.index:
            row = cust_features_indexed.loc[cid]
            X[idx] = [
                1, row["total_transactions"], row["avg_amount"], row["distinct_merchants"],
                row["distinct_countries"], row["avg_merchant_risk_score"],
                row["max_merchant_concentration"], 0,
            ]

    merchants_indexed = merchants.set_index("merchant_id")
    for mid in merchant_ids:
        idx = node_index[mid]
        row = merchants_indexed.loc[mid]
        X[idx] = [0, 0, 0, 0, 0, row["risk_score"], 0, row["avg_ticket_size"]]

    # normalize each feature column (z-score) - important for GNN training stability
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds == 0] = 1
    X_norm = (X - means) / stds

    # --- edges: customer -> merchant (deduplicated) ---
    cust_merch_pairs = tx_merch[["customer_id", "merchant_id"]].drop_duplicates()
    edge_list = []
    for _, row in cust_merch_pairs.iterrows():
        c_idx = node_index[row["customer_id"]]
        m_idx = node_index[row["merchant_id"]]
        edge_list.append((c_idx, m_idx))
        edge_list.append((m_idx, c_idx))  # undirected

    # --- edges: customer -> customer, from shared IPs (rare, cheap) ---
    ip_groups = transactions.groupby("ip_address")["customer_id"].unique()
    shared_ips = ip_groups[ip_groups.apply(len) > 1]
    for ip, customers_sharing in shared_ips.items():
        customers_sharing = list(customers_sharing)
        for i in range(len(customers_sharing)):
            for j in range(i + 1, len(customers_sharing)):
                c1, c2 = node_index[customers_sharing[i]], node_index[customers_sharing[j]]
                edge_list.append((c1, c2))
                edge_list.append((c2, c1))

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # --- labels: 1 for known ring customers, 0 otherwise (merchants excluded via mask) ---
    y = torch.zeros(n_nodes, dtype=torch.float)
    ring_customer_ids = set(ground_truth["customer_id"])
    for cid in ring_customer_ids:
        y[node_index[cid]] = 1.0

    # mask: only customer nodes participate in training/evaluation
    customer_mask = torch.zeros(n_nodes, dtype=torch.bool)
    customer_mask[:len(customer_ids)] = True

    return {
        "x": torch.tensor(X_norm, dtype=torch.float),
        "edge_index": edge_index,
        "y": y,
        "customer_mask": customer_mask,
        "node_index": node_index,
        "customer_ids": customer_ids,
    }


if __name__ == "__main__":
    data = build_graph_data(
        "data/raw/transactions.csv", "data/raw/merchants.csv", "data/raw/ring_ground_truth.csv"
    )
    print("Nodes:", data["x"].shape[0], "Features per node:", data["x"].shape[1])
    print("Edges:", data["edge_index"].shape[1])
    print("Ring customers (positive labels):", int(data["y"].sum()))