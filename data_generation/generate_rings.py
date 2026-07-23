"""
Extended laundering ring generator - 3 rings, 3 different concealment
strategies, so a GNN has genuinely different patterns to learn from
(not just one memorized group of IDs).
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import uuid

rng = np.random.default_rng(123)


def _random_ip(rng):
    return f"{rng.integers(1,255)}.{rng.integers(1,255)}.{rng.integers(1,255)}.{rng.integers(1,255)}"


def generate_ring_ip_sharing(customers, merchants, ring_id, ring_size=6, n_tx=15):
    """Ring A: shares an IP address some of the time. The 'sloppy' ring."""
    ring_customers = customers.sample(ring_size, random_state=200 + ring_id)
    ring_merchants = merchants.sample(2, random_state=200 + ring_id)
    shared_ip = f"203.0.113.{40 + ring_id}"

    rows = []
    for _, cust in ring_customers.iterrows():
        for i in range(n_tx):
            merch = ring_merchants.sample(1).iloc[0]
            use_shared_ip = rng.random() < 0.4
            rows.append({
                "transaction_id": str(uuid.uuid4()),
                "customer_id": cust["customer_id"],
                "merchant_id": merch["merchant_id"],
                "timestamp": datetime(2024, 1, 1) + timedelta(seconds=int(rng.integers(0, 365 * 24 * 3600))),
                "amount": round(float(rng.uniform(100, 800)), 2),
                "country": merch["country"],
                "channel": "online",
                "fraud_label": 0,
                "fraud_type": "none",
                "ip_address": shared_ip if use_shared_ip else _random_ip(rng),
            })
    return pd.DataFrame(rows), list(ring_customers["customer_id"])


def generate_ring_merchant_concentration(customers, merchants, ring_id, ring_size=6, n_tx=15):
    """Ring B: no shared IP at all, but heavily concentrated on 1-2 merchants.
    Meant to be invisible to a pure Louvain-on-merchant-edges approach,
    since normal customers also share merchants (just not this concentrated)."""
    ring_customers = customers.sample(ring_size, random_state=300 + ring_id)
    ring_merchants = merchants.sample(2, random_state=300 + ring_id)

    rows = []
    for _, cust in ring_customers.iterrows():
        for i in range(n_tx):
            merch = ring_merchants.sample(1).iloc[0]  # always from just these 2
            rows.append({
                "transaction_id": str(uuid.uuid4()),
                "customer_id": cust["customer_id"],
                "merchant_id": merch["merchant_id"],
                "timestamp": datetime(2024, 1, 1) + timedelta(seconds=int(rng.integers(0, 365 * 24 * 3600))),
                "amount": round(float(rng.uniform(100, 800)), 2),
                "country": merch["country"],
                "channel": "online",
                "fraud_label": 0,
                "fraud_type": "none",
                "ip_address": _random_ip(rng),  # always unique, no shared IP
            })
    return pd.DataFrame(rows), list(ring_customers["customer_id"])


def generate_ring_subtle(customers, merchants, ring_id, ring_size=6, n_tx=15):
    """Ring C: mild merchant concentration (not exclusive) + slightly
    elevated amounts + no shared IP. The subtlest case - meant to require
    combining structure AND features to catch."""
    ring_customers = customers.sample(ring_size, random_state=400 + ring_id)
    ring_merchants = merchants.sample(3, random_state=400 + ring_id)
    other_merchants = merchants.sample(20, random_state=401 + ring_id)

    rows = []
    for _, cust in ring_customers.iterrows():
        for i in range(n_tx):
            # 60% of transactions at the "core" ring merchants, 40% elsewhere (blend in)
            if rng.random() < 0.6:
                merch = ring_merchants.sample(1).iloc[0]
            else:
                merch = other_merchants.sample(1).iloc[0]
            rows.append({
                "transaction_id": str(uuid.uuid4()),
                "customer_id": cust["customer_id"],
                "merchant_id": merch["merchant_id"],
                "timestamp": datetime(2024, 1, 1) + timedelta(seconds=int(rng.integers(0, 365 * 24 * 3600))),
                "amount": round(float(rng.uniform(300, 1200)), 2),  # slightly elevated vs normal
                "country": merch["country"],
                "channel": "online",
                "fraud_label": 0,
                "fraud_type": "none",
                "ip_address": _random_ip(rng),
            })
    return pd.DataFrame(rows), list(ring_customers["customer_id"])


def generate_all_rings(customers, merchants):
    ring_a_tx, ring_a_ids = generate_ring_ip_sharing(customers, merchants, ring_id=1)
    ring_b_tx, ring_b_ids = generate_ring_merchant_concentration(customers, merchants, ring_id=2)
    ring_c_tx, ring_c_ids = generate_ring_subtle(customers, merchants, ring_id=3)

    all_ring_tx = pd.concat([ring_a_tx, ring_b_tx, ring_c_tx], ignore_index=True)

    ground_truth = pd.concat([
        pd.DataFrame({"customer_id": ring_a_ids, "ring_id": "A_ip_sharing"}),
        pd.DataFrame({"customer_id": ring_b_ids, "ring_id": "B_merchant_concentration"}),
        pd.DataFrame({"customer_id": ring_c_ids, "ring_id": "C_subtle"}),
    ], ignore_index=True)

    return all_ring_tx, ground_truth