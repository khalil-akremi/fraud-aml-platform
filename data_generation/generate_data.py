import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import uuid

from generate_rings import generate_all_rings

rng = np.random.default_rng(42)  # fixed seed = reproducible results

COUNTRIES = ["US", "GB", "FR", "DE", "AE", "NG", "RU", "CN", "BR", "TN"]
HIGH_RISK_COUNTRIES = ["NG", "RU", "AE"]


def generate_customers(n_customers: int) -> pd.DataFrame:
    customer_ids = [f"CUST_{i:07d}" for i in range(n_customers)]
    countries = rng.choice(COUNTRIES, size=n_customers)
    is_pep = rng.random(n_customers) < 0.003
    kyc_status = rng.choice(["verified", "pending", "failed"], size=n_customers, p=[0.92, 0.06, 0.02])
    income = rng.lognormal(mean=10.5, sigma=0.6, size=n_customers).round(2)
    age = rng.integers(18, 85, size=n_customers)

    df = pd.DataFrame({
        "customer_id": customer_ids,
        "age": age,
        "income": income,
        "nationality": countries,
        "residence_country": countries,
        "is_pep": is_pep,
        "kyc_status": kyc_status,
    })

    def risk_row(r):
        score = 0
        if r["is_pep"]:
            score += 2
        if r["residence_country"] in HIGH_RISK_COUNTRIES:
            score += 1
        if r["kyc_status"] != "verified":
            score += 1
        return "high" if score >= 2 else ("medium" if score == 1 else "low")

    df["risk_level"] = df.apply(risk_row, axis=1)
    return df


MERCHANT_CATEGORIES = ["grocery", "electronics", "travel", "gambling", "crypto", "jewelry", "utilities"]


def generate_merchants(n_merchants: int) -> pd.DataFrame:
    merchant_ids = [f"MERCH_{i:06d}" for i in range(n_merchants)]
    categories = rng.choice(MERCHANT_CATEGORIES, size=n_merchants)
    countries = rng.choice(COUNTRIES, size=n_merchants)
    CATEGORY_TICKET_MEAN = {
        "jewelry": 5.5,
        "travel": 5.5,
        "crypto": 4.5,
        "gambling": 4.5,
        "electronics": 4.5,
        "grocery": 2.5,
        "utilities": 2.5,
    }

    base_risk = np.select(
        [np.isin(categories, ["gambling", "crypto"]), categories == "jewelry"],
        [0.7, 0.4],
        default=0.1,
    )
    risk_score = np.clip(base_risk + rng.normal(0, 0.1, n_merchants), 0, 1).round(3)
    category_means = pd.Series(categories).map(CATEGORY_TICKET_MEAN)
    avg_ticket = rng.lognormal(mean=category_means, sigma=0.8, size=n_merchants).round(2)

    return pd.DataFrame({
        "merchant_id": merchant_ids,
        "category": categories,
        "country": countries,
        "risk_score": risk_score,
        "avg_ticket_size": avg_ticket,
    })


CHANNELS = ["online", "pos", "atm", "mobile"]


def _random_ip():
    return f"{rng.integers(1,255)}.{rng.integers(1,255)}.{rng.integers(1,255)}.{rng.integers(1,255)}"


def generate_normal_transactions(customers, merchants, n):
    cust_sample = customers.sample(n, replace=True, random_state=1)
    merch_sample = merchants.sample(n, replace=True, random_state=2)

    amounts = np.abs(rng.normal(
        loc=merch_sample["avg_ticket_size"].values,
        scale=merch_sample["avg_ticket_size"].values * 0.3 + 1,
    )).round(2)

    start = datetime(2024, 1, 1)
    timestamps = [start + timedelta(seconds=int(s)) for s in rng.integers(0, 365 * 24 * 3600, size=n)]

    ip_addresses = [
        f"{a}.{b}.{c}.{d}" for a, b, c, d in zip(
            rng.integers(1, 255, size=n),
            rng.integers(1, 255, size=n),
            rng.integers(1, 255, size=n),
            rng.integers(1, 255, size=n),
        )
    ]

    return pd.DataFrame({
        "transaction_id": [str(uuid.uuid4()) for _ in range(n)],
        "customer_id": cust_sample["customer_id"].values,
        "merchant_id": merch_sample["merchant_id"].values,
        "timestamp": timestamps,
        "amount": amounts,
        "country": merch_sample["country"].values,
        "channel": rng.choice(CHANNELS, size=n),
        "fraud_label": 0,
        "fraud_type": "none",
        "ip_address": ip_addresses,
    })


def generate_card_testing_fraud(customers, merchants, n_incidents):
    rows = []
    for _ in range(n_incidents):
        cust = customers.sample(1).iloc[0]
        merch = merchants.sample(1).iloc[0]
        burst_size = rng.integers(5, 16)
        start_time = datetime(2024, 1, 1) + timedelta(seconds=int(rng.integers(0, 365 * 24 * 3600)))
        for i in range(burst_size):
            rows.append({
                "transaction_id": str(uuid.uuid4()),
                "customer_id": cust["customer_id"],
                "merchant_id": merch["merchant_id"],
                "timestamp": start_time + timedelta(seconds=int(rng.integers(1, 8)) * i),
                "amount": round(rng.uniform(0.5, 3.0), 2),
                "country": merch["country"],
                "channel": "online",
                "fraud_label": 1,
                "fraud_type": "card_testing",
                "ip_address": _random_ip(),
            })
    return pd.DataFrame(rows)


def generate_account_takeover_fraud(customers, merchants, n_incidents):
    rows = []
    for _ in range(n_incidents):
        cust = customers.sample(1).iloc[0]
        merch = merchants[merchants["country"] != cust["residence_country"]].sample(1).iloc[0]
        rows.append({
            "transaction_id": str(uuid.uuid4()),
            "customer_id": cust["customer_id"],
            "merchant_id": merch["merchant_id"],
            "timestamp": datetime(2024, 1, 1) + timedelta(seconds=int(rng.integers(0, 365 * 24 * 3600))),
            "amount": round(float(rng.uniform(2000, 15000)), 2),
            "country": merch["country"],
            "channel": "online",
            "fraud_label": 1,
            "fraud_type": "account_takeover",
            "ip_address": _random_ip(),
        })
    return pd.DataFrame(rows)


def generate_structuring_fraud(customers, merchants, n_incidents):
    rows = []
    for _ in range(n_incidents):
        cust = customers.sample(1).iloc[0]
        merch = merchants.sample(1).iloc[0]
        n_parts = rng.integers(3, 7)
        base_day = datetime(2024, 1, 1) + timedelta(days=int(rng.integers(0, 360)))
        for i in range(n_parts):
            rows.append({
                "transaction_id": str(uuid.uuid4()),
                "customer_id": cust["customer_id"],
                "merchant_id": merch["merchant_id"],
                "timestamp": base_day + timedelta(days=i, hours=int(rng.integers(0, 24))),
                "amount": round(float(rng.uniform(9000, 9999)), 2),
                "country": merch["country"],
                "channel": "pos",
                "fraud_label": 1,
                "fraud_type": "structuring",
                "ip_address": _random_ip(),
            })
    return pd.DataFrame(rows)


def generate_dataset(n_customers=5000, n_merchants=300, n_normal_tx=80000):
    customers = generate_customers(n_customers)
    merchants = generate_merchants(n_merchants)

    normal = generate_normal_transactions(customers, merchants, n_normal_tx)
    card_testing = generate_card_testing_fraud(customers, merchants, n_incidents=15)
    account_takeover = generate_account_takeover_fraud(customers, merchants, n_incidents=20)
    structuring = generate_structuring_fraud(customers, merchants, n_incidents=15)
    ring_transactions, ring_ground_truth = generate_all_rings(customers, merchants)

    transactions = pd.concat(
        [normal, card_testing, account_takeover, structuring, ring_transactions],
        ignore_index=True,
    )
    transactions = transactions.sample(frac=1, random_state=7).reset_index(drop=True)

    return customers, merchants, transactions, ring_ground_truth


if __name__ == "__main__":
    customers, merchants, transactions, ring_ground_truth = generate_dataset()

    print(f"Customers:   {len(customers):,}")
    print(f"Merchants:   {len(merchants):,}")
    print(f"Transactions:{len(transactions):,}")
    print(f"Fraud rate:  {transactions['fraud_label'].mean():.4%}")
    print(transactions["fraud_type"].value_counts())
    print(f"\nRing ground truth ({len(ring_ground_truth)} customers across {ring_ground_truth['ring_id'].nunique()} rings):")
    print(ring_ground_truth.groupby("ring_id").size())

    customers.to_csv("data/raw/customers.csv", index=False)
    merchants.to_csv("data/raw/merchants.csv", index=False)
    transactions.to_csv("data/raw/transactions.csv", index=False)

    # save ring ground truth separately for later graph validation
    ring_ground_truth.to_csv("data/raw/ring_ground_truth.csv", index=False)
    
    