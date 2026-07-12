# ADR 006: Binary customer risk target, not 3-tier

## Decision
Predict a binary customer-level target (has_fraud_history: yes/no),
built from aggregated transaction behavior, rather than a 3-tier
high/medium/low label.

## Reasoning
The original customers.risk_level label is directly computed from the
same fields (is_pep, residence_country, kyc_status) that would be used
as model inputs — training on it would be leakage, not real prediction.
A behavior-based alternative (fraud history per customer) only yields
50 positive customers out of 5,000 (1%). Splitting that further into
3 tiers would leave some tiers with too few examples to train reliably,
the same data-scarcity problem identified in ADR 005. A binary target
keeps the label grounded in actual fraud history rather than a
subjective threshold, while behavioral features (avg merchant risk,
transaction count, distinct countries) still let the model learn
nuanced risk signal.