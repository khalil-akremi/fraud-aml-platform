# Model Card: Anomaly Detection (Isolation Forest)

## Purpose
Unsupervised safety net alongside the supervised fraud classifier.
Catches statistically unusual transactions without relying on known
fraud labels — intended to surface fraud patterns not present in
labeled training data.

## Approach
Isolation Forest, contamination=0.01 (roughly 3x the known ~0.3% fraud
rate, deliberately set above the known rate to allow the model to flag
behavior beyond labeled fraud types).

## Features used
amount, hour_of_day, day_of_week, transactions_last_60_sec,
amount_last_7_days, merchant_risk_score — a smaller, purely numeric/
behavioral subset. One-hot encoded categorical columns (country,
channel, merchant_category, customer_risk_level) were deliberately
excluded: unlike XGBoost, Isolation Forest has no supervised signal to
learn which features matter, so noisy dimensions dilute what counts as
"anomalous" rather than being ignored.

## Validation against known labels
93/237 (39%) of known fraud rows were flagged as anomalies. Expected to
underperform the supervised classifier on known fraud types, since this
model never saw fraud_label during training — it is not trying to learn
those specific patterns.

## Key finding
The top unlabeled "false positive" anomalies show a coherent, plausible
pattern: transactions in the $2,300-$2,950 range at unusual hours
(22:00-4:00) at moderately high-risk merchants — large enough and
risky enough to look suspicious, but not matching the account_takeover
rule (no country change) or the card_testing velocity signature. This
is exactly the kind of edge case unsupervised anomaly detection is
meant to surface that a supervised/rule-based approach would miss by
construction.

## Known limitation
Synthetic data only contains the 3 fraud types deliberately engineered
in this project, so the model's real value — catching genuinely unknown
fraud patterns — cannot be fully validated here. A synthetic dataset
cannot contain true "unknown unknowns" by construction.