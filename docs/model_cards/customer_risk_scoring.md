# Model Card: Customer Risk Scoring

## Purpose
Predicts whether a customer is likely to have a history of fraudulent
activity, based on aggregated transaction behavior — an entity-level,
longitudinal risk view, distinct from the transaction-level fraud
classifier (which scores individual events, not customers as a whole).

## Target
Binary: has_fraud_history (1 if the customer has ever had at least one
transaction with fraud_label=1, else 0). See ADR 006 for why this is
binary rather than a 3-tier high/medium/low label.

## Why not use the original customers.risk_level label
That label is deterministically computed from is_pep, residence_country,
and kyc_status. Training a model on those same fields to predict it
would be leakage — the model would just be reverse-engineering a known
formula, not learning anything.

## Features used
total_transactions, avg_transaction_amount, distinct_countries_used,
avg_merchant_risk_score — all aggregated from transaction history via
SQL (COUNT, AVG, COUNT DISTINCT, GROUP BY). total_fraud_transactions
was deliberately excluded as a feature since it is what the label was
directly derived from — including it would reintroduce leakage at this
level.

## Validation
80/20 stratified random split (not time-based — each row is already a
full historical summary per customer, so there is no time-leakage risk
the way there is for transaction-level models). XGBoost with
scale_pos_weight to address the 50/4950 (1%) class imbalance.

Precision: 0.778, Recall: 0.700 on the positive class (10 test examples).

## Known limitation
Only 50 customers in the entire dataset have any fraud history, giving
just 10 positive examples in the test set. Precision/recall figures
have wide uncertainty at this sample size — missing or catching even
1-2 additional customers would shift these numbers by 10+ percentage
points. These results should be read as directionally positive, not as
precise, stable performance metrics.