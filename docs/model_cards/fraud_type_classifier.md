# Model Card: Fraud Type Classifier (Stage 2, rule-based)

## Purpose
Runs only on transactions already flagged as fraud by the Stage 1 XGBoost
classifier. Routes each flagged transaction to a fraud_type
(card_testing, account_takeover, structuring, unclassified) to support
different downstream business responses (block vs. regulatory report).

## Approach
Rule-based thresholds derived from known statistical signatures, not a
trained model — chosen because the 3 fraud types have too few labeled
examples (237 total, as few as 20 per class) to train a reliable
multi-class classifier.

## Rules
1. transactions_last_60_sec >= 3         -> card_testing
2. amount_last_7_days >= 25000 and amount < 10000  -> structuring
3. country_changed == True and amount >= 1500      -> account_takeover
4. else -> unclassified

## Validation results
- card_testing: 115/144 correctly classified (29 unclassified — below
  the velocity threshold, likely shorter bursts)
- account_takeover: 20/20 correctly classified
- structuring: 43/73 correctly classified, 15 misclassified as
  account_takeover

## Known limitation
The first transaction in a structuring sequence has no rolling history
yet, so amount_last_7_days cannot exceed the threshold at that point.
At that single moment, it is statistically indistinguishable from an
account_takeover transaction (large amount, new country) — this is a
genuine, unavoidable ambiguity given only the current transaction's
context, not a bug in the rule logic. Later transactions in the same
sequence are correctly classified once the rolling sum accumulates.

## Possible future improvement
A stateful/sequential model (or simply re-scoring earlier transactions
retroactively once a customer's structuring pattern becomes clear) could
resolve this, at the cost of not being able to flag it in true real-time
on the very first transaction.