# ADR 005: Single fraud classifier instead of one model per pattern

## Decision
Train one binary classifier (fraud vs. not-fraud) using the engineered
features (velocity, geo-change, rolling-sum) rather than a separate model
per fraud pattern. Structuring is conceptually AML, not card fraud, and
would be documented as a separate downstream module in a real system.

## Reasoning
Per-pattern models would be trained on extremely few positive examples
(e.g. only 20 account_takeover rows), which is not enough data to train
a reliable classifier. A single model can share signal across patterns
while still using pattern-specific features. In a real bank, fraud and
AML are separated by business response (block a transaction vs. file a
regulatory report), not because the underlying ML technique differs —
this project acknowledges that distinction without building two full
production pipelines for it.