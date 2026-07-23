# Model Card: Graph Neural Network - Ring Member Detection (GraphSAGE)

## Purpose
Predicts whether a customer node is a money-laundering ring member,
using both graph structure AND behavioral features together - catching
patterns that pure structural methods (Louvain, Milestone 5) cannot see
on their own.

## Why a GNN, given Louvain already worked in Milestone 5
Milestone 5's Louvain approach only worked because that ring shared an
identifier (IP address) that created an unambiguous, isolated graph
structure. Two new ring designs were deliberately added to test the
actual boundary of what pure structure detection can do:
- Ring B: no shared IP, only merchant concentration
- Ring C: mild merchant concentration + slightly elevated amounts, no
  shared IP, deliberately blended with normal-looking transactions

A GNN combines each node's own behavioral features (total transactions,
average amount, merchant concentration, avg merchant risk score) with
message-passing across the graph structure - learning a pattern that
neither structure alone (Louvain) nor features alone (a plain
classifier with no graph awareness) could catch by themselves.

## Architecture
2-layer GraphSAGE (SAGEConv), 32 hidden channels, on a bipartite
customer-merchant graph (5,300 nodes: 5,000 customers + 300 merchants)
plus direct customer-customer edges from shared IPs. Node features are
z-score normalized; merchant and customer nodes share one unified
8-column feature schema (customer-specific fields zeroed for merchant
nodes and vice versa).

## Training
Binary node classification (is_ring_member), transductive setting
(train/val split via node masks on the same graph, not separate
graphs). BCEWithLogitsLoss with pos_weight for severe class imbalance
(18 ring customers out of 5,000, same principle as the XGBoost fraud
classifier's scale_pos_weight).

## Results
Validation set: 6 held-out ring members (2 per ring), 1,495 normal
customers.
- Recall: 100% (all 6 ring members correctly identified, including one
  from the deliberately subtle Ring C with no shared IP)
- Precision: 85.7% (1 false positive out of 7 flagged nodes)

## Key finding: the danger of 100% and how it was caught
An earlier training run on a simplified test dataset showed 100%
precision AND recall. This was investigated rather than accepted at
face value - the cause was a bug in the test data generator (a missing
`country` column caused every normal customer to show
distinct_countries=0, an artificially large and fake separating
signal). Once fixed and rerun on the real, full dataset, the more
believable 85.7%/100% result appeared instead. This mirrors a
recurring theme throughout this project: perfect metrics are a signal
to investigate, not celebrate, especially on synthetic data.

## Known limitation
Even after the bug fix, synthetic ring behavioral signatures remain
more cleanly separable than real-world laundering rings would be
(e.g. normal customers average 0.088 merchant concentration vs. 0.13-
0.37 for ring members - a clean, deliberately engineered gap). Real
rings would likely show a messier, more overlapping distribution,
and true recall/precision would likely be lower than what is measured
here. This ceiling is inherent to any synthetic dataset with
deliberately encoded ground truth, not specific to this GNN