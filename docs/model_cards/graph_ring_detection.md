# Model Card: Money-Laundering Ring Detection (Graph/NetworkX)

## Purpose
Detects groups of customers secretly connected through shared identifiers
(IP addresses), simulating a real money-laundering ring - a pattern
invisible at the individual-transaction level, only detectable through
graph structure.

## Approach
NetworkX + Louvain community detection (python-louvain).

## Key finding: initial approach failed, and why
The first graph design used customer->merchant AND customer->IP edges
across the full transaction dataset. This technically placed all 5 ring
customers in the same community, but that community ballooned to 1,008
nodes - because ring customers also have normal, everyday transactions
at ordinary merchants, and those merchants are shared by thousands of
unrelated customers. The common, low-signal edge type (merchant sharing)
diluted the rare, high-signal edge type (IP sharing).

## Fix
Rebuilt the graph using only customer-to-customer edges, created when
two customers are seen sharing the same IP address (excluded from
normal customer->merchant edges entirely). This produced a much smaller,
cleaner graph (9 nodes, 12 edges out of ~80,000 transactions) where the
5 ring customers formed their own isolated community, correctly
separated from 2 unrelated coincidental IP collisions (2-customer
communities, filtered out via a size >= 3 threshold).

## Validation
100% precision and recall against the known ring ground truth (5/5
correctly identified, 0 false positives) on this synthetic dataset.

## Known limitation
This is validated against a single planted ring on synthetic data with
a deliberately clean signal (one clearly reused IP). Real laundering
rings may use more sophisticated evasion (rotating IPs, VPNs, shared
devices instead of shared IPs) that would require additional identifier
types (device fingerprints, phone numbers) not present in this dataset -
noted as a natural extension for the GNN stage (Milestone 6), which can
learn from graph structure more flexibly than fixed community detection
thresholds.