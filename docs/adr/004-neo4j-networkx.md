## Decision
Use NetworkX for graph construction and analysis, with Neo4j as an
optional stretch goal later.

## Reasoning
NetworkX is a Python library with no separate server to install, and it
plugs directly into PyTorch Geometric for training the GNN — no new
infrastructure to debug while the graph logic itself is still unproven.
Neo4j would only become worth the added setup once the graph needs to
scale beyond what fits in memory, or needs its built-in visualization
tools.