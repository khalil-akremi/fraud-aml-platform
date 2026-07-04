## Decision
Build the full batch pipeline first; add Kafka streaming later as a separate layer.

## Reasoning
Building streaming and the fraud model at the same time makes it hard to tell
whether a failure comes from the model logic or from the streaming
infrastructure. Validating the model on batch data first isolates that risk —
once it works, streaming becomes an orchestration change, not a rebuild.