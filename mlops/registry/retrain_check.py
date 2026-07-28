"""
Compares a newly trained model's recall against the current champion's
recall, and automatically promotes it if it performs better - a simple,
measurable, defensible promotion rule instead of a manual judgment call.
"""

import mlflow
from mlflow import MlflowClient

client = MlflowClient()
model_name = "fraud_detection_classifier"

# get the current champion's metrics
champion_version = client.get_model_version_by_alias(model_name, "champion")
champion_run = client.get_run(champion_version.run_id)
champion_recall = champion_run.data.metrics["recall_fraud"]

# get the candidate's metrics (paste the run ID of the model you want to evaluate)
new_run_id = "4dd3c5b895b84b26ae32a7cb72fbca4c"
new_run = client.get_run(new_run_id)
new_recall = new_run.data.metrics["recall_fraud"]

print(f"Champion (version {champion_version.version}) recall: {champion_recall:.3f}")
print(f"Candidate recall: {new_recall:.3f}")

if new_recall > champion_recall:
    new_version = mlflow.register_model(f"runs:/{new_run_id}/model", model_name)
    client.set_registered_model_alias(model_name, "champion", new_version.version)
    print(f"Promoted version {new_version.version} to champion.")
else:
    print("Candidate did not outperform champion. Not promoted.")