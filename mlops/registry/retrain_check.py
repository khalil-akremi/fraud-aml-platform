from mlflow import MlflowClient

client = MlflowClient()
model_name = "fraud_detection_classifier"

# get the current champion's metrics
champion_version = client.get_model_version_by_alias(model_name, "champion")
champion_run = client.get_run(champion_version.run_id)
champion_recall = champion_run.data.metrics["recall_fraud"]

# get the new candidate's metrics (the run you just logged)
new_run_id = "<paste the Run ID from your latest training run>"
new_run = client.get_run(new_run_id)
new_recall = new_run.data.metrics["recall_fraud"]

print(f"Champion recall: {champion_recall:.3f}")
print(f"Candidate recall: {new_recall:.3f}")

if new_recall > champion_recall:
    # register the new run's model as a new version, then promote it
    new_version = mlflow.register_model(f"runs:/{new_run_id}/model", model_name)
    client.set_registered_model_alias(model_name, "champion", new_version.version)
    print(f"Promoted version {new_version.version} to champion.")
else:
    print("Candidate did not outperform champion. Not promoted.")