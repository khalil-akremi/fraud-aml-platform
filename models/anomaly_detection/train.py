import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.ensemble import IsolationForest

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

connection_string = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(connection_string)

with open("sql/queries/model_training_features.sql", "r") as f:
    query = f.read()

df = pd.read_sql(query, engine)
df = df.sort_values("timestamp").reset_index(drop=True)

df["hour_of_day"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek

feature_columns = [
    "amount", "hour_of_day", "day_of_week",
    "transactions_last_60_sec", "amount_last_7_days", "merchant_risk_score"
]

X = df[feature_columns]

model = IsolationForest(contamination=0.01, random_state=42)
model.fit(X)

df["anomaly_score"] = model.decision_function(X)
df["is_anomaly"] = model.predict(X) == -1
comparison = df[["fraud_label", "is_anomaly"]].value_counts()
print(comparison)
false_positive_anomalies = df[(df["fraud_label"] == 0) & (df["is_anomaly"] == True)]
print(false_positive_anomalies.sort_values("anomaly_score").head(10)[
    ["customer_id", "amount", "hour_of_day", "transactions_last_60_sec", "amount_last_7_days", "merchant_risk_score", "anomaly_score"]
])