import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

connection_string = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(connection_string)

with open("sql/queries/customer_risk_features.sql", "r") as f:
    query = f.read()

df = pd.read_sql(query, engine)

feature_columns = [
    "total_transactions", "avg_transaction_amount",
    "distinct_countries_used", "avg_merchant_risk_score"
]

X = df[feature_columns]
y = df["has_fraud_history"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model = XGBClassifier(eval_metric="logloss", scale_pos_weight=scale_pos_weight)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print(classification_report(y_test, preds, digits=3))