import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.model_selection import TimeSeriesSplit
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

with open("sql/queries/model_training_features.sql", "r") as f:
    query = f.read()

df = pd.read_sql(query, engine)
df = df.sort_values("timestamp").reset_index(drop=True)

df["hour_of_day"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek

country_changed_dummies = pd.get_dummies(df["country_changed"], prefix="country_changed", dummy_na=True)
country_dummies = pd.get_dummies(df["country"], prefix="country")
channel_dummies = pd.get_dummies(df["channel"], prefix="channel")
merchant_category_dummies = pd.get_dummies(df["merchant_category"], prefix="merchant_cat")
customer_risk_dummies = pd.get_dummies(df["customer_risk_level"], prefix="customer_risk")

df = pd.concat([
    df, country_changed_dummies, country_dummies, channel_dummies,
    merchant_category_dummies, customer_risk_dummies
], axis=1)

feature_columns = (
    ["amount", "hour_of_day", "day_of_week", "transactions_last_60_sec",
     "amount_last_7_days", "merchant_risk_score"]
    + list(country_changed_dummies.columns)
    + list(country_dummies.columns)
    + list(channel_dummies.columns)
    + list(merchant_category_dummies.columns)
    + list(customer_risk_dummies.columns)
)

X = df[feature_columns]
y = df["fraud_label"]

tscv = TimeSeriesSplit(n_splits=5)

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(eval_metric="logloss", scale_pos_weight=scale_pos_weight)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(f"--- Fold {fold} (weighted, scale_pos_weight={scale_pos_weight:.1f}) ---")
    print(classification_report(y_test, preds, digits=3))
