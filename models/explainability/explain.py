import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt

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

# train on everything except the last fold, same as your validated approach
tscv = TimeSeriesSplit(n_splits=5)
train_idx, test_idx = list(tscv.split(X))[-1]
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
model = XGBClassifier(eval_metric="logloss", scale_pos_weight=scale_pos_weight)
model.fit(X_train, y_train)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

print(shap_values.shape)


shap.summary_plot(shap_values, X_test, show=False)
plt.savefig("models/explainability/global_summary.png", bbox_inches="tight")
plt.close()
print("Saved global summary plot.")
# pick one specific fraud case to explain
fraud_indices = X_test[y_test == 1].index
sample_idx = fraud_indices[0]
sample_position = X_test.index.get_loc(sample_idx)

shap.force_plot(
    explainer.expected_value, shap_values[sample_position], X_test.iloc[sample_position],
    matplotlib=True, show=False
)
plt.savefig("models/explainability/single_case_explanation.png", bbox_inches="tight")
plt.close()
print(f"Explained transaction_id: {df.loc[sample_idx, 'transaction_id']}")
def generate_narrative(shap_row, feature_row, top_n=3):
    contributions = list(zip(feature_row.index, shap_row, feature_row.values))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    top_features = contributions[:top_n]

    reasons = []
    for feature_name, shap_val, feature_val in top_features:
        direction = "increased" if shap_val > 0 else "decreased"
        reasons.append(f"{feature_name} (value={feature_val}) {direction} fraud risk")

    return "This transaction was flagged because: " + "; ".join(reasons) + "."

narrative = generate_narrative(shap_values[sample_position], X_test.iloc[sample_position])
print(narrative)