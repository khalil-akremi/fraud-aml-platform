import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

connection_string = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(connection_string)

with engine.connect() as conn:
    print("Connected successfully!")

# --- merchants ---
with engine.begin() as conn:
    conn.execute(text("TRUNCATE TABLE merchants;"))

merchants = pd.read_csv("data/raw/merchants.csv")
merchants.to_sql("merchants", engine, if_exists="append", index=False)
print(f"Loaded {len(merchants)} merchants.")

# --- customers ---
with engine.begin() as conn:
    conn.execute(text("TRUNCATE TABLE customers;"))

customers = pd.read_csv("data/raw/customers.csv")
customers["effective_start_date"] = pd.Timestamp.today().date()
customers["effective_end_date"] = None
customers["is_current"] = True
customers.to_sql("customers", engine, if_exists="append", index=False)
print(f"Loaded {len(customers)} customers.")

# --- transactions ---
with engine.begin() as conn:
    conn.execute(text("TRUNCATE TABLE transactions;"))

transactions = pd.read_csv("data/raw/transactions.csv")
transactions.to_sql("transactions", engine, if_exists="append", index=False)
print(f"Loaded {len(transactions)} transactions.")