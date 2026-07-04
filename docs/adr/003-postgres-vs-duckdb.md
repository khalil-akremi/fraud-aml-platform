## Decision
Use PostgreSQL as the primary database instead of DuckDB.

## Reasoning
DuckDB is a lightweight, file-based engine built for single-process
analytics — it isn't designed for multiple processes writing and reading
at the same time. This project has Airflow, Dataiku, and the API all
touching the same tables concurrently, which needs a real database server.
Postgres also supports the SQL features the project needs later
(stored procedures, triggers, concurrent transactions).