WITH transactions_with_geo AS (
    SELECT
        transaction_id,
        customer_id,
        timestamp,
        country,
        fraud_type,
        LAG(country) OVER (PARTITION BY customer_id ORDER BY timestamp) AS previous_country
    FROM transactions
)
SELECT
    transaction_id,
    customer_id,
    timestamp,
    country,
    previous_country,
    fraud_type,
    country != previous_country AS country_changed
FROM transactions_with_geo
WHERE fraud_type = 'account_takeover';