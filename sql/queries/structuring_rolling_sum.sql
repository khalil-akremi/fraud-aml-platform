WITH transactions_with_rolling_sum AS (
    SELECT
        transaction_id,
        customer_id,
        timestamp,
        amount,
        fraud_type,
        SUM(amount) OVER (
            PARTITION BY customer_id
            ORDER BY timestamp
            RANGE BETWEEN INTERVAL '7 days' PRECEDING AND CURRENT ROW
        ) AS amount_last_7_days
    FROM transactions
)
SELECT *
FROM transactions_with_rolling_sum
WHERE fraud_type = 'structuring'
ORDER BY customer_id, timestamp;