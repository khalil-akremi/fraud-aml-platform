SELECT
    transaction_id,
    customer_id,
    timestamp,
    amount,
    fraud_type,
    COUNT(*) OVER (
        PARTITION BY customer_id
        ORDER BY timestamp
        RANGE BETWEEN INTERVAL '60 seconds' PRECEDING AND CURRENT ROW
    ) AS transactions_last_60_sec
FROM transactions
WHERE fraud_type = 'none'
ORDER BY customer_id, timestamp
LIMIT 20;