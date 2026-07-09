SELECT
    t.transaction_id,
    t.customer_id,
    t.timestamp,
    t.amount,
    t.country,
    t.channel,
    t.fraud_label,
    t.fraud_type,
    COUNT(*) OVER (
        PARTITION BY t.customer_id
        ORDER BY t.timestamp
        RANGE BETWEEN INTERVAL '60 seconds' PRECEDING AND CURRENT ROW
    ) AS transactions_last_60_sec,
    t.country != LAG(t.country) OVER (
        PARTITION BY t.customer_id ORDER BY t.timestamp
    ) AS country_changed,
    SUM(t.amount) OVER (
        PARTITION BY t.customer_id
        ORDER BY t.timestamp
        RANGE BETWEEN INTERVAL '7 days' PRECEDING AND CURRENT ROW
    ) AS amount_last_7_days,
    m.risk_score AS merchant_risk_score,
    m.category AS merchant_category,
    c.risk_level AS customer_risk_level
FROM transactions t
JOIN merchants m ON t.merchant_id = m.merchant_id
JOIN customers c ON t.customer_id = c.customer_id AND c.is_current = true
ORDER BY t.customer_id, t.timestamp;