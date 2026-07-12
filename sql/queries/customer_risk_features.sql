SELECT
    t.customer_id,
    COUNT(*) AS total_transactions,
    SUM(t.fraud_label) AS total_fraud_transactions,
    AVG(t.amount) AS avg_transaction_amount,
    COUNT(DISTINCT t.country) AS distinct_countries_used,
    AVG(m.risk_score) AS avg_merchant_risk_score,
    CASE WHEN SUM(t.fraud_label) > 0 THEN 1 ELSE 0 END AS has_fraud_history
FROM transactions t
JOIN merchants m ON t.merchant_id = m.merchant_id
GROUP BY t.customer_id;

