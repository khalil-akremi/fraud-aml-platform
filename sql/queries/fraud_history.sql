SELECT has_fraud_history, COUNT(*) FROM (
    SELECT
        t.customer_id,
        CASE WHEN SUM(t.fraud_label) > 0 THEN 1 ELSE 0 END AS has_fraud_history
    FROM transactions t
    GROUP BY t.customer_id
) sub
GROUP BY has_fraud_history;