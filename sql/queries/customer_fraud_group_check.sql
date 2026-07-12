SELECT
    CASE WHEN total_fraud_transactions = 0 THEN 'no_fraud' ELSE 'has_fraud' END AS group_label,
    COUNT(*) AS num_customers
FROM (
    SELECT customer_id, SUM(fraud_label) AS total_fraud_transactions
    FROM transactions
    GROUP BY customer_id
) sub
GROUP BY group_label;