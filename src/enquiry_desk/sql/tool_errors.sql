-- Tools ranked by how often a call to them fails.
--
-- RANK leaves gaps after a tie (1, 1, 3). DENSE_RANK would not (1, 1, 2). ROW_NUMBER would
-- break the tie arbitrarily (1, 2, 3).
--
-- SUM(errors) OVER () has an empty window, which is every row: the grand total, repeated on
-- each row, so each tool's share of all errors can be computed without a second query.
WITH per_tool AS (
    SELECT name,
           COUNT(*)        AS calls,
           SUM(ok = 0)     AS errors,
           AVG(latency_ms) AS avg_ms
    FROM usage_events
    WHERE kind = 'tool'
    GROUP BY name
)
SELECT name,
       calls,
       errors,
       ROUND(100.0 * errors / calls, 1)                             AS error_rate_pct,
       RANK() OVER (ORDER BY 1.0 * errors / calls DESC)             AS rank_by_error_rate,
       ROUND(100.0 * errors / NULLIF(SUM(errors) OVER (), 0), 1)    AS share_of_all_errors_pct,
       ROUND(avg_ms, 1)                                             AS avg_ms
FROM per_tool
ORDER BY rank_by_error_rate, name;
