-- The :n slowest turns of each day. This is the "top N per group" pattern.
--
-- A window function cannot be used in WHERE, because WHERE runs before windows are computed.
-- So the rank is computed in one CTE and filtered in the query that reads it.
WITH turns AS (
    SELECT session_id,
           turn,
           date(MIN(ts) + :tz, 'unixepoch') AS day,
           SUM(latency_ms)                  AS turn_ms,
           SUM(kind = 'llm')                AS model_calls,
           SUM(kind = 'tool')               AS tool_calls
    FROM usage_events
    GROUP BY session_id, turn
),
ranked AS (
    SELECT *,
           DENSE_RANK() OVER (PARTITION BY day ORDER BY turn_ms DESC) AS place
    FROM turns
)
SELECT day, place, session_id, turn, turn_ms, model_calls, tool_calls
FROM ranked
WHERE place <= :n
ORDER BY day, place, session_id;
