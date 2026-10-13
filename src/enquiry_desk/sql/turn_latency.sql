-- Median and 95th percentile turn time for each day.
--
-- A turn is one caller message and everything the desk did to answer it, so its time is the
-- sum of its model calls and tool calls.
--
-- SQLite has no PERCENTILE_CONT. The percentile is taken by rank instead: sort the day's turns,
-- number them, and pick the row at position ceil(p * n). This is the nearest-rank method.
-- Integer division does the ceil: ceil(0.5 * n) = (n + 1) / 2, ceil(0.95 * n) = (95n + 99) / 100.
WITH turns AS (
    SELECT session_id,
           turn,
           date(MIN(ts) + :tz, 'unixepoch') AS day,
           SUM(latency_ms)                  AS turn_ms
    FROM usage_events
    GROUP BY session_id, turn
),
ranked AS (
    SELECT day,
           turn_ms,
           ROW_NUMBER() OVER (PARTITION BY day ORDER BY turn_ms) AS position,
           COUNT(*)     OVER (PARTITION BY day)                  AS turns_in_day
    FROM turns
)
SELECT day,
       MAX(turns_in_day)                                                          AS turns,
       MAX(CASE WHEN position = (turns_in_day + 1) / 2 THEN turn_ms END)          AS p50_ms,
       MAX(CASE WHEN position = (95 * turns_in_day + 99) / 100 THEN turn_ms END)  AS p95_ms,
       MAX(turn_ms)                                                               AS slowest_ms
FROM ranked
GROUP BY day
ORDER BY day;
