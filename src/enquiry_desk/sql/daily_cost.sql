-- Cost per day, with a seven-day moving average, the change from the day before, and the
-- running total.
--
-- A day with no callers has no rows in usage_events. A moving average over "the last 7 rows"
-- would then silently reach back further than seven days. So a recursive CTE first builds
-- every calendar day between the first and the last, and the costs are joined onto it.
WITH RECURSIVE
spend AS (
    SELECT date(ts + :tz, 'unixepoch') AS day,
           COUNT(DISTINCT session_id)  AS sessions,
           SUM(cost_usd)               AS cost_usd
    FROM usage_events
    GROUP BY day
),
calendar (day) AS (
    SELECT MIN(day) FROM spend
    UNION ALL
    SELECT date(day, '+1 day') FROM calendar WHERE day < (SELECT MAX(day) FROM spend)
),
daily AS (
    SELECT calendar.day,
           COALESCE(spend.sessions, 0) AS sessions,
           COALESCE(spend.cost_usd, 0) AS cost_usd
    FROM calendar
    LEFT JOIN spend ON spend.day = calendar.day
)
SELECT day,
       sessions,
       cost_usd,
       AVG(cost_usd) OVER (ORDER BY day ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS avg_7_days,
       cost_usd - LAG(cost_usd) OVER (ORDER BY day)                               AS change_from_day_before,
       SUM(cost_usd) OVER (ORDER BY day)                                          AS total_so_far
FROM daily
WHERE day IS NOT NULL
ORDER BY day;
