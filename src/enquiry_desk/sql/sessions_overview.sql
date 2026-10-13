-- One row per session: its size, its cost, and what happened last.
--
-- "The latest row of each group" is the other everyday use of ROW_NUMBER: number each
-- session's rows from newest to oldest and keep number 1. GROUP BY alone cannot do this,
-- because it can return MAX(ts) but not the other columns of the row that holds it.
WITH numbered AS (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY session_id ORDER BY ts DESC, id DESC) AS newest_first
    FROM usage_events
),
totals AS (
    SELECT session_id,
           MAX(turn)          AS turns,
           SUM(kind = 'llm')  AS model_calls,
           SUM(kind = 'tool') AS tool_calls,
           SUM(ok = 0)        AS errors,
           SUM(cost_usd)      AS cost_usd
    FROM usage_events
    GROUP BY session_id
)
SELECT totals.session_id,
       totals.turns,
       totals.model_calls,
       totals.tool_calls,
       totals.errors,
       totals.cost_usd,
       datetime(numbered.ts + :tz, 'unixepoch') AS last_activity,
       numbered.kind                            AS last_event,
       numbered.name                            AS last_event_name
FROM totals
JOIN numbered ON numbered.session_id = totals.session_id AND numbered.newest_first = 1
ORDER BY numbered.ts DESC;
