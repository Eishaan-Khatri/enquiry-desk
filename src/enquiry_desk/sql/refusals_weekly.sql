-- Week by week: how many caller turns there were, how many drafts the reply check sent back,
-- and how many turns ended with the fixed refusal line.
--
-- This is the desk's drift signal. Nothing in the code changes from one week to the next, so
-- a refusal rate that climbs means something outside changed: the model behind the API was
-- updated, the documents were edited, or callers began asking about something new.
--
-- Two sources with different grain are joined: usage_events has many rows per turn and
-- reply_checks has at most two. Each is rolled up to one row per week first, in its own CTE,
-- and only then joined. Joining the raw tables would multiply the rows and inflate both counts.
WITH turns AS (
    SELECT session_id, turn, MIN(ts) AS started
    FROM usage_events
    GROUP BY session_id, turn
),
weekly_turns AS (
    SELECT strftime('%Y-W%W', started + :tz, 'unixepoch') AS week,
           COUNT(*)                                       AS turns
    FROM turns
    GROUP BY week
),
weekly_checks AS (
    SELECT strftime('%Y-W%W', ts + :tz, 'unixepoch') AS week,
           SUM(outcome = 'retried')                  AS sent_back,
           SUM(outcome = 'refused')                  AS refused
    FROM reply_checks
    GROUP BY week
),
weekly AS (
    SELECT weekly_turns.week,
           weekly_turns.turns,
           COALESCE(weekly_checks.sent_back, 0)                                    AS sent_back,
           COALESCE(weekly_checks.refused, 0)                                      AS refused,
           ROUND(100.0 * COALESCE(weekly_checks.refused, 0) / weekly_turns.turns, 1) AS refused_pct
    FROM weekly_turns
    LEFT JOIN weekly_checks ON weekly_checks.week = weekly_turns.week
)
SELECT week,
       turns,
       sent_back,
       refused,
       refused_pct,
       ROUND(refused_pct - LAG(refused_pct) OVER (ORDER BY week), 1) AS change_from_week_before
FROM weekly
ORDER BY week;
