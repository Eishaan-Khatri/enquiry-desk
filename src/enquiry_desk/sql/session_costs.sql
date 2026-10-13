-- Each turn of each session, with running totals.
--
-- The token column shows why long conversations cost more per turn: the whole history is sent
-- again with every model call, so prompt tokens grow turn by turn.
--
-- Two ways to write a running total are used on purpose:
--   running_cost_usd  names its frame: every row from the start of the session to this one.
--   running_tokens    leaves the frame out. With ORDER BY, the default frame is RANGE BETWEEN
--                     UNBOUNDED PRECEDING AND CURRENT ROW, which gives the same answer here
--                     because (session_id, turn) has no ties. With ties, RANGE would add all
--                     tied rows at once and ROWS would not.
WITH turns AS (
    SELECT session_id,
           turn,
           MIN(ts)                                                       AS started,
           MAX(ts)                                                       AS ended,
           SUM(COALESCE(prompt_tokens, 0) + COALESCE(completion_tokens, 0)) AS tokens,
           SUM(cost_usd)                                                 AS cost_usd
    FROM usage_events
    GROUP BY session_id, turn
)
SELECT session_id,
       turn,
       tokens,
       cost_usd,
       SUM(cost_usd) OVER (
           PARTITION BY session_id ORDER BY turn
           ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
       )                                                                       AS running_cost_usd,
       SUM(tokens) OVER (PARTITION BY session_id ORDER BY turn)                AS running_tokens,
       -- How long the caller took to reply: this turn's start minus the previous turn's end.
       ROUND(started - LAG(ended) OVER (PARTITION BY session_id ORDER BY turn), 1) AS seconds_since_last_turn,
       -- An empty OVER (PARTITION BY ...) with no ORDER BY is the total for the whole session.
       ROUND(100.0 * cost_usd / SUM(cost_usd) OVER (PARTITION BY session_id), 1)   AS share_of_session_pct
FROM turns
ORDER BY session_id, turn;
