-- How long leads stay in each stage before they move on.
--
-- lead_events has one row per stage entered. LEAD looks at the next row for the same lead,
-- which is when the lead left this stage. A lead still in a stage has no next row, so LEAD
-- returns NULL: it is counted in `still_here` and left out of the average.
WITH spans AS (
    SELECT lead_id,
           stage,
           ts                                                     AS entered,
           LEAD(ts) OVER (PARTITION BY lead_id ORDER BY ts, id)   AS left_at
    FROM lead_events
)
SELECT stage,
       COUNT(*)                                       AS entered,
       SUM(left_at IS NULL)                           AS still_here,
       ROUND(AVG((left_at - entered) / 60.0), 2)      AS avg_minutes_before_moving_on
FROM spans
GROUP BY stage
ORDER BY CASE stage WHEN 'new' THEN 1 WHEN 'qualified' THEN 2 ELSE 3 END;
