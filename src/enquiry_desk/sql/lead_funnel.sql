-- The lead funnel: how many leads got at least as far as each stage.
--
-- A lead can skip a stage (book a call before every detail is collected). Counting only the
-- leads that were recorded in a stage would then show more bookings than qualified leads.
-- So each lead is placed at the furthest stage it reached, and a stage counts every lead at
-- that stage or beyond it.
WITH stages (stage, step) AS (
    VALUES ('new', 1), ('qualified', 2), ('counselling_booked', 3)
),
furthest AS (
    SELECT lead_events.lead_id, MAX(stages.step) AS step
    FROM lead_events
    JOIN stages ON stages.stage = lead_events.stage
    GROUP BY lead_events.lead_id
),
counts AS (
    SELECT stages.stage, stages.step, COUNT(furthest.lead_id) AS leads
    FROM stages
    LEFT JOIN furthest ON furthest.step >= stages.step
    GROUP BY stages.stage, stages.step
)
SELECT stage,
       leads,
       ROUND(100.0 * leads / NULLIF(FIRST_VALUE(leads) OVER (ORDER BY step), 0), 1) AS pct_of_all_leads,
       ROUND(100.0 * leads / NULLIF(LAG(leads) OVER (ORDER BY step), 0), 1)         AS pct_of_stage_before
FROM counts
ORDER BY step;
