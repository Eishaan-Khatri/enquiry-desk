-- Voice response time per day, stage by stage.
--
-- first_audio_ms is the gap the caller sits through: from the end of their last word to the
-- start of the desk's voice. The other columns say where that time went.
--
-- The same percentile-by-rank pattern as turn_latency.sql, applied to one column.
WITH turns AS (
    SELECT date(ts + :tz, 'unixepoch') AS day,
           stt_ms,
           first_sentence_ms,
           first_audio_ms,
           interrupted
    FROM voice_turns
    WHERE first_audio_ms IS NOT NULL
),
ranked AS (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY day ORDER BY first_audio_ms) AS position,
           COUNT(*)     OVER (PARTITION BY day)                         AS turns_in_day
    FROM turns
)
SELECT day,
       MAX(turns_in_day)                                                                AS turns,
       ROUND(AVG(stt_ms))                                                               AS avg_speech_to_text_ms,
       -- From the transcript being ready to the desk's first sentence: the model, and any tools.
       ROUND(AVG(first_sentence_ms - stt_ms))                                           AS avg_first_sentence_ms,
       -- From the first sentence to its audio: text to speech.
       ROUND(AVG(first_audio_ms - first_sentence_ms))                                   AS avg_text_to_speech_ms,
       ROUND(MAX(CASE WHEN position = (turns_in_day + 1) / 2 THEN first_audio_ms END))          AS p50_first_audio_ms,
       ROUND(MAX(CASE WHEN position = (95 * turns_in_day + 99) / 100 THEN first_audio_ms END))  AS p95_first_audio_ms,
       SUM(interrupted)                                                                 AS interrupted_turns
FROM ranked
GROUP BY day
ORDER BY day;
