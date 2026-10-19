# Interaction Ground Truth (GT) Schema & Annotation Guide

## File Location

`src/eval/gt_summary/<video>_true_summary.csv`

## Columns

| Column        | Type   | Description                                               |
|---------------|--------|-----------------------------------------------------------|
| frame_start   | int    | Inclusive start frame (0-indexed)                         |
| frame_end     | int    | Inclusive end frame                                       |
| human         | str/int| Stable human label per video (e.g. `1`, `2`, `3` or `A`)  |
| object        | str/int| Stable object instance label (e.g. `1`, `2` or `phone1`)  |
| object_cls    | str    | Canonical class name (see below)                          |

## Annotation Rules

- **Source**: Annotate from the RAW input video, never the produced output video.
- **`human`/`object`**: Stable labels per video. Use integers (`1`, `2`, `3`) or
  descriptive strings (`A`, `B`, `phone1`, `phone2`). Must be consistent within
  a video. These are NOT tracker IDs — they are stable across runs.
- **`object_cls`**: MUST be one of the canonical model class names:
  `cell phone`, `laptop`, `wallet`, `bottle`, `bag`, `cup`.
- **Intervals**: Raw interaction intervals. No stitching — the evaluator does
  not merge intervals; annotate raw contact intervals.
- **Negative control**: Empty file (header only) for videos with zero
  interactions (e.g. `ReachingNotTouching`).

## ID Map (Per Run)

`src/evaluators/id_maps/<video>_map.csv`

Maps GT labels to per-run tracker IDs. See `id_maps/README.md`.

## Cross-Run Consistency

Tracker IDs (human_id/object_id from BoT-SORT) are non-deterministic run-to-run
due to GPU nondeterminism. They must NOT be stored in GT files. The ID map is
the per-run bridge between stable GT labels and transient tracker IDs.

## Legacy Logs

Older interaction logs (pre-2026-10-18) may say `"phone"` instead of
`"cell phone"`. When comparing across runs, apply this mapping:
`phone` → `cell phone`.

## Class Names

| Model Class Index | Class Name    |
|-------------------|---------------|
| 0                 | cell phone    |
| 1                 | laptop        |
| 2                 | wallet        |
| 3                 | bottle        |
| 4                 | bag           |
| 5                 | cup           |