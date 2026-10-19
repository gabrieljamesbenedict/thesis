# Interaction-ReID ID Maps

One file per video per run: `src/evaluators/id_maps/<video>_map.csv`

## Format

```
gt_label,tracker_id,type
1,7,human
3,2,human
1,13,object
1,21,object
6,4,object
12,9,object
```

## Columns

| Column    | Type   | Meaning                                        |
|-----------|--------|------------------------------------------------|
| gt_label  | str    | Stable label from GT summary (`human`/`object`) |
| tracker_id| int    | BoT-SORT track ID from this run's prediction   |
| type      | str    | `human` or `object`                            |

## Rules

- **Many-to-one allowed**: one GT label may map to multiple tracker IDs when
  the tracker splits/reassigns IDs mid-video.
- **One-to-many forbidden**: one tracker ID cannot map to two GT labels of
  the same type (validation error).
- **Per-run**: tracker IDs change run-to-run (GPU nondeterminism), so regenerate
  the map for each run.
- **Per-video**: tracker IDs are per-video (not global).
- **Unmapped prediction tracker IDs**: counted as FP, warning printed.
- **GT label with no mapped tracker ID**: warning; its rows become FN
  (legitimate — the interaction may genuinely be missed).

## Creating a map

1. Run the pipeline to get predictions.
2. Run `python -m evaluators.build_map` to auto-suggest a map by time-overlap +
   class similarity.
3. Manually review/edit the suggested map.
4. Run `evaluatorB_v2` with `--map-dir` pointing at this directory.