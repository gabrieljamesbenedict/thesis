"""Helper to suggest ID maps from GT + predictions.

Matches each GT (human, object) pair to predicted tracks by:
    1. temporal overlap (IoU > 0)
    2. matching object_class
    3. best overlap per human_id / object_id

Usage:
    from evaluators.build_map import suggest_map, write_map_template
    sugg = suggest_map(gt_df, pred_df)
    write_map_template("src/evaluators/id_maps/template.csv")
"""

from pathlib import Path
from typing import List, Tuple
import pandas as pd


def suggest_map(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    """Suggest ID map rows by time-overlap + class heuristic.

    For each GT (human, object) pair, find predicted tracks that:
      - overlap in time (IoU > 0)
      - have matching object_class (pred object_class == GT object_cls)
    Assign by max temporal overlap per (gt_label, track_id).

    Returns a DataFrame with columns: gt_label, tracker_id, type.
    """
    gt = gt_df.copy()
    pred = pred_df.copy()

    req_gt = ["frame_start", "frame_end", "human", "object", "object_cls"]
    req_pred = ["frame_start", "frame_end", "human_id", "object_id", "object_class"]
    for c in req_gt:
        if c not in gt.columns:
            raise ValueError(f"GT missing column: {c}")
    for c in req_pred:
        if c not in pred.columns:
            raise ValueError(f"Pred missing column: {c}")

    gt["object_cls"] = gt["object_cls"].astype(str).str.strip()
    pred["object_class"] = pred["object_class"].astype(str).str.strip()

    suggestions: List[Tuple[str, int, str]] = []

    for _, grow in gt.iterrows():
        h_label = str(grow["human"])
        o_label = str(grow["object"])
        cls = grow["object_cls"]
        gs, ge = int(grow["frame_start"]), int(grow["frame_end"])

        cand = pred[
            (pred["object_class"] == cls)
            & (pred["frame_end"] >= gs)
            & (pred["frame_start"] <= ge)
        ].copy()

        if cand.empty:
            continue

        def _tiou(s1: int, e1: int, s2: int, e2: int) -> float:
            inter = max(0, min(e1, e2) - max(s1, s2) + 1)
            union = (e1 - s1 + 1) + (e2 - s2 + 1) - inter
            return inter / union if union > 0 else 0.0

        cand["tiou"] = cand.apply(
            lambda r: _tiou(gs, ge, int(r["frame_start"]), int(r["frame_end"])),
            axis=1,
        )

        human_best = cand.groupby("human_id")["tiou"].sum().sort_values(ascending=False)
        obj_best = cand.groupby("object_id")["tiou"].sum().sort_values(ascending=False)

        if not human_best.empty:
            suggestions.append((h_label, int(human_best.index[0]), "human"))
        if not obj_best.empty:
            suggestions.append((o_label, int(object_best.index[0]), "object"))

    sugg_df = pd.DataFrame(suggestions, columns=["gt_label", "tracker_id", "type"])
    if not sugg_df.empty:
        sugg_df = sugg_df.drop_duplicates(subset=["gt_label", "tracker_id", "type"])
        sugg_df["gt_label"] = sugg_df["gt_label"].astype(str)
        sugg_df["tracker_id"] = sugg_df["tracker_id"].astype(int)
        sugg_df["type"] = sugg_df["type"].astype(str)
    return sugg_df


def write_map_template(path: Path | str) -> None:
    """Write an empty map template with header columns."""
    pd.DataFrame(columns=["gt_label", "tracker_id", "type"]).to_csv(Path(path), index=False)