"""ID map loader and validator for interaction-ReID evaluation.

Maps GT labels (human/object labels from ground truth) to per-run tracker IDs.
Allows many-to-one mapping (one GT label -> multiple tracker IDs when tracker
splits/reassigns IDs mid-video), but forbids one-to-many (one tracker ID mapped
to two GT labels of the same type).

Usage:
    from evaluators.id_map import load_id_map, validate_id_map, apply_id_map
    mdf = load_id_map("src/evaluators/id_maps/vid01_map.csv")
    warns, errs = validate_id_map(mdf, pred_df)
    pdf = apply_id_map(pred_df, mdf)
"""

from pathlib import Path
from typing import List, Tuple
import pandas as pd


REQUIRED_COLS = ["gt_label", "tracker_id", "type"]
VALID_TYPES = {"human", "object"}


def load_id_map(map_path: Path | str) -> pd.DataFrame:
    """Load an ID map CSV and return a validated DataFrame.

    Columns: gt_label (str), tracker_id (int), type ('human'|'object').
    Duplicate identical rows are dropped.
    """
    df = pd.read_csv(Path(map_path))
    missing = set(REQUIRED_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"ID map missing columns: {sorted(missing)}")

    df["gt_label"] = df["gt_label"].astype(str).str.strip()
    df["tracker_id"] = df["tracker_id"].astype(int)
    df["type"] = df["type"].str.lower().str.strip()

    bad_types = set(df["type"].unique()) - VALID_TYPES
    if bad_types:
        raise ValueError(f"ID map has invalid type values: {sorted(bad_types)}")

    df = df.drop_duplicates(subset=REQUIRED_COLS).reset_index(drop=True)
    return df


def validate_id_map(map_df: pd.DataFrame, pred_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Validate the ID map against predictions.

    Returns (warnings, errors).

    Checks:
        - tracker_id mapped to multiple gt_labels of same type -> ERROR
        - tracker_id used for both human and object -> ERROR
        - prediction tracker IDs not in map -> WARNING (counted as FP)
        - duplicate identical rows -> silently deduped in loader
    """
    warnings: List[str] = []
    errors: List[str] = []

    # 1. One tracker_id -> multiple gt_labels of same type = error
    dup = map_df.groupby(["tracker_id", "type"])["gt_label"].nunique()
    bad_dup = dup[dup > 1]
    for (tid, typ), n in bad_dup.items():
        labels = map_df[
            (map_df["tracker_id"] == tid) & (map_df["type"] == typ)
        ]["gt_label"].tolist()
        errors.append(
            f"tracker_id {tid} (type={typ}) maps to {n} gt_labels: {labels}"
        )

    # 2. Same tracker_id used for both human and object = error
    cross = map_df.groupby("tracker_id")["type"].nunique()
    cross_bad = cross[cross > 1]
    for tid in cross_bad.index:
        errors.append(f"tracker_id {tid} used for both human and object")

    # 3. Unmapped prediction tracker IDs -> warning (counted as FP downstream)
    if "human_id" in pred_df.columns:
        pred_human_ids = set(pred_df["human_id"].dropna().unique())
    else:
        pred_human_ids = set()
    if "object_id" in pred_df.columns:
        pred_object_ids = set(pred_df["object_id"].dropna().unique())
    else:
        pred_object_ids = set()

    mapped_human = set(map_df[map_df["type"] == "human"]["tracker_id"].tolist())
    mapped_object = set(map_df[map_df["type"] == "object"]["tracker_id"].tolist())

    unmapped_human = pred_human_ids - mapped_human
    unmapped_object = pred_object_ids - mapped_object
    if unmapped_human:
        warnings.append(f"Unmapped human tracker IDs: {sorted(unmapped_human)} -> counted as FP")
    if unmapped_object:
        warnings.append(f"Unmapped object tracker IDs: {sorted(unmapped_object)} -> counted as FP")

    return warnings, errors


def apply_id_map(pred_df: pd.DataFrame, map_df: pd.DataFrame) -> pd.DataFrame:
    """Translate pred_df human_id/object_id to human_label/object_label via map.

    Returns pred_df with added columns human_label (str) and object_label (str).
    Unmapped IDs map to NaN and are treated as FP downstream.
    """
    human_map = dict(
        zip(
            map_df[map_df["type"] == "human"]["tracker_id"],
            map_df[map_df["type"] == "human"]["gt_label"],
        )
    )
    object_map = dict(
        zip(
            map_df[map_df["type"] == "object"]["tracker_id"],
            map_df[map_df["type"] == "object"]["gt_label"],
        )
    )

    df = pred_df.copy()
    human_map = dict(
        zip(
            map_df[map_df["type"] == "human"]["tracker_id"],
            map_df[map_df["type"] == "human"]["gt_label"],
        )
    )
    object_map = dict(
        zip(
            map_df[map_df["type"] == "object"]["tracker_id"],
            map_df[map_df["type"] == "object"]["gt_label"],
        )
    )
    # Use list comprehension with explicit str() to avoid pandas dtype coercion
    # (pandas .map() and Series construction can convert string values to float
    #  when mixed with None/NaN).
    df["human_label"] = pd.Series(
        [
            str(human_map.get(hid)) if hid in human_map else None
            for hid in df["human_id"]
        ],
        dtype=object,
    )
    df["object_label"] = pd.Series(
        [
            str(object_map.get(oid)) if oid in object_map else None
            for oid in df["object_id"]
        ],
        dtype=object,
    )
    return df