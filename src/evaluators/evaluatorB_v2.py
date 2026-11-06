"""EvaluatorB_v2: label-verified interaction-ReID evaluation.

Per-class temporal IoU (tIoU) matching between ground truth interaction
intervals and predicted interaction intervals, using per-run ID maps to
resolve tracker IDs to stable ground-truth labels.

Outputs (under --out-dir):
    primary.csv                  - per-video overall P/R/F1 per threshold
    primary_summary.csv          - micro/macro aggregates per threshold
    primary_by_class.csv         - per-video per-class interval P/R/F1
    primary_by_class_summary.csv - per-class micro/macro aggregates

Usage:
    python -m evaluators.evaluatorB_v2 \
        --gt-dir src/eval/gt_summary \
        --pred-dir src/video/logs/logs_summary \
        --map-dir src/evaluators/id_maps \
        --out-dir src/eval/result \
        --thresholds 0.3 0.4 0.5 0.6 0.7
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from evaluators.id_map import apply_id_map, load_id_map, validate_id_map
from evaluators.gt_io import read_gt

CLASS_NAMES = ["cell phone", "laptop", "wallet", "tumbler", "bag"]

DEFAULT_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)


def pair_tiou(a: tuple, b: tuple) -> float:
    """Temporal IoU between two inclusive intervals [start, end]."""
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
    union = (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - inter
    return inter / union if union > 0 else 0.0


def _score_split(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    map_df: pd.DataFrame,
    thresholds: Tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> Dict:
    """Score a single video split (one GT file + one pred file).

    Returns dict with per-class and overall metrics per threshold, plus
    legacy ID-agnostic tIoU metrics.
    """
    warnings, errors = validate_id_map(map_df, pred_df)
    for w in warnings:
        print(f"[WARN] {w}")
    if errors:
        raise ValueError("; ".join(errors))

    # Translate predictions: human_id/object_id -> human_label/object_label
    pdf = apply_id_map(pred_df, map_df)
    pdf["object_class"] = pdf["object_class"].astype(str).str.strip()
    # Only predictions with both labels mapped are eligible for matching
    pdf_valid = pdf.dropna(subset=["human_label", "object_label"]).copy()

    # Normalize GT labels
    gt_df = gt_df.copy()
    gt_df["object_cls"] = gt_df["object_cls"].astype(str).str.strip()
    gt_df["human"] = gt_df["human"].astype(str).str.strip()
    gt_df["object"] = gt_df["object"].astype(str).str.strip()
    pdf_valid["human_label"] = pdf_valid["human_label"].astype(str).str.strip()
    pdf_valid["object_label"] = pdf_valid["object_label"].astype(str).str.strip()

    # Classes from GT + ALL preds (a class present only in unmapped preds must
    # still be scored, so its rows count as FP rather than vanishing)
    class_names = sorted(
        set(gt_df["object_cls"].unique()) | set(pdf["object_class"].unique())
    )

    results: Dict[str, dict] = {}
    legacy_matches = 0  # ID-agnostic tIoU matches (best threshold)

    for cls in class_names:
        g = gt_df[gt_df["object_cls"] == cls][
            ["frame_start", "frame_end", "human", "object"]
        ].values.tolist()
        n_gt = len(g)

        # ALL pred rows for this class (including unmapped) count toward n_pred
        n_pred_all = int(pdf[pdf["object_class"] == cls].shape[0])

        # Only mapped preds can match; unmapped preds are FP
        p_matchable = [
            (ps, pe, ph, po)
            for (ps, pe, ph, po) in zip(
                pdf_valid[pdf_valid["object_class"] == cls]["frame_start"],
                pdf_valid[pdf_valid["object_class"] == cls]["frame_end"],
                pdf_valid[pdf_valid["object_class"] == cls]["human_label"],
                pdf_valid[pdf_valid["object_class"] == cls]["object_label"],
            )
            if ph is not None and po is not None and str(ph) != "nan" and str(po) != "nan"
        ]
        n_pred_matchable = len(p_matchable)

        per_threshold: Dict[float, dict] = {}
        best_match_count = 0

        if n_gt == 0 and n_pred_matchable == 0:
            # No GT and no matchable preds for this class
            for t in thresholds:
                per_threshold[float(t)] = {
                    "TP": 0,
                    "FP": int(n_pred_all),  # All preds are FP
                    "FN": 0,
                    "P": 0.0,
                    "R": 0.0,
                    "F1": 0.0,
                    "n_match": 0,
                }
        elif n_gt == 0:
            # GT empty but preds exist -> all preds are FP
            for t in thresholds:
                per_threshold[float(t)] = {
                    "TP": 0,
                    "FP": int(n_pred_all),
                    "FN": 0,
                    "P": 0.0,
                    "R": 0.0,
                    "F1": 0.0,
                    "n_match": 0,
                }
        elif n_pred_matchable == 0:
            # GT exists but no matchable preds -> all GT are FN, all preds are FP
            for t in thresholds:
                per_threshold[float(t)] = {
                    "TP": 0,
                    "FP": int(n_pred_all),
                    "FN": int(n_gt),
                    "P": 0.0,
                    "R": 0.0,
                    "F1": 0.0,
                    "n_match": 0,
                }
        else:
            # Build tIoU matrix for candidate pairs (same human+object label)
            t_cost = np.full((n_gt, n_pred_matchable), 1.0, dtype=float)
            for i, (gs, ge, gh, go) in enumerate(g):
                for j, (ps, pe, ph, po) in enumerate(p_matchable):
                    if gh == ph and go == po:  # same human AND object label
                        tiou = pair_tiou((gs, ge), (ps, pe))
                        t_cost[i, j] = 1.0 - tiou

            # Hungarian assignment at each threshold
            for t in thresholds:
                cost = t_cost.copy()
                # Invalid pairs (different human/object) get infinite cost
                for i in range(n_gt):
                    for j in range(n_pred_matchable):
                        if t_cost[i, j] >= 1.0:  # was never set (different pair)
                            cost[i, j] = 1e9

                row_ind, col_ind = linear_sum_assignment(cost)
                matched = [
                    (r, c)
                    for r, c in zip(row_ind, col_ind)
                    # (1 - cost) is the pair's tIoU; accept pairs with tIoU >= threshold
                    if cost[r, c] < 1e9 and (1.0 - cost[r, c]) >= t
                ]
                tp = len(matched)
                # FP = all preds (including unmapped) minus matched
                fp = int(n_pred_all) - tp
                fn = int(n_gt) - tp
                P = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                R = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
                per_threshold[float(t)] = {
                    "TP": int(tp),
                    "FP": int(fp),
                    "FN": int(fn),
                    "P": round(P, 4),
                    "R": round(R, 4),
                    "F1": round(F1, 4),
                    "n_match": int(tp),
                }
                if tp > best_match_count:
                    best_match_count = tp

        results[cls] = {
            "n_gt": int(n_gt),
            "n_pred": int(n_pred_all),
            "per_threshold": per_threshold,
        }
        legacy_matches = max(legacy_matches, best_match_count)

    # Aggregate per-class to overall micro/macro at each threshold
    overall_per_threshold: Dict[float, dict] = {}
    for t in thresholds:
        tp = sum(r["per_threshold"][float(t)]["TP"] for r in results.values())
        fp = sum(r["per_threshold"][float(t)]["FP"] for r in results.values())
        fn = sum(r["per_threshold"][float(t)]["FN"] for r in results.values())
        P = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        R = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
        overall_per_threshold[float(t)] = {
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "P": round(P, 4),
            "R": round(R, 4),
            "F1": round(F1, 4),
        }

    return {
        "per_class": results,
        "per_threshold": overall_per_threshold,
        "legacy_tIoU_matches": int(legacy_matches),
    }


def score_video(
    gt_path: Path | str,
    pred_path: Path | str,
    map_path: Path | str,
    thresholds: Tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> dict:
    """Score a single video.

    Returns a dict with per-class and overall metrics per threshold, plus
    legacy ID-agnostic tIoU results.
    """
    gt_df = read_gt(gt_path)
    pred_df = pd.read_csv(Path(pred_path))
    map_df = load_id_map(Path(map_path))

    return _score_split(gt_df, pred_df, map_df, thresholds)


def _pair_key(name: str) -> str | None:
    """Pairing key from a GT/prediction filename.

    vidNN_* -> 'vidNN'; otherwise the name with a known suffix stripped
    (e.g., 'DrinkBottle_gt.csv' -> 'DrinkBottle').
    """
    import re

    m = re.search(r"vid(\d+)", name)
    if m:
        return f"vid{int(m.group(1)):02d}"
    for suf in ("_true_summary.csv", "_summary_true.csv", "_gt.csv", "_summary_log.csv"):
        if name.endswith(suf):
            stem = name[: -len(suf)]
            return stem if stem else None
    return None


def discover_pairs(
    gt_dir: Path | str, pred_dir: Path | str
) -> List[Tuple[str, Path, Path]]:
    """Discover matching GT/prediction pairs.

    GT accepts '*_true_summary.csv' (vid corpus) and '*_gt.csv' (scenario
    corpus, e.g. groundtruths/DrinkBottle_gt.csv); predictions use
    '*_summary_log.csv'. Files pair by key: 'vidNN' for vid-named videos,
    otherwise the shared name prefix (e.g. 'DrinkBottle').
    """
    gt_dir = Path(gt_dir)
    pred_dir = Path(pred_dir)

    gt_files: dict[str, Path] = {}
    for pattern in ("*_true_summary.csv", "*_gt.csv"):
        for f in gt_dir.glob(pattern):
            key = _pair_key(f.name)
            if key is not None:
                gt_files[key] = f

    pred_files: dict[str, Path] = {}
    for f in pred_dir.glob("*_summary_log.csv"):
        key = _pair_key(f.name)
        if key is not None:
            pred_files[key] = f

    pairs = []
    for k in sorted(set(gt_files) & set(pred_files)):
        pairs.append((k, gt_files[k], pred_files[k]))
    for k in sorted(set(gt_files) - set(pred_files)):
        print(f"[WARN] {k}: GT without prediction ({gt_files[k].name})")
    for k in sorted(set(pred_files) - set(gt_files)):
        print(f"[WARN] {k}: prediction without GT ({pred_files[k].name})")
    print(f"[INFO] scoring {len(pairs)} videos")
    return pairs


def run(
    gt_dir: Path | str,
    pred_dir: Path | str,
    map_dir: Path | str,
    out_dir: Path | str,
    thresholds: Tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> None:
    """Run evaluatorB_v2 on all GT/pred pairs."""
    gt_dir = Path(gt_dir)
    pred_dir = Path(pred_dir)
    map_dir = Path(map_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = discover_pairs(gt_dir, pred_dir)

    per_class_rows: List[dict] = []
    overall_rows: List[dict] = []

    for v, gp, sp in pairs:
        map_path = map_dir / f"{v}_map.csv"
        if not map_path.exists():
            print(f"[WARN] {v}: no map file at {map_path}, skipping")
            continue

        try:
            result = score_video(gp, sp, map_path, thresholds)
        except Exception as e:
            print(f"[ERROR] {v}: {e}")
            continue

        # Per-class rows
        for cls, r in result["per_class"].items():
            for t, m in r["per_threshold"].items():
                per_class_rows.append(
                    {
                        "video": v,
                        "class": cls,
                        "threshold": t,
                        "TP": m["TP"],
                        "FP": m["FP"],
                        "FN": m["FN"],
                        "P": m["P"],
                        "R": m["R"],
                        "F1": m["F1"],
                        "n_gt": r["n_gt"],
                        "n_pred": r["n_pred"],
                    }
                )

        # Overall per threshold
        for t, m in result["per_threshold"].items():
            overall_rows.append(
                {
                    "video": v,
                    "threshold": t,
                    "TP": m["TP"],
                    "FP": m["FP"],
                    "FN": m["FN"],
                    "P": m["P"],
                    "R": m["R"],
                    "F1": m["F1"],
                    "level": "video",
                }
            )

    # Aggregate across all videos
    df_pc = pd.DataFrame(
        per_class_rows,
        columns=["video", "class", "threshold", "TP", "FP", "FN", "P", "R", "F1", "n_gt", "n_pred"],
    )
    df_ov = pd.DataFrame(
        overall_rows,
        columns=["video", "threshold", "TP", "FP", "FN", "P", "R", "F1", "level"],
    )

    # Micro + macro aggregates
    summary_rows: List[dict] = []
    for t in thresholds:
        sub = df_ov[df_ov["threshold"] == t]
        if not sub.empty:
            tp, fp, fn = int(sub["TP"].sum()), int(sub["FP"].sum()), int(sub["FN"].sum())
            P = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            R = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
            summary_rows.append(
                {
                    "level": "micro",
                    "threshold": t,
                    "TP": tp,
                    "FP": fp,
                    "FN": fn,
                    "P": round(P, 4),
                    "R": round(R, 4),
                    "F1": round(F1, 4),
                    "n_videos": len(sub),
                }
            )
            macro_P = sub["P"].mean()
            macro_R = sub["R"].mean()
            macro_F1 = sub["F1"].mean()
            summary_rows.append(
                {
                    "level": "macro",
                    "threshold": t,
                    "TP": tp,
                    "FP": fp,
                    "FN": fn,
                    "P": round(macro_P, 4),
                    "R": round(macro_R, 4),
                    "F1": round(macro_F1, 4),
                    "n_videos": len(sub),
                }
            )

    # Per-class summary (micro over videos)
    pc_summary_rows: List[dict] = []
    for cls in CLASS_NAMES:
        sub = df_pc[df_pc["class"] == cls]
        if sub.empty:
            continue
        for t in thresholds:
            ssub = sub[sub["threshold"] == t]
            if ssub.empty:
                continue
            tp, fp, fn = int(ssub["TP"].sum()), int(ssub["FP"].sum()), int(ssub["FN"].sum())
            P = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            R = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
            pc_summary_rows.append(
                {
                    "level": "micro",
                    "class": cls,
                    "threshold": t,
                    "TP": tp,
                    "FP": fp,
                    "FN": fn,
                    "P": round(P, 4),
                    "R": round(R, 4),
                    "F1": round(F1, 4),
                    "n_videos": len(ssub),
                }
            )
            macro_P = ssub["P"].mean()
            macro_R = ssub["R"].mean()
            macro_F1 = ssub["F1"].mean()
            pc_summary_rows.append(
                {
                    "level": "macro",
                    "class": cls,
                    "threshold": t,
                    "TP": tp,
                    "FP": fp,
                    "FN": fn,
                    "P": round(macro_P, 4),
                    "R": round(macro_R, 4),
                    "F1": round(macro_F1, 4),
                    "n_videos": len(ssub),
                }
            )

    # Save outputs
    df_ov.to_csv(out_dir / "primary.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(out_dir / "primary_summary.csv", index=False)
    df_pc.to_csv(out_dir / "primary_by_class.csv", index=False)
    pd.DataFrame(pc_summary_rows).to_csv(
        out_dir / "primary_by_class_summary.csv", index=False
    )

    print(f"[INFO] wrote {out_dir / 'primary.csv'}")
    print(f"[INFO] wrote {out_dir / 'primary_summary.csv'}")
    print(f"[INFO] wrote {out_dir / 'primary_by_class.csv'}")
    print(f"[INFO] wrote {out_dir / 'primary_by_class_summary.csv'}")


def main(argv: List[str] | None = None) -> None:
    """CLI entry point for evaluatorB_v2."""
    parser = argparse.ArgumentParser(
        description="Label-verified interaction evaluation (interval tIoU)."
    )
    parser.add_argument("--gt-dir", required=True, help="Directory with *_true_summary.csv (ground truth)")
    parser.add_argument("--pred-dir", required=True, help="Directory with *_summary_log.csv (predictions)")
    parser.add_argument("--map-dir", required=True, help="Directory with <video>_map.csv (per-run ID maps)")
    parser.add_argument("--out-dir", required=True, help="Output directory for result CSVs")
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=list(DEFAULT_THRESHOLDS),
        help="tIoU thresholds (default: 0.3 0.4 0.5 0.6 0.7)",
    )
    args = parser.parse_args(argv)
    run(
        args.gt_dir,
        args.pred_dir,
        args.map_dir,
        args.out_dir,
        tuple(args.thresholds),
    )


if __name__ == "__main__":
    main()