"""EvaluatorB_frame_v2: frame-level binary interaction scorer.

Frame-level scorer: each stride-grid frame is labeled INTERACTING if any
verified interaction interval covers it, else NOT INTERACTING.
Confusion matrix (TP/FP/FN/TN) -> Precision, Recall, F1.

Additional: 5-tuple frame scoring (exact frame/human/object/class match).

Outputs:
    primary_frames.csv              - per-video binary P/R/F1 (unchanged schema)
    primary_frames_summary.csv      - micro/macro summary (unchanged schema)
    primary_frames_by_class.csv     # NEW: per-class frame P/R/F1 (binary)
    primary_frames_5tuple.csv       # NEW: per-video 5-tuple P/R/F1
    primary_frames_5tuple_summary.csv     # NEW: micro/macro 5-tuple summary
    primary_frames_5tuple_by_class.csv   # NEW: per-class 5-tuple P/R/F1
    primary_frames_5tuple_by_class_summary.csv  # NEW

Usage:
    python -m evaluators.evaluatorB_frame_v2 \
        --gt-dir src/eval/gt_summary \
        --pred-dir src/video/logs/logs_summary \
        --map-dir src/evaluators/id_maps \
        --out-dir src/eval/result \
        --stride 5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from evaluators.id_map import apply_id_map, load_id_map, validate_id_map

CLASS_NAMES = ["cell phone", "laptop", "wallet", "bottle", "bag", "cup"]
DEFAULT_STRIDE = 5


def _vidnum(name: str) -> int | None:
    import re

    m = re.search(r"vid(\d+)", name)
    return int(m.group(1)) if m else None


def expand_intervals(intervals: List[Tuple[int, int]], max_end: int, stride: int) -> set[int]:
    """Stride-grid frames covered by any inclusive [start, end] interval."""
    active: set[int] = set()
    for s, e in intervals:
        first = s + ((stride - s % stride) % stride)
        for f in range(first, e + 1, stride):
            active.add(f)
    return active


def expand_5tuples(
    gt_df: pd.DataFrame,
    max_end: int,
    stride: int,
) -> set[Tuple[int, str, str, str]]:
    """Expand GT interaction intervals to (frame, human_label, object_label, object_cls)
    5-tuples on the stride grid."""
    active: set[Tuple[int, str, str, str]] = set()
    for _, g in gt_df.iterrows():
        gs, ge = int(g["frame_start"]), int(g["frame_end"])
        h_label = str(g["human"])
        o_label = str(g["object"])
        c = str(g["object_cls"])
        first = gs + ((stride - gs % stride) % stride)
        for f in range(first, ge + 1, stride):
            active.add((f, h_label, o_label, c))
    return active


def expand_pred_5tuples(
    pred_df: pd.DataFrame,
    max_end: int,
    stride: int,
) -> set[Tuple[int, str, str, str]]:
    """Expand detected interactions to (frame, human_label, object_label, object_class)
    5-tuples on the stride grid."""
    active: set[Tuple[int, str, str, str]] = set()
    for _, r in pred_df.iterrows():
        gs, ge = int(r["frame_start"]), int(r["frame_end"])
        h_label = str(r["human_label"])
        o_label = str(r["object_label"])
        c = str(r["object_class"])
        first = gs + ((stride - gs % stride) % stride)
        for f in range(first, ge + 1, stride):
            active.add((f, h_label, o_label, c))
    return active


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    P = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    R = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    F = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
    return P, R, F


def score_video_frame(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    map_df: pd.DataFrame,
    stride: int = DEFAULT_STRIDE,
) -> dict:
    """Score a single video at frame level.

    Returns dict with binary P/R/F1 and per-class P/R/F1, plus 5-tuple
    (frame, human_label, object_label, object_cls) P/R/F1.
    """
    warnings, errors = validate_id_map(map_df, pred_df)
    for w in warnings:
        print(f"[WARN] {w}")
    if errors:
        raise ValueError("; ".join(errors))

    pdf = apply_id_map(pred_df, map_df)
    pdf_valid = pdf.dropna(subset=["human_label", "object_label"]).copy()

    gt_df = gt_df.copy()
    gt_df["object_cls"] = gt_df["object_cls"].astype(str).str.strip()
    gt_df["human"] = gt_df["human"].astype(str).str.strip()
    gt_df["object"] = gt_df["object"].astype(str).str.strip()
    pdf_valid["object_class"] = pdf_valid["object_class"].astype(str).str.strip()

    # ---- Binary frame scoring (any interaction) ----
    gt_any: List[Tuple[int, int]] = [
        (int(g["frame_start"]), int(g["frame_end"])) for _, g in gt_df.iterrows()
    ]
    pred_any: List[Tuple[int, int]] = [
        (int(r["frame_start"]), int(r["frame_end"])) for _, r in pdf_valid.iterrows()
    ]

    max_end = 0
    if gt_any:
        max_end = max(e for _, e in gt_any)
    if pred_any:
        max_end = max(max_end, int(pred_df["frame_end"].max()))
    if not gt_df.empty:
        max_end = max(max_end, int(gt_df["frame_end"].max()))

    grid = list(range(0, max_end + 1, stride))

    gt_active = expand_intervals(gt_any, max_end, stride)
    pred_active = expand_intervals(pred_any, max_end, stride)

    tp = len(gt_active & pred_active)
    fp = len(pred_active - gt_active)
    fn = len(gt_active - pred_active)
    tn = len(set(grid) - gt_active - pred_active)
    P, R, F = _prf(tp, fp, fn)

    # ---- Per-class frame metrics (binary: any interaction of class) ----
    per_class: dict[str, dict] = {}
    for cls in CLASS_NAMES:
        gt_cls = gt_df[gt_df["object_cls"] == cls][
            ["frame_start", "frame_end"]
        ].values.tolist()
        pred_cls = pdf_valid[pdf_valid["object_class"] == cls][
            ["frame_start", "frame_end"]
        ].values.tolist()
        gt_active_cls = expand_intervals(gt_cls, max_end, stride)
        pred_active_cls = expand_intervals(pred_cls, max_end, stride)
        tp_c = len(gt_active_cls & pred_active_cls)
        fp_c = len(pred_active_cls - gt_active_cls)
        fn_c = len(gt_active_cls - pred_active_cls)
        P_c, R_c, F_c = _prf(tp_c, fp_c, fn_c)
        per_class[cls] = {
            "TP": int(tp_c),
            "FP": int(fp_c),
            "FN": int(fn_c),
            "P": round(P_c, 4),
            "R": round(R_c, 4),
            "F1": round(F_c, 4),
        }

    # ---- 5-tuple frame scoring (exact human/object/class match per frame) ----
    gt_tuples = expand_5tuples(gt_df, max_end, stride)
    pred_tuples = expand_pred_5tuples(pdf_valid, max_end, stride)

    tp_5 = len(gt_tuples & pred_tuples)
    fp_5 = len(pred_tuples - gt_tuples)
    fn_5 = len(gt_tuples - pred_tuples)
    P_5, R_5, F_5 = _prf(tp_5, fp_5, fn_5)

    per_class_5: dict[str, dict] = {}
    for cls in CLASS_NAMES:
        gt_cls_tuples = {t for t in gt_tuples if t[3] == cls}
        pred_cls_tuples = {t for t in pred_tuples if t[3] == cls}
        tp_c = len(gt_cls_tuples & pred_cls_tuples)
        fp_c = len(pred_cls_tuples - gt_cls_tuples)
        fn_c = len(gt_cls_tuples - pred_cls_tuples)
        P_c, R_c, F_c = _prf(tp_c, fp_c, fn_c)
        per_class_5[cls] = {
            "TP": int(tp_c),
            "FP": int(fp_c),
            "FN": int(fn_c),
            "P": round(P_c, 4),
            "R": round(R_c, 4),
            "F1": round(F_c, 4),
        }

    return {
        "n_frames": len(grid),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "P": round(P, 4),
        "R": round(R, 4),
        "F1": round(F, 4),
        "per_class": per_class,
        # 5-tuple results
        "TP_5": int(tp_5),
        "FP_5": int(fp_5),
        "FN_5": int(fn_5),
        "P_5": round(P_5, 4),
        "R_5": round(R_5, 4),
        "F1_5": round(F_5, 4),
        "per_class_5": per_class_5,
    }


def discover_pairs(
    gt_dir: Path | str, pred_dir: Path | str
) -> List[Tuple[str, Path, Path]]:
    """Discover matching GT/pred video pairs by video number."""
    gt_dir = Path(gt_dir)
    pred_dir = Path(pred_dir)

    gt_files: dict[int, Path] = {}
    for f in gt_dir.glob("*_true_summary.csv"):
        m = _vidnum(f.name)
        if m is not None:
            gt_files[m] = f

    pred_files: dict[int, Path] = {}
    for f in pred_dir.glob("*_summary_log.csv"):
        m = _vidnum(f.name)
        if m is not None:
            pred_files[m] = f

    pairs = []
    for v in sorted(set(gt_files) & set(pred_files)):
        pairs.append((f"vid{v:02d}", gt_files[v], pred_files[v]))
    for v in sorted(set(gt_files) - set(pred_files)):
        print(f"[WARN] vid{v:02d}: GT without prediction ({gt_files[v].name})")
    for v in sorted(set(pred_files) - set(gt_files)):
        print(f"[WARN] vid{v:02d}: prediction without GT ({pred_files[v].name})")
    print(f"[INFO] scoring {len(pairs)} videos")
    return pairs


def run(
    gt_dir: Path | str,
    pred_dir: Path | str,
    map_dir: Path | str,
    out_dir: Path | str,
    stride: int = DEFAULT_STRIDE,
) -> None:
    """Run evaluatorB_frame_v2 on all GT/pred pairs."""
    gt_dir = Path(gt_dir)
    pred_dir = Path(pred_dir)
    map_dir = Path(map_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = discover_pairs(gt_dir, pred_dir)

    frame_rows: List[dict] = []
    per_class_rows: List[dict] = []
    frame_5_rows: List[dict] = []
    per_class_5_rows: List[dict] = []

    for v, gp, sp in pairs:
        map_path = map_dir / f"{v}_map.csv"
        if not map_path.exists():
            print(f"[WARN] {v}: no map file at {map_path}, skipping")
            continue

        try:
            gt_df = pd.read_csv(gp)
            pred_df = pd.read_csv(sp)
            map_df = load_id_map(map_path)
            result = score_video_frame(gt_df, pred_df, map_df, stride)
        except Exception as e:
            print(f"[ERROR] {v}: {e}")
            continue

        frame_rows.append(
            {
                "video": f"vid{v:02d}",
                "n_frames": result["n_frames"],
                "TP": result["TP"],
                "FP": result["FP"],
                "FN": result["FN"],
                "TN": result["TN"],
                "P": result["P"],
                "R": result["R"],
                "F1": result["F1"],
            }
        )

        for cls, m in result["per_class"].items():
            per_class_rows.append(
                {
                    "video": f"vid{v:02d}",
                    "class": cls,
                    "TP": m["TP"],
                    "FP": m["FP"],
                    "FN": m["FN"],
                    "P": m["P"],
                    "R": m["R"],
                    "F1": m["F1"],
                }
            )

        # 5-tuple frame rows
        frame_5_rows.append(
            {
                "video": f"vid{v:02d}",
                "n_frames": result["n_frames"],
                "TP_5": result["TP_5"],
                "FP_5": result["FP_5"],
                "FN_5": result["FN_5"],
                "P_5": result["P_5"],
                "R_5": result["R_5"],
                "F1_5": result["F1_5"],
            }
        )

        for cls, m in result["per_class_5"].items():
            per_class_5_rows.append(
                {
                    "video": f"vid{v:02d}",
                    "class": cls,
                    "TP_5": m["TP"],
                    "FP_5": m["FP"],
                    "FN_5": m["FN"],
                    "P_5": m["P"],
                    "R_5": m["R"],
                    "F1_5": m["F1"],
                }
            )

    df_frames = pd.DataFrame(frame_rows)
    df_pc = pd.DataFrame(per_class_rows)
    df_5 = pd.DataFrame(frame_5_rows)
    df_pc_5 = pd.DataFrame(per_class_5_rows)

    # Summary for binary frame
    tp = int(df_frames["TP"].sum()) if not df_frames.empty else 0
    fp = int(df_frames["FP"].sum()) if not df_frames.empty else 0
    fn = int(df_frames["FN"].sum()) if not df_frames.empty else 0
    tn = int(df_frames["TN"].sum()) if not df_frames.empty else 0
    P, R, F = _prf(tp, fp, fn)

    summary_rows: List[dict] = []
    summary_rows.append(
        {
            "level": "micro",
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "P": round(P, 4),
            "R": round(R, 4),
            "F1": round(F, 4),
            "n_videos": len(df_frames),
            "n_frames": int(df_frames["n_frames"].sum()) if not df_frames.empty else 0,
        }
    )
    if not df_frames.empty:
        summary_rows.append(
            {
                "level": "macro",
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "P": round(df_frames["P"].mean(), 4),
                "R": round(df_frames["R"].mean(), 4),
                "F1": round(df_frames["F1"].mean(), 4),
                "n_videos": len(df_frames),
                "n_frames": int(df_frames["n_frames"].sum()),
            }
        )

    # Per-class summary (binary)
    pc_summary_rows: List[dict] = []
    for cls in CLASS_NAMES:
        sub = df_pc[df_pc["class"] == cls]
        if sub.empty:
            continue
        tp_c, fp_c, fn_c = int(sub["TP"].sum()), int(sub["FP"].sum()), int(sub["FN"].sum())
        P_c, R_c, F_c = _prf(tp_c, fp_c, fn_c)
        pc_summary_rows.append(
            {
                "level": "micro",
                "class": cls,
                "TP": tp_c,
                "FP": fp_c,
                "FN": fn_c,
                "P": round(P_c, 4),
                "R": round(R_c, 4),
                "F1": round(F_c, 4),
                "n_videos": len(sub),
            }
        )
        pc_summary_rows.append(
            {
                "level": "macro",
                "class": cls,
                "TP": tp_c,
                "FP": fp_c,
                "FN": fn_c,
                "P": round(sub["P"].mean(), 4),
                "R": round(sub["R"].mean(), 4),
                "F1": round(sub["F1"].mean(), 4),
                "n_videos": len(sub),
            }
        )

    # 5-tuple summary
    tp_5 = int(df_5["TP_5"].sum()) if not df_5.empty else 0
    fp_5 = int(df_5["FP_5"].sum()) if not df_5.empty else 0
    fn_5 = int(df_5["FN_5"].sum()) if not df_5.empty else 0
    P_5, R_5, F_5 = _prf(tp_5, fp_5, fn_5)

    summary_5_rows: List[dict] = []
    summary_5_rows.append(
        {
            "level": "micro",
            "TP_5": tp_5,
            "FP_5": fp_5,
            "FN_5": fn_5,
            "P_5": round(P_5, 4),
            "R_5": round(R_5, 4),
            "F1_5": round(F_5, 4),
            "n_videos": len(df_5),
            "n_frames": int(df_5["n_frames"].sum()) if not df_5.empty else 0,
        }
    )
    if not df_5.empty:
        summary_5_rows.append(
            {
                "level": "macro",
                "TP_5": tp_5,
                "FP_5": fp_5,
                "FN_5": fn_5,
                "P_5": round(df_5["P_5"].mean(), 4),
                "R_5": round(df_5["R_5"].mean(), 4),
                "F1_5": round(df_5["F1_5"].mean(), 4),
                "n_videos": len(df_5),
                "n_frames": int(df_5["n_frames"].sum()),
            }
        )

    pc_5_summary_rows: List[dict] = []
    for cls in CLASS_NAMES:
        sub = df_pc_5[df_pc_5["class"] == cls]
        if sub.empty:
            continue
        tp_c, fp_c, fn_c = (
            int(sub["TP_5"].sum()),
            int(sub["FP_5"].sum()),
            int(sub["FN_5"].sum()),
        )
        P_c, R_c, F_c = _prf(tp_c, fp_c, fn_c)
        pc_5_summary_rows.append(
            {
                "level": "micro",
                "class": cls,
                "TP_5": tp_c,
                "FP_5": fp_c,
                "FN_5": fn_c,
                "P_5": round(P_c, 4),
                "R_5": round(R_c, 4),
                "F1_5": round(F_c, 4),
                "n_videos": len(sub),
            }
        )
        pc_5_summary_rows.append(
            {
                "level": "macro",
                "class": cls,
                "TP_5": tp_c,
                "FP_5": fp_c,
                "FN_5": fn_c,
                "P_5": round(sub["P_5"].mean(), 4),
                "R_5": round(sub["R_5"].mean(), 4),
                "F1_5": round(sub["F1_5"].mean(), 4),
                "n_videos": len(sub),
            }
        )

    # Write outputs
    df_frames.to_csv(out_dir / "primary_frames.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(out_dir / "primary_frames_summary.csv", index=False)
    df_pc.to_csv(out_dir / "primary_frames_by_class.csv", index=False)
    pd.DataFrame(pc_summary_rows).to_csv(out_dir / "primary_frames_by_class_summary.csv", index=False)

    df_5.to_csv(out_dir / "primary_frames_5tuple.csv", index=False)
    pd.DataFrame(summary_5_rows).to_csv(out_dir / "primary_frames_5tuple_summary.csv", index=False)
    df_pc_5.to_csv(out_dir / "primary_frames_5tuple_by_class.csv", index=False)
    pd.DataFrame(pc_5_summary_rows).to_csv(out_dir / "primary_frames_5tuple_by_class_summary.csv", index=False)

    print(f"[INFO] wrote {out_dir / 'primary_frames.csv'}")
    print(f"[INFO] wrote {out_dir / 'primary_frames_summary.csv'}")
    print(f"[INFO] wrote {out_dir / 'primary_frames_by_class.csv'}")
    print(f"[INFO] wrote {out_dir / 'primary_frames_by_class_summary.csv'}")
    print(f"[INFO] wrote {out_dir / 'primary_frames_5tuple.csv'}")
    print(f"[INFO] wrote {out_dir / 'primary_frames_5tuple_summary.csv'}")
    print(f"[INFO] wrote {out_dir / 'primary_frames_5tuple_by_class.csv'}")
    print(f"[INFO] wrote {out_dir / 'primary_frames_5tuple_by_class_summary.csv'}")
    print(f"[INFO] binary   P={P:.4f} R={R:.4f} F1={F:.4f}")
    print(f"[INFO] 5-tuple P_5={P_5:.4f} R_5={R_5:.4f} F1_5={F_5:.4f}")