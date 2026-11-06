"""hoid_engine - inference core ported from src/hoid_model_run.ipynb.

The notebook (hoid_model_run.ipynb) is the reference and is left untouched.
This module copies its logic verbatim (config, Interaction-ReID wrappers,
pad-aware IoU, process_video loop) and adds ONLY lightweight, non-blocking
hooks for a UI: an optional queue for preview/progress/summary messages,
a stop_event for cancellation, and a preview toggle.

Lightweight rules:
- No Tk imports here.
- Inference stays full-resolution; preview thumbnails are downscaled copies.
- Queue sends are put_nowait (drop-if-full) so the UI can never slow inference.
- Models are loaded once per process via ensure_models_loaded(), not per video.
"""

from pathlib import Path

import cv2
import numpy as np
import queue as queue_mod
import threading
import time
import torch
import torch.nn.functional as F

from ultralytics import YOLO
from transformers import AutoModelForDepthEstimation

# ---------------------------------------------------------------------------
# Paths / config (verbatim from notebook Cells 5-7)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

detect_model_name = "models/detect/1027best.pt"
pose_model_name = "models/pose/yolo11x-pose-trained.pt"
phone_botsort = "trackers/detect_botsort.yaml"
person_botsort = "trackers/person_botsort.yaml"

# Detect target classes (subset of best.pt names; aliases map model->target)
detect_target_names = ["phone", "laptop", "wallet", "bottle", "bag", "cup"]
detect_aliases = {"cell phone": "phone"}

video_input_directory = BASE_DIR / "video" / "input"
video_output_directory = BASE_DIR / "video" / "output"
logs_frame_directory = BASE_DIR / "video" / "logs" / "logs_frames"
logs_summary_directory = BASE_DIR / "video" / "logs" / "logs_summary"

# Systems configuration (Cell 7)
detect_conf_thres = 0.2
# Per-class output thresholds (applied post-track; track() runs at the min).
# Calibrated on Scenario1-7 spotchecks (146 frames): laptop/cup strong,
# wallet/bottle weak tails, phone ~zero recall at 640px (kept low to catch
# rare hits like the cafe 0.55 without adding scenario noise: 0 hits @0.03).
detect_conf_by_class = {
    "phone": 0.05,
    "laptop": 0.30,
    "wallet": 0.15,
    "bottle": 0.20,
    "bag": 0.20,
    "cup": 0.30,
}
pose_conf_thres = 0.3
video_stride = 5
ioa_thres = 0.0
frame_stitching_threshold = 60

# Progress print flags (headless mode only; UI uses queue messages instead)
progress_frame_enabled = True
progress_bar_enabled = True
progress_percent_enabled = True
progress_elapsed_time_enabled = True
progress_bar_length = 40

# Depth Anything V2 configuration (from notebook)
DEPTH_USE = True  # False = pure 2D wrist-in-box rule (depths logged as -1)
DEPTH_GATE = 1.0  # max |matched-wrist minus object| depth gap (relative units)
VETO_GATE = 3.0   # torso veto: drop link if |person-torso minus object| exceeds this
DEPTH_BOX_INSET = 0.25  # erode each box side by this fraction for median sampling
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
DEPTH_CACHE_DIR = "models/depth"

# KP_KEEP skeleton configuration (shoulders/elbows/wrists/hips only)
KP_KEEP = [5, 6, 7, 8, 9, 10, 11, 12]
KP_EDGES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (11, 12), (5, 11), (6, 12)]
KP_CONF_MIN = 0.5  # skip low-confidence joints in overlay (tracking/logic unaffected)

# Colors (BGR, as used by cv2 in notebook)
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
white = (255, 255, 255)

# KP_KEEP skeleton palette (BGR per edge)
KP_PALETTE = [(0, 255, 255), (255, 0, 0), (255, 0, 255), (0, 165, 255),
              (255, 255, 0), (180, 0, 180), (0, 255, 0), (255, 255, 255)]  ### BGR per edge

# Per-class overlay colors (true BGR) for detect boxes + ID text
class_colors = {
    "phone": (0, 255, 255),
    "laptop": (255, 0, 0),
    "wallet": (255, 0, 255),
    "bottle": (0, 165, 255),
    "bag": (255, 255, 0),
    "cup": (180, 0, 180),
}

pad_w = 15
pad_h = 40

plot_keypoints = True  # False => skip p_result.plot() skeleton render

# Preview thumbnail budget (UI display only, never affects inference/writer)
# Default display box; UI may pass a per-video aspect-fit box instead.
PREVIEW_MAX_W = 640
PREVIEW_MAX_H = 360
PREVIEW_MIN_INTERVAL_S = 0.1   # ~10 fps max thumbnails
SUMMARY_MIN_INTERVAL_S = 1.0   # live summary refresh throttle


def get_video_info(video_path):
    """Cheap one-time probe: return (w, h, fps, total_frames) or Nones."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None, None, None, None
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        cap.release()
        if w <= 0 or h <= 0:
            return None, None, None, None
        return w, h, fps, total if total > 0 else 1
    except Exception:
        return None, None, None, None


def fit_display_box(vw, vh, cap_w=PREVIEW_MAX_W, cap_h=PREVIEW_MAX_H):
    """Aspect-fit (vw, vh) inside (cap_w, cap_h). Never upscales."""
    try:
        vw, vh = int(vw), int(vh)
        cap_w, cap_h = int(cap_w), int(cap_h)
    except Exception:
        return PREVIEW_MAX_W, PREVIEW_MAX_H
    if vw <= 0 or vh <= 0 or cap_w <= 0 or cap_h <= 0:
        return PREVIEW_MAX_W, PREVIEW_MAX_H
    scale = min(cap_w / float(vw), cap_h / float(vh), 1.0)
    return max(1, int(vw * scale)), max(1, int(vh * scale))

# ---------------------------------------------------------------------------
# Device (Cell 4)
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    using_gpu = 0
else:
    using_gpu = "cpu"

use_half = 16 if using_gpu != "cpu" else None

# Globals used by the pad-aware IoU wrapper (same as notebook)
video_width = 0
video_height = 0

# ---------------------------------------------------------------------------
# Interaction-ReID supplementary rescue pass (Cell 8, verbatim)
# ---------------------------------------------------------------------------
RESCUE_COST = 0.25
MAX_AGE = 90
HANDOFF_DIST = 75.0

from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.basetrack import TrackState
from ultralytics.trackers.utils import matching

_IR_STATE = {
    "person_boxes": {},
    "person_wrists": {},
    "last_box": {},
    "last_held_by": {},
}


def _conf_keep_mask(d_clss, d_confs, idx_to_target, device):
    """Bool mask (on `device`): keep boxes at/above their class's output threshold.

    Tracker runs at the min active threshold for continuity; this drops
    weak boxes before HOI/logs/drawing without disturbing track IDs.
    """
    keep = []
    for c, s in zip(d_clss.tolist(), d_confs.tolist()):
        tn = idx_to_target.get(int(c))
        keep.append(bool(float(s) >= detect_conf_by_class.get(tn, detect_conf_thres)))
    return torch.tensor(keep, dtype=torch.bool, device=device)


def _ir_prune(frame_id):
    """Age out every memory independently so dicts never grow unbounded."""
    stale_box = [k for k, (_, f) in _IR_STATE["last_box"].items() if frame_id - f > MAX_AGE]
    for tid in stale_box:
        _IR_STATE["last_box"].pop(tid, None)
    stale_held = [k for k in _IR_STATE["last_held_by"] if k not in _IR_STATE["last_box"]]
    for tid in stale_held:
        _IR_STATE["last_held_by"].pop(tid, None)
    _IR_STATE["person_wrists"] = {}


def _ir_inter_area(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h)


def _ir_area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _ir_wrist_holder(dbox):
    """Person whose wrist lies inside the padded dbox; None if none."""
    px1 = dbox[0] - pad_w
    py1 = dbox[1] - pad_h
    px2 = dbox[2] + pad_w
    py2 = dbox[3] + pad_h
    for pkey, wrists in _IR_STATE["person_wrists"].items():
        for wx, wy in wrists:
            if px1 <= wx <= px2 and py1 <= wy <= py2:
                return pkey
    return None


def _ir_center_dist(a, b):
    ca = ((a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0)
    cb = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
    return ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5


def _ir_owner_wrist_dist(owner_key, dbox):
    """Min center-distance from dbox center to owner's nearest wrist."""
    wrists = _IR_STATE["person_wrists"].get(owner_key)
    if not wrists:
        return None
    cx = (dbox[0] + dbox[2]) / 2.0
    cy = (dbox[1] + dbox[3]) / 2.0
    return min(((wx - cx) ** 2 + (wy - cy) ** 2) ** 0.5 for wx, wy in wrists)


if not getattr(YOLO.track, "_ir_wrapped", False):
    _ir_orig_yolo_track = YOLO.track

    def _ir_yolo_track(self, *args, **kwargs):
        results = _ir_orig_yolo_track(self, *args, **kwargs)
        try:
            if getattr(self, "task", "") == "pose" and results:
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    xyxys = boxes.xyxy.detach().cpu().numpy()
                    tids = boxes.id
                    tids_list = tids.int().tolist() if tids is not None else [None] * len(xyxys)
                    stash = {}
                    for j in range(len(xyxys)):
                        key = int(tids_list[j]) if tids_list[j] is not None else "idx" + str(j)
                        stash[key] = xyxys[j]
                    _IR_STATE["person_boxes"] = stash
                    if results[0].keypoints is not None:
                        kxy = results[0].keypoints.xy.detach().cpu().numpy()
                        wr = {}
                        for j in range(len(xyxys)):
                            key = int(tids_list[j]) if tids_list[j] is not None else "idx" + str(j)
                            wr[key] = [kxy[j, 9], kxy[j, 10]]
                        _IR_STATE["person_wrists"] = wr
        except Exception as err:
            print(f"[interaction-reid] person stash skipped ({err})")
        return results

    _ir_yolo_track._ir_wrapped = True
    YOLO.track = _ir_yolo_track
    print("[interaction-reid] YOLO.track wrapper installed")


if not getattr(BOTSORT, "_ir_installed", False):
    _ir_orig_post_first = BOTSORT._post_first_association

    def _ir_post_first(self, strack_pool, detections, u_track, u_detection,
                       activated_stracks, refind_stracks):
        if getattr(self.args, "with_reid", False):
            return _ir_orig_post_first(self, strack_pool, detections, u_track, u_detection,
                                       activated_stracks, refind_stracks)

        _mx = _IR_STATE.get("_max_frame", -1)
        if self.frame_id < _mx:
            for _k in ("last_box", "last_held_by"):
                _IR_STATE.get(_k, {}).clear()
        _IR_STATE["_max_frame"] = int(self.frame_id)

        u_track = list(u_track)
        u_detection = list(u_detection)

        for trk in strack_pool:
            if trk.state == TrackState.Tracked:
                box = np.asarray(trk.xyxy, dtype=float)
                _IR_STATE["last_box"][trk.track_id] = [box, self.frame_id]

        rescue_pos = [k for k, i in enumerate(u_track)
                      if strack_pool[i].state == TrackState.Lost]
        if rescue_pos and u_detection:
            rescue_ids = [u_track[k] for k in rescue_pos]
            rescue_trks = [strack_pool[i] for i in rescue_ids]
            det_objs = [detections[j] for j in u_detection]

            cost = np.full((len(rescue_trks), len(det_objs)), 1.0, dtype=float)
            det_boxes = [np.asarray(det.xyxy, dtype=float) for det in det_objs]
            det_holders = [_ir_wrist_holder(db) for db in det_boxes]
            for a, trk in enumerate(rescue_trks):
                owner_key = _IR_STATE["last_held_by"].get(trk.track_id)
                if owner_key is None:
                    continue
                ref_entry = _IR_STATE["last_box"].get(trk.track_id)
                ref_box = ref_entry[0] if ref_entry is not None else np.asarray(trk.xyxy, dtype=float)
                for b, det in enumerate(det_objs):
                    dbox = det_boxes[b]
                    holder = det_holders[b]
                    wrist_d = _ir_owner_wrist_dist(owner_key, dbox)
                    ref_d = _ir_center_dist(dbox, ref_box)
                    tier1 = holder == owner_key
                    tier2 = (holder is None and wrist_d is not None and
                             (wrist_d <= HANDOFF_DIST or ref_d <= HANDOFF_DIST))
                    if not (tier1 or tier2):
                        continue
                    union = (_ir_area(ref_box) + _ir_area(dbox) - _ir_inter_area(ref_box, dbox)) or 1e-9
                    geo = 1.0 - (_ir_inter_area(ref_box, dbox) / union)
                    cost[a, b] = RESCUE_COST + 0.15 * geo + min(0.09, ref_d / 2000.0)

            matches, _, _ = matching.linear_assignment(cost, thresh=self.args.match_thresh)

            if matches:
                self._apply_matches(matches, rescue_trks, det_objs, activated_stracks, refind_stracks)
                matched_track_ids = {rescue_trks[a].track_id for a, _ in matches}
                matched_det_ids = {id(det_objs[b]) for _, b in matches}
                u_track = [i for i in u_track if strack_pool[i].track_id not in matched_track_ids]
                u_detection = [j for j in u_detection if id(detections[j]) not in matched_det_ids]

        _ir_prune(self.frame_id)
        return u_track, u_detection

    _ir_post_first._ir_original = _ir_orig_post_first
    BOTSORT._post_first_association = _ir_post_first
    BOTSORT._ir_installed = True

# ---------------------------------------------------------------------------
# Pad-aware IoU (Cell 9, verbatim)
# ---------------------------------------------------------------------------
from ultralytics.trackers.utils import matching as _ir_matching_mod
import os


# Stage profiler (additive, off by default). Set HOID_PROFILE=1 to print a
# per-stage mean-ms table at the end of process_video (thesis perf numbers).
# With sync boundaries: each bucket owns exactly its GPU work (slight
# serialization overhead — compare profiled runs only against each other).
# gpu_ms column (CUDA events) is device-busy time; host-gpu gap = CPU + syncs.
_PROF = os.environ.get("HOID_PROFILE") == "1"
_prof_t = {}
_prof_n = {}
_prof_g = {}


def _prof_mark():
    if _PROF and torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def _prof_tick(name, t0, ev0=None):
    if _PROF:
        if ev0 is not None and torch.cuda.is_available():
            ev1 = torch.cuda.Event(True)
            ev1.record()
            torch.cuda.synchronize()
            _prof_g[name] = _prof_g.get(name, 0.0) + ev0.elapsed_time(ev1)
        dt = (time.perf_counter() - t0) * 1000.0
        _prof_t[name] = _prof_t.get(name, 0.0) + dt
        _prof_n[name] = _prof_n.get(name, 0) + 1


def _prof_ev():
    if _PROF and torch.cuda.is_available():
        ev = torch.cuda.Event(True)
        ev.record()
        return ev
    return None


def _prof_report():
    if _PROF and _prof_n:
        print("[hoid-profile] stage means (ms): host [, gpu]:", flush=True)
        for k in ("pose_track", "det_track", "hoi", "draw", "write", "depth"):
            if k in _prof_n:
                g = _prof_g.get(k, 0.0) / _prof_n[k] if k in _prof_g else None
                gs = f", gpu {_prof_g[k] / _prof_n[k]:8.1f}" if g is not None else ""
                print(f"  {k:12s} {_prof_t[k] / _prof_n[k]:8.1f}{gs} x{_prof_n[k]}", flush=True)


def _draw_skeleton_limited(frame, kpts_xy, kpts_conf=None):
    """Draw dots + limbs for KP_KEEP only. Pure overlay; tracking/logic untouched."""
    import cv2 as _cv2
    import numpy as _np
    import torch as _torch
    def _to_np(_t):
        if isinstance(_t, _torch.Tensor):
            return _t.detach().cpu().numpy()
        return _np.asarray(_t)
    xy = _to_np(kpts_xy)
    cf = _to_np(kpts_conf) if kpts_conf is not None else None
    H, W = frame.shape[0], frame.shape[1]
    ok = {}
    for _j in KP_KEEP:
        if _j >= xy.shape[0]:
            continue
        if cf is not None and _j < len(cf) and cf[_j] < KP_CONF_MIN:
            continue
        _x, _y = int(xy[_j, 0]), int(xy[_j, 1])
        if 0 <= _x < W and 0 <= _y < H:
            ok[_j] = (_x, _y)
            _cv2.circle(frame, (_x, _y), 4, (255, 255, 255), -1)
    for _e, (_a, _b) in enumerate(KP_EDGES):
        if _a in ok and _b in ok:
            _cv2.line(frame, ok[_a], ok[_b], KP_PALETTE[_e % len(KP_PALETTE)], 2)
    return frame


def _box_median_depth(depth, x1, y1, x2, y2, W, H, inset=DEPTH_BOX_INSET):
    """Median depth within a box, inset from edges to avoid boundary noise."""
    if depth is None:
        return -1.0
    # Clip coordinates to image bounds
    x1 = int(max(0, min(W - 1, x1)))
    y1 = int(max(0, min(H - 1, y1)))
    x2 = int(max(0, min(W - 1, x2)))
    y2 = int(max(0, min(H - 1, y2)))
    
    # Apply inset
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return -1.0
        
    dx = int(w * inset)
    dy = int(h * inset)
    x1i = x1 + dx
    y1i = y1 + dy
    x2i = x2 - dx
    y2i = y2 - dy
    
    # Validate inset box
    if x1i >= x2i or y1i >= y2i:
        return -1.0
        
    # Extract ROI and compute median
    roi = depth[y1i:y2i+1, x1i:x2i+1]
    if roi.size == 0:
        return -1.0
    return float(np.median(roi))

_IR_PAD_ACTIVE = False  # set True only around the detect track() call; person tracker uses original IoU
if not getattr(_ir_matching_mod.iou_distance, "_ir_pad_wrapped", False):
    _ir_orig_iou_distance = _ir_matching_mod.iou_distance

    def _ir_pad_iou_distance(atracks, btracks):
        if not _IR_PAD_ACTIVE:
            return _ir_orig_iou_distance(atracks, btracks)
        vw = float(globals().get("video_width", 0) or 0) or 1e9
        vh = float(globals().get("video_height", 0) or 0) or 1e9
        def _expand(items):
            boxes = []
            for t in items:
                if hasattr(t, "xyxy") and getattr(t, "angle", None) is None:
                    b = np.asarray(t.xyxy, dtype=np.float32).ravel()
                    if b.size != 4:
                        return None
                    boxes.append(np.array([
                        max(0.0, b[0] - pad_w), max(0.0, b[1] - pad_h),
                        min(vw, b[2] + pad_w), min(vh, b[3] + pad_h),
                    ], dtype=np.float32))
                else:
                    return None
            return boxes
        ea = _expand(atracks) if atracks else []
        eb = _expand(btracks) if btracks else []
        if ea is None or eb is None:
            return _ir_orig_iou_distance(atracks, btracks)
        return _ir_orig_iou_distance(ea, eb)

    _ir_pad_iou_distance._ir_pad_wrapped = True
    _ir_matching_mod.iou_distance = _ir_pad_iou_distance

# ---------------------------------------------------------------------------
# Model cache (UI optimisation: load once, not per video)
# ---------------------------------------------------------------------------
_detect_model = None
_pose_model = None
_depth_model = None
_depth_processor = None
_detect_classes = None
_detect_idx_to_target = {}
_pose_classes = None
_depth_ok = False
_depth_device = None
_depth_dtype = None
_depth_ms = 0.0
_depth_n = 0


def ensure_models_loaded():
    """Load YOLO models once; return (detect_model, pose_model)."""
    global _detect_model, _pose_model, _depth_model, _depth_processor, _detect_classes, _detect_idx_to_target, _pose_classes, _depth_ok, _depth_device, _depth_dtype
    if _detect_model is None:
        _detect_model = YOLO(str(BASE_DIR / detect_model_name))
        _mnames_lc = {int(k): str(v).lower() for k, v in _detect_model.names.items()}
        _mname_to_idx = {v: k for k, v in _mnames_lc.items()}
        _alias_rev = {v: k for k, v in detect_aliases.items()}
        _detect_classes = []
        for _c in detect_target_names:
            _look = str(_alias_rev.get(_c, _c)).lower()
            if _look in _mname_to_idx:
                _ci = _mname_to_idx[_look]
                _detect_classes.append(_ci)
                _detect_idx_to_target[_ci] = _c
            else:
                print(f"[hoid] WARNING: '{_c}' not in detect model names — skipped.")
    if _pose_model is None:
        _pose_model = YOLO(str(BASE_DIR / pose_model_name))
        _pose_classes = [k for k, v in _pose_model.names.items()
                         if str(v).lower() == "person"]
    
    # Load Depth Anything V2 model if enabled
    if DEPTH_USE and _depth_model is None:
        try:
            _depth_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            _depth_dtype = torch.float16 if (_depth_device.type == "cuda") else torch.float32
            _depth_model = AutoModelForDepthEstimation.from_pretrained(
                DEPTH_MODEL_ID, 
                cache_dir=str(BASE_DIR / DEPTH_CACHE_DIR),
                local_files_only=True
            ).to(_depth_device, dtype=_depth_dtype).eval()
            _depth_ok = True
            print(f"[hoid] depth: {DEPTH_MODEL_ID} on {_depth_device} ({_depth_dtype}), gate={DEPTH_GATE}")
        except Exception as e:
            print(f"[hoid] WARNING: Failed to load depth model: {e}")
            _depth_ok = False
    
    return _detect_model, _pose_model


# ---------------------------------------------------------------------------
# Summary formatting — single source of truth for live pane + final file.
# Unified to (f2 - f1) >= 30 (notebook CSV used >=30, readable used >30).
# ---------------------------------------------------------------------------
def format_summary(interactions_summary):
    lines = []
    for (k1, kc, k2), intervals in sorted(interactions_summary.items()):
        kept = [[f1, f2] for f1, f2 in intervals if (f2 - f1) >= 30]
        if not kept:
            continue
        lines.append(f"Human {k1} and {kc} {k2}")
        for f1, f2 in kept:
            lines.append(f"    Frame {f1:5d} to Frame {f2:5d}")
    return "\n".join(lines) if lines else "(no sustained interactions yet — need >=30 frame span)"


# ---------------------------------------------------------------------------
# Queue / preview helpers (additive only; never block inference)
# ---------------------------------------------------------------------------
def _preview_on(is_preview_on):
    if is_preview_on is None:
        return False
    try:
        import threading as _th
        if isinstance(is_preview_on, _th.Event):
            return is_preview_on.is_set()
    except Exception:
        pass
    if isinstance(is_preview_on, bool):
        return is_preview_on
    if callable(is_preview_on):
        try:
            return bool(is_preview_on())
        except Exception:
            return False
    return False


def _emit_drop(q, msg):
    """Non-blocking send; drop if UI is behind."""
    if q is None:
        return
    try:
        q.put_nowait(msg)
    except Exception:
        pass  # queue full -> skip frame, inference continues at full speed


def _emit_force(q, msg):
    """Deliver terminal messages even if queue is full (drop oldest first)."""
    if q is None:
        return
    try:
        q.put_nowait(msg)
    except Exception:
        try:
            q.get_nowait()
        except Exception:
            pass
        try:
            q.put_nowait(msg)
        except Exception:
            pass


def _thumbnail_rgb(bgr_frame):
    return _thumbnail_fit(bgr_frame, PREVIEW_MAX_W, PREVIEW_MAX_H)


def _thumbnail_fit(bgr_frame, box_w, box_h):
    """Single aspect-preserving resize of bgr_frame to fit (box_w, box_h)."""
    h, w = bgr_frame.shape[:2]
    try:
        box_w, box_h = int(box_w), int(box_h)
    except Exception:
        box_w, box_h = PREVIEW_MAX_W, PREVIEW_MAX_H
    if box_w <= 0 or box_h <= 0:
        box_w, box_h = PREVIEW_MAX_W, PREVIEW_MAX_H
    if w <= 0 or h <= 0:
        return cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    scale = min(box_w / float(w), box_h / float(h), 1.0)
    if scale < 1.0:
        thumb = cv2.resize(bgr_frame, (max(1, int(w * scale)), max(1, int(h * scale))),
                           interpolation=cv2.INTER_LINEAR)
    else:
        thumb = bgr_frame
    return cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Async video writer: cv2.VideoWriter owned by a worker thread.
# Main loop enqueues annotated frames (blocking put: never drops,
# backpressures instead); close() flushes, joins and releases.
# ---------------------------------------------------------------------------
class _AsyncVideoWriter:
    def __init__(self, path, fourcc, fps, wh, queue_size=30):
        self._q = queue_mod.Queue(maxsize=queue_size)
        self._err = None
        self._w = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*fourcc), fps, wh)
        self._opened = self._w.isOpened()
        if self._opened:
            self._t = threading.Thread(target=self._drain, daemon=True)
            self._t.start()

    def isOpened(self):
        return self._opened

    def write_async(self, frame):
        self._q.put(frame)

    def _drain(self):
        try:
            while True:
                f = self._q.get()
                if f is None:
                    break
                self._w.write(f)
        except Exception as e:
            self._err = e
        finally:
            self._w.release()

    def close(self):
        self._q.put(None)
        self._t.join()
        if self._err is not None:
            print(f"[hoid] writer thread error: {self._err}")


# ---------------------------------------------------------------------------
# Main function (Cell 11 logic + additive UI hooks)
# ---------------------------------------------------------------------------
def process_video(video_path, q=None, stop_event=None, is_preview_on=None, preview_box=None,
                    active_classes=None):
    """Process one video. Same outputs as notebook; optionally streams UI msgs.

    Messages pushed to q (all put_nowait, drop-if-full except terminal):
      ("meta", video_w, video_h, fps, total_frames)
      ("frame", frame_index, total_frames, thumb_rgb_or_None, elapsed_s)
      ("summary", text)
      ("done", {"output_video": str, "summary_text": str, "stopped": bool})
      ("error", message)

    preview_box: optional (box_w, box_h) aspect-fit target for thumbnails,
      snapshotted once here (safe: written before the worker starts).
    """
    global video_width, video_height, _IR_PAD_ACTIVE

    video_path = Path(video_path)
    video_input_directory.mkdir(parents=True, exist_ok=True)
    video_output_directory.mkdir(parents=True, exist_ok=True)
    logs_frame_directory.mkdir(parents=True, exist_ok=True)
    logs_summary_directory.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    _prof_t.clear()
    _prof_n.clear()
    video_name = video_path.stem

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        msg = f"Could not open video: {video_path}"
        print(msg)
        _emit_force(q, ("error", msg))
        return None

    video_fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    if total_frames <= 0:
        total_frames = 1

    # Snapshot the display box once (UI thread wrote it before starting us).
    try:
        if preview_box is not None:
            _box_w, _box_h = int(preview_box[0]), int(preview_box[1])
            if _box_w <= 0 or _box_h <= 0:
                raise ValueError
        else:
            raise ValueError
    except Exception:
        _box_w, _box_h = fit_display_box(video_width, video_height)

    if q is not None:
        _emit_drop(q, ("meta", video_width, video_height, video_fps, total_frames))

    output_video = video_output_directory / f"{video_name}_output.mp4"
    fourcc = "mp4v"
    writer = _AsyncVideoWriter(
        str(output_video),
        fourcc,
        video_fps,
        (video_width, video_height),
    )
    if not writer.isOpened():
        msg = f"Failed to initialize VideoWriter for {output_video}"
        print(msg)
        cap.release()
        _emit_force(q, ("error", msg))
        return None

    hoid_log_path = logs_frame_directory / f"{video_name}_interaction_log.csv"
    hoid_log_summary_path = logs_summary_directory / f"{video_name}_summary_log.csv"
    hoid_log_readable_path = logs_frame_directory / f"{video_name}_interaction_readable_log.txt"
    hoid_log_summary_readable_path = logs_summary_directory / f"{video_name}_summary_readable_log.txt"

    detect_model, pose_model = ensure_models_loaded()

    # Resolve active classes for this run (None = all mapped classes).
    if active_classes is None:
        use_map = dict(_detect_idx_to_target)
    else:
        _want = {str(t).lower() for t in active_classes}
        use_map = {ci: tn for ci, tn in _detect_idx_to_target.items()
                   if tn.lower() in _want}
    use_classes = list(use_map)
    use_confs = {tn: detect_conf_by_class.get(tn, detect_conf_thres)
                 for tn in use_map.values()}
    track_conf = min(use_confs.values()) if use_confs else detect_conf_thres
    if not use_classes:
        print("[hoid] WARNING: active_classes matched no model classes — output will be empty.")

    # Depth model handling
    depth_model = _depth_model if DEPTH_USE and _depth_ok else None
    depth_device = _depth_device if DEPTH_USE and _depth_ok else None
    depth_dtype = _depth_dtype if DEPTH_USE and _depth_ok else None
    if DEPTH_USE and depth_model is not None:
        _depth_ms, _depth_n = 0.0, 0
        _vetoes, _veto_gaps = 0, []

    stopped = False
    result = None
    try:
        with (
            open(hoid_log_path, "w") as hoid_log,
            open(hoid_log_summary_path, "w") as hoid_log_summary,
            open(hoid_log_readable_path, "w") as hoid_log_readable,
            open(hoid_log_summary_readable_path, "w") as hoid_log_summary_readable,
        ):
            hoid_log.write("frame_index,human_id,object_class,object_id,wrist_depth,object_depth,depth_gap,torso_depth,wrist_in_raw\n")
            hoid_log_summary.write("frame_start,frame_end,human_id,object_class,object_id\n")

            interactions_summary = {}
            last_frame_emit = 0.0
            last_summary_emit = 0.0
            last_summary_text = ""

            frame_index = 0
            current_frame = None
            while True:
                if stop_event is not None and stop_event.is_set():
                    stopped = True
                    break
                if not cap.grab():
                    break
                on_stride = frame_index % video_stride == 0

                interactions = []

                if on_stride:
                    success, frame = cap.retrieve()
                    if not success:
                        frame = None
                    current_frame = frame

                    if current_frame is None:
                        frame_index += 1
                        continue

                        # Pose FIRST so person_wrists is fresh for the interaction rescue hook
                    _pt = _prof_mark(); _pev = _prof_ev()
                    pose_results = pose_model.track(
                        device=using_gpu,
                        source=current_frame,
                        tracker=str(BASE_DIR / person_botsort),
                        persist=True,
                        conf=pose_conf_thres,
                        verbose=False,
                        iou=0.45,
                        classes=_pose_classes,
                        quantize=use_half,
                    )
                    _prof_tick("pose_track", _pt, _pev)
                    _pt = _prof_mark(); _pev = _prof_ev()
                    _IR_PAD_ACTIVE = True  # pad-IoU only for detect matching
                    detect_results = detect_model.track(
                        device=using_gpu,
                        source=current_frame,
                        tracker=str(BASE_DIR / phone_botsort),
                        persist=True,
                        conf=track_conf,
                        verbose=False,
                        classes=use_classes,
                        iou=0.45,
                        quantize=use_half,
                    )
                    _IR_PAD_ACTIVE = False
                    _prof_tick("det_track", _pt, _pev)
                    
                    # Depth processing (if enabled)
                    if DEPTH_USE and depth_model is not None:
                        _dt0 = _prof_mark()
                        try:
                            # Preprocess frame for depth model
                            _rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
                            _small = cv2.resize(_rgb, (480, 480))  # Default size for Depth Anything V2
                            _rgb_small = _small.astype(np.float32) / 255.0
                            _rgb_small -= np.array([0.485, 0.456, 0.406])
                            _rgb_small /= np.array([0.229, 0.224, 0.225])
                            _rgb_small = _rgb_small.transpose(2, 0, 1)
                            _rgb_tensor = torch.from_numpy(_rgb_small).unsqueeze(0).to(depth_device, dtype=depth_dtype)
                            
                            with torch.no_grad():
                                _dpred = depth_model(pixel_values=_rgb_tensor).predicted_depth[0]
                            
                            # Resize depth prediction to original frame size
                            _dpred = torch.nn.functional.interpolate(
                                _dpred.unsqueeze(0).unsqueeze(0),
                                size=(current_frame.shape[0], current_frame.shape[1]),
                                mode="bicubic",
                                align_corners=False
                            )[0, 0]
                            depth_np = _dpred.float().cpu().numpy()
                            _prof_tick("depth", _dt0, _prof_ev())
                        except Exception as e:
                            print(f"[hoid] Depth processing error: {e}")
                            depth_np = None
                    else:
                        depth_np = None
                    depth_info = {}  # Initialize depth info dictionary

                    d_result = detect_results[0]
                    p_result = pose_results[0]

                    d_boxes = d_result.boxes
                    p_boxes = p_result.boxes
                    # Always defined: drawing gates below test `is not None`
                    # (empty-filter runs, e.g. phone-only with no phones, hit neither branch).
                    d_xyxys = p_xyxys = None
                    d_clss = d_track_ids = None

                    if d_boxes is not None and p_boxes is not None and len(d_boxes) > 0 and len(p_boxes) > 0:
                        d_xyxys = d_boxes.xyxy.clone()
                        d_xyxys[:, 0] -= pad_w
                        d_xyxys[:, 2] += pad_w
                        d_xyxys[:, 1] -= pad_h
                        d_xyxys[:, 3] += pad_h
                        d_xyxys[:, [0, 2]] = d_xyxys[:, [0, 2]].clamp(min=0, max=video_width)
                        d_xyxys[:, [1, 3]] = d_xyxys[:, [1, 3]].clamp(min=0, max=video_height)
                        d_clss = d_boxes.cls.int()
                        d_track_ids = d_boxes.id.int() if d_boxes.id is not None else torch.zeros(
                            len(d_boxes), dtype=torch.int32, device=d_xyxys.device)
                        # Per-class output thresholds (tracker ran at the min for continuity)
                        _keep = _conf_keep_mask(d_clss, d_boxes.conf, use_map, d_xyxys.device)
                        d_xyxys, d_clss, d_track_ids = d_xyxys[_keep], d_clss[_keep], d_track_ids[_keep]

                        p_xyxys = p_boxes.xyxy.to(d_xyxys.device)
                        p_track_ids = p_boxes.id.int() if p_boxes.id is not None else torch.zeros(
                            len(p_boxes), dtype=torch.int32, device=d_xyxys.device)
                        kpts = p_result.keypoints.xy.to(d_xyxys.device) if p_result.keypoints is not None else None

                    # HOI detection block
                    _pt = _prof_mark(); _pev = _prof_ev()
                    obj_area = (d_xyxys[:, 2] - d_xyxys[:, 0]) * (d_xyxys[:, 3] - d_xyxys[:, 1])
                    xi1 = torch.maximum(p_xyxys[:, None, 0], d_xyxys[None, :, 0])
                    yi1 = torch.maximum(p_xyxys[:, None, 1], d_xyxys[None, :, 1])
                    xi2 = torch.minimum(p_xyxys[:, None, 2], d_xyxys[None, :, 2])
                    yi2 = torch.minimum(p_xyxys[:, None, 3], d_xyxys[None, :, 3])
                    inter_area = torch.clamp(xi2 - xi1, min=0) * torch.clamp(yi2 - yi1, min=0)

                    ioa = torch.zeros_like(inter_area)
                    valid_obj = obj_area > 0
                    ioa[:, valid_obj] = inter_area[:, valid_obj] / obj_area[valid_obj]

                    p_indices, d_indices = torch.where(ioa >= ioa_thres)

                    if kpts is not None and len(p_indices) > 0:
                        matched_keypoints = kpts[p_indices][:, [9, 10]]
                        matched_boxes = d_xyxys[d_indices]
                        kx, ky = matched_keypoints[..., 0], matched_keypoints[..., 1]
                        bx1 = matched_boxes[:, None, 0]
                        by1 = matched_boxes[:, None, 1]
                        bx2 = matched_boxes[:, None, 2]
                        by2 = matched_boxes[:, None, 3]
                        keypoints_inside_box = (kx >= bx1) & (kx <= bx2) & (ky >= by1) & (ky <= by2)
                        has_interacting_keypoint = keypoints_inside_box.any(dim=1)
                        confirmed_person_indices = p_indices[has_interacting_keypoint].cpu().tolist()
                        confirmed_object_indices = d_indices[has_interacting_keypoint].cpu().tolist()
                        p_ids_list = p_track_ids.tolist()
                        d_ids_list = d_track_ids.tolist()
                        d_cls_list = d_clss.tolist()
                        
                        # Depth-aware interaction filtering
                        filtered_interactions = []
                        for p, d in zip(confirmed_person_indices, confirmed_object_indices):
                            # Get depth values using median depth within detection box (like notebook)
                            if depth_np is not None:
                                _od = _box_median_depth(
                                    depth_np, 
                                    int(d_xyxys[d, 0].item()), 
                                    int(d_xyxys[d, 1].item()), 
                                    int(d_xyxys[d, 2].item()), 
                                    int(d_xyxys[d, 3].item()),
                                    depth_np.shape[1],  # width
                                    depth_np.shape[0],  # height
                                    DEPTH_BOX_INSET
                                )
                            else:
                                _od = -1.0
                            
                            # Find best wrist match (left or right)
                            _best_gap = float('inf')
                            _best_wrist_depth = -1.0
                            _best_wrist_raw = None
                            _wrist_in_raw = 0
                            
                            for _wi in (0, 1):  # 0=left wrist (9), 1=right wrist (10)
                                if _wi < len(matched_keypoints[p]):
                                    _wx = int(matched_keypoints[p, _wi, 0].item())
                                    _wy = int(matched_keypoints[p, _wi, 1].item())
                                    if 0 <= _wx < current_frame.shape[1] and 0 <= _wy < current_frame.shape[0]:
                                        _wdd = float(depth_np[_wy, _wx]) if depth_np is not None else -1.0
                                        _g = abs(_wdd - _od) if _wdd >= 0 and _od >= 0 else float('inf')
                                        if _g < _best_gap:
                                            _best_gap = _g
                                            _best_wrist_depth = _wdd
                                            _best_wrist_raw = _wdd
                                            
                                            # Check if wrist is inside raw detection box
                                            _raw_x1, _raw_y1, _raw_x2, _raw_y2 = d_xyxys[d, :4].int().tolist()
                                            _inraw = 1 if (_raw_x1 <= _wx <= _raw_x2 and _raw_y1 <= _wy <= _raw_y2) else 0
                                        
                            # Apply depth gating
                            if _od < 0 or _best_gap <= DEPTH_GATE:
                                # Apply torso veto
                                _td = -1.0
                                _tgap = float('inf')
                                if p_result.keypoints is not None and len(p_result.keypoints.xy) > 8:
                                    # Torso center (midpoint of shoulders 5 and 6)
                                    _tx = int((p_result.keypoints.xy[p, 5, 0] + p_result.keypoints.xy[p, 6, 0]) / 2)
                                    _ty = int((p_result.keypoints.xy[p, 5, 1] + p_result.keypoints.xy[p, 6, 1]) / 2)
                                    if 0 <= _tx < current_frame.shape[1] and 0 <= _ty < current_frame.shape[0]:
                                        _td = float(depth_np[_ty, _tx]) if depth_np is not None else -1.0
                                        _tgap = abs(_td - _od) if _td >= 0 and _od >= 0 else float('inf')
                                
                                # Apply torso veto
                                if _td >= 0 and _tgap > VETO_GATE:
                                    _vetoes += 1
                                    _veto_gaps.append(round(_tgap, 3))
                                    continue  # Skip this interaction
                                
                                # Apply wrist confidence check (if available)
                                _wconf_ok = True
                                if kpts is not None and hasattr(kpts, 'conf') and kpts.conf is not None:
                                    # Check confidence of selected wrist
                                    if _best_wrist_depth >= 0:  # We found a valid wrist
                                        wrist_idx = 0 if abs(matched_keypoints[p, 0, 0].item() - _) < abs(matched_keypoints[p, 1, 0].item() - _) else 1
                                        if wrist_idx < len(kpts.conf[p]):
                                            _wconf = float(kpts.conf[p, wrist_idx].item())
                                            _wconf_ok = _wconf >= pose_conf_thres  # Use pose confidence threshold
                                
                                if _wconf_ok:
                                    interaction_tuple = (
                                        frame_index,
                                        int(p_ids_list[p]),
                                        int(d_ids_list[d]),
                                        use_map.get(int(d_cls_list[d]),
                                            str(detect_model.names[int(d_cls_list[d])])),
                                        _best_wrist_depth, 
                                        _od, 
                                        _best_gap, 
                                        _td, 
                                        _wrist_in_raw
                                    )
                                    filtered_interactions.append(interaction_tuple)
                                    depth_info[(frame_index, int(p_ids_list[p]), int(d_ids_list[d]))] = (
                                        _best_wrist_depth, _od, _best_gap, _td, _wrist_in_raw
                                    )
                        interactions.extend(filtered_interactions)
                        _prof_tick("hoi", _pt, _pev)
                    elif d_boxes is not None and len(d_boxes) > 0:
                        # Objects visible but no person boxes: still draw object rects below.
                        # Rebuild padded object tensors for drawing only.
                        d_xyxys = d_boxes.xyxy.clone()
                        d_xyxys[:, 0] -= pad_w
                        d_xyxys[:, 2] += pad_w
                        d_xyxys[:, 1] -= pad_h
                        d_xyxys[:, 3] += pad_h
                        d_xyxys[:, [0, 2]] = d_xyxys[:, [0, 2]].clamp(min=0, max=video_width)
                        d_xyxys[:, [1, 3]] = d_xyxys[:, [1, 3]].clamp(min=0, max=video_height)
                        d_track_ids = d_boxes.id.int() if d_boxes.id is not None else torch.zeros(
                            len(d_boxes), dtype=torch.int32, device=d_xyxys.device)
                        d_clss = d_boxes.cls.int()
                        _keep = _conf_keep_mask(d_clss, d_boxes.conf, use_map)
                        d_xyxys, d_clss, d_track_ids = d_xyxys[_keep], d_clss[_keep], d_track_ids[_keep]
                        p_xyxys = None
                        p_track_ids = None
                    else:
                        d_xyxys = None

                    _pt = _prof_mark(); _pev = _prof_ev()
                    if plot_keypoints and p_result.keypoints is not None:
                        current_frame = _draw_skeleton_limited(current_frame, p_result.keypoints.xy, p_result.keypoints.conf)

                    if p_boxes is not None and len(p_boxes) > 0 and p_xyxys is not None:
                        p_ids_draw = p_track_ids.tolist()
                        for i, (x1, y1, x2, y2) in enumerate(p_xyxys.int().tolist()):
                            cv2.rectangle(img=current_frame, pt1=(x1, y1), pt2=(x2, y2),
                                          color=blue, thickness=2)
                            cv2.putText(img=current_frame, text=f"ID:{p_ids_draw[i]}",
                                        org=(x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=1.0, color=blue, thickness=2)

                    if d_boxes is not None and len(d_boxes) > 0 and d_xyxys is not None:
                        interacting_obj_ids = [interaction[2] for interaction in interactions]
                        d_ids_draw = d_track_ids.tolist()
                        d_cls_draw = d_clss.tolist()
                        for i, (x1, y1, x2, y2) in enumerate(d_xyxys.int().tolist()):
                            interacting_with_human = d_ids_draw[i] in interacting_obj_ids
                            use_rgb = green if interacting_with_human else red
                            cv2.rectangle(img=current_frame, pt1=(x1, y1), pt2=(x2, y2),
                                          color=use_rgb, thickness=2)
                            text_x = max(0, x1)
                            text_y = y1 - 10 if y1 > 30 else y1 + 30
                            obj_tname = use_map.get(int(d_cls_draw[i]), "?")
                            cv2.putText(img=current_frame, text=f"{obj_tname}:{d_ids_draw[i]}",
                                        org=(text_x, text_y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=1.0, color=class_colors.get(obj_tname, white),
                                        thickness=2)
                            if interacting_with_human:
                                interacting_humans = [inter[1] for inter in interactions
                                                      if inter[2] == d_ids_draw[i]]
                                humans_str = ",".join(map(str, interacting_humans))
                                cv2.putText(img=current_frame,
                                            text=f"Human:{humans_str} {obj_tname}:{d_ids_draw[i]}",
                                            org=(text_x, text_y + 35),
                                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                            fontScale=1.0, color=green, thickness=2)

                    _prof_tick("draw", _pt, _pev)
                    if interactions:
                        hoid_log.writelines([
                            f"{f_index},{per_id},{o_cls},{o_id},{w_depth:.3f},{o_depth:.3f},{gap:.3f},{t_depth:.3f},{w_in_raw}\n"
                            for (f_index, per_id, o_id, o_cls, w_depth, o_depth, gap, t_depth, w_in_raw) in interactions
                        ])
                        hoid_log_readable.writelines([
                            f"FRAME {frame_index:6d} : PERSON {per_id:4d} and {o_cls} {o_id:4d} (depth_gap={gap:.3f})\n"
                            for (f_index, per_id, o_id, o_cls, _, _, gap, _, _) in interactions
                        ])
                        # Stash interaction output for next frame's rescue hook
                        _IR_STATE["last_held_by"].update(
                            {d_id: p_id for f_index, p_id, d_id, o_cls, _, _, _, _, _ in interactions
                             if d_id is not None}
                        )

                # Interactions summary stitching (runs for stride AND non-stride;
                # interactions is [] off-stride so this is a no-op there — same as notebook)
                for f_index, per_id, o_id, o_cls, _, _, _, _, _ in interactions:
                    key = (per_id, o_cls, o_id)
                    if key not in interactions_summary:
                        interactions_summary[key] = [[f_index, f_index]]
                    elif f_index - interactions_summary[key][-1][1] <= frame_stitching_threshold:
                        interactions_summary[key][-1][1] = f_index
                    else:
                        interactions_summary[key].append([f_index, f_index])

                # Frame writing (every source frame, same as notebook)
                if current_frame is not None:
                    output_frame = current_frame.copy()
                    cv2.putText(img=output_frame, text=f"Frame {frame_index}", org=(30, 40),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0,
                                color=white, thickness=3)
                    _pt = _prof_mark(); _pev = _prof_ev()
                    writer.write_async(output_frame)
                    _prof_tick("write", _pt, _pev)
                else:
                    output_frame = None

                # UI stream (throttled, drop-if-full; skipped entirely when preview off
                # except for the lightweight progress/summary messages)
                if q is not None and output_frame is not None:
                    now = time.perf_counter()
                    elapsed = now - start_time
                    if now - last_frame_emit >= PREVIEW_MIN_INTERVAL_S:
                        last_frame_emit = now
                        if _preview_on(is_preview_on):
                            try:
                                thumb = _thumbnail_fit(output_frame, _box_w, _box_h)
                            except Exception:
                                thumb = None
                            _emit_drop(q, ("frame", frame_index, total_frames, thumb, elapsed))
                        else:
                            _emit_drop(q, ("progress", frame_index, total_frames, elapsed))
                    if now - last_summary_emit >= SUMMARY_MIN_INTERVAL_S:
                        text = format_summary(interactions_summary)
                        if text != last_summary_text:
                            last_summary_text = text
                            last_summary_emit = now
                            _emit_drop(q, ("summary", text))

                # Headless console progress (notebook behaviour; suppressed when UI owns q)
                if q is None:
                    progress_frame = f"{frame_index}/{total_frames}" if progress_frame_enabled else ""
                    progress_bar = ""
                    if progress_bar_enabled:
                        filled = int(progress_bar_length * frame_index // total_frames)
                        progress_bar = "[" + "#" * filled + "-" * (progress_bar_length - filled) + "]"
                    progress_percent = f"{(frame_index / total_frames) * 100:.2f}%" if progress_percent_enabled else ""
                    progress_elapsed = f"{time.perf_counter() - start_time:.2f}s" if progress_elapsed_time_enabled else ""
                    print(f"\r{video_name:<7} {progress_frame:13} {progress_bar} {progress_percent:7} {progress_elapsed:7}",
                          end="", flush=True)

                frame_index += 1

            # Write summary logs (unified >= 30 filter)
            if interactions_summary:
                hoid_log_summary.writelines([
                    f"{f1},{f2},{k1},{kc},{k2}\n"
                    for (k1, kc, k2), v in interactions_summary.items()
                    for f1, f2 in v if (f2 - f1) >= 30
                ])
                hoid_log_summary_readable.writelines([
                    item
                    for (k1, kc, k2), v in interactions_summary.items()
                    if any((f2 - f1) >= 30 for f1, f2 in v)
                    for item in [f"Human {k1} and {kc} {k2}\n"] +
                                [f"    Frame {f1:5} to Frame {f2:5}\n" for f1, f2 in v if (f2 - f1) >= 30]
                ])

            final_text = format_summary(interactions_summary)
            _prof_report()
            if q is not None:
                _emit_drop(q, ("summary", final_text))
            result = {
                "video": str(video_path),
                "output_video": str(output_video),
                "frame_log": str(hoid_log_path),
                "summary_log": str(hoid_log_summary_path),
                "summary_text": final_text,
                "stopped": stopped,
            }
            if q is not None:
                _emit_force(q, ("done", result))
    finally:
        try:
            cap.release()
        except Exception:
            pass
        try:
            writer.close()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    if q is None:
        print()
    return result


def list_input_videos():
    video_input_directory.mkdir(parents=True, exist_ok=True)
    return sorted(video_input_directory.glob("*.mp4"))
