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
import time
import torch

from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Paths / config (verbatim from notebook Cells 5-7)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

detect_model_name = "models/detect/yolo11x-trained.pt"
pose_model_name = "models/pose/yolo11x-pose-trained.pt"
phone_botsort = "trackers/phone_botsort.yaml"
person_botsort = "trackers/person_botsort.yaml"

video_input_directory = BASE_DIR / "video" / "input"
video_output_directory = BASE_DIR / "video" / "output"
logs_frame_directory = BASE_DIR / "video" / "logs" / "logs_frames"
logs_summary_directory = BASE_DIR / "video" / "logs" / "logs_summary"

# Systems configuration (Cell 7)
detect_conf_thres = 0.2
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

# Colors (BGR, as used by cv2 in notebook)
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
white = (255, 255, 255)

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
                    stash = {}
                    for j in range(len(xyxys)):
                        key = int(tids[j].item()) if tids is not None else "idx" + str(j)
                        stash[key] = xyxys[j]
                    _IR_STATE["person_boxes"] = stash
                    if results[0].keypoints is not None:
                        kxy = results[0].keypoints.xy.detach().cpu().numpy()
                        wr = {}
                        for j in range(len(xyxys)):
                            key = int(tids[j].item()) if tids is not None else "idx" + str(j)
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
            for a, trk in enumerate(rescue_trks):
                owner_key = _IR_STATE["last_held_by"].get(trk.track_id)
                if owner_key is None:
                    continue
                ref_entry = _IR_STATE["last_box"].get(trk.track_id)
                ref_box = ref_entry[0] if ref_entry is not None else np.asarray(trk.xyxy, dtype=float)
                for b, det in enumerate(det_objs):
                    dbox = np.asarray(det.xyxy, dtype=float)
                    holder = _ir_wrist_holder(dbox)
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

if not getattr(_ir_matching_mod.iou_distance, "_ir_pad_wrapped", False):
    _ir_orig_iou_distance = _ir_matching_mod.iou_distance

    def _ir_pad_iou_distance(atracks, btracks):
        def _expand(items):
            boxes = []
            for t in items:
                if hasattr(t, "xyxy") and getattr(t, "angle", None) is None:
                    b = np.asarray(t.xyxy, dtype=np.float32).ravel()
                    if b.size != 4:
                        return None
                    vw = float(globals().get("video_width", 0) or 0) or 1e9
                    vh = float(globals().get("video_height", 0) or 0) or 1e9
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
_detect_classes = None
_pose_classes = None


def ensure_models_loaded():
    """Load YOLO models once; return (detect_model, pose_model)."""
    global _detect_model, _pose_model, _detect_classes, _pose_classes
    if _detect_model is None:
        _detect_model = YOLO(str(BASE_DIR / detect_model_name))
        _detect_classes = [k for k, v in _detect_model.names.items()
                           if str(v).lower() == "phone"]
    if _pose_model is None:
        _pose_model = YOLO(str(BASE_DIR / pose_model_name))
        _pose_classes = [k for k, v in _pose_model.names.items()
                         if str(v).lower() == "person"]
    return _detect_model, _pose_model


# ---------------------------------------------------------------------------
# Summary formatting — single source of truth for live pane + final file.
# Unified to (f2 - f1) >= 30 (notebook CSV used >=30, readable used >30).
# ---------------------------------------------------------------------------
def format_summary(interactions_summary):
    lines = []
    for (k1, k2), intervals in sorted(interactions_summary.items()):
        kept = [[f1, f2] for f1, f2 in intervals if (f2 - f1) >= 30]
        if not kept:
            continue
        lines.append(f"Human {k1} and Object {k2}")
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
# Main function (Cell 11 logic + additive UI hooks)
# ---------------------------------------------------------------------------
def process_video(video_path, q=None, stop_event=None, is_preview_on=None, preview_box=None):
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
    global video_width, video_height

    video_path = Path(video_path)
    video_input_directory.mkdir(parents=True, exist_ok=True)
    video_output_directory.mkdir(parents=True, exist_ok=True)
    logs_frame_directory.mkdir(parents=True, exist_ok=True)
    logs_summary_directory.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
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
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*fourcc),
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

    stopped = False
    result = None
    try:
        with (
            open(hoid_log_path, "w") as hoid_log,
            open(hoid_log_summary_path, "w") as hoid_log_summary,
            open(hoid_log_readable_path, "w") as hoid_log_readable,
            open(hoid_log_summary_readable_path, "w") as hoid_log_summary_readable,
        ):
            hoid_log.write("frame_index,human_id,object_id\n")
            hoid_log_summary.write("frame_start,frame_end,human_id,object_id\n")

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

                    # Pose FIRST so person_wrists is fresh for the phone hook
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
                    detect_results = detect_model.track(
                        device=using_gpu,
                        source=current_frame,
                        tracker=str(BASE_DIR / phone_botsort),
                        persist=True,
                        conf=detect_conf_thres,
                        verbose=False,
                        classes=_detect_classes,
                        iou=0.45,
                        quantize=use_half,
                    )

                    d_result = detect_results[0]
                    p_result = pose_results[0]

                    d_boxes = d_result.boxes
                    p_boxes = p_result.boxes

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

                        p_xyxys = p_boxes.xyxy.to(d_xyxys.device)
                        p_track_ids = p_boxes.id.int() if p_boxes.id is not None else torch.zeros(
                            len(p_boxes), dtype=torch.int32, device=d_xyxys.device)
                        kpts = p_result.keypoints.xy.to(d_xyxys.device) if p_result.keypoints is not None else None

                        # HOI detection block
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
                            confirmed_person_indices = p_indices[has_interacting_keypoint].cpu()
                            confirmed_object_indices = d_indices[has_interacting_keypoint].cpu()
                            interactions.extend([
                                (frame_index, p_track_ids[p], d_track_ids[d],
                                 detect_model.names[d_clss[d].item()])
                                for p, d in zip(confirmed_person_indices, confirmed_object_indices)
                            ])
                    elif d_boxes is not None and len(d_boxes) > 0:
                        # Phones visible but no person boxes: still draw phone rects below.
                        # Rebuild padded phone tensors for drawing only.
                        d_xyxys = d_boxes.xyxy.clone()
                        d_xyxys[:, 0] -= pad_w
                        d_xyxys[:, 2] += pad_w
                        d_xyxys[:, 1] -= pad_h
                        d_xyxys[:, 3] += pad_h
                        d_xyxys[:, [0, 2]] = d_xyxys[:, [0, 2]].clamp(min=0, max=video_width)
                        d_xyxys[:, [1, 3]] = d_xyxys[:, [1, 3]].clamp(min=0, max=video_height)
                        d_track_ids = d_boxes.id.int() if d_boxes.id is not None else torch.zeros(
                            len(d_boxes), dtype=torch.int32, device=d_xyxys.device)
                        p_xyxys = None
                        p_track_ids = None
                    else:
                        d_xyxys = None

                    if plot_keypoints:
                        try:
                            current_frame = p_result.plot(boxes=False)
                        except Exception:
                            pass

                    if p_boxes is not None and len(p_boxes) > 0 and p_xyxys is not None:
                        for i, (x1, y1, x2, y2) in enumerate(p_xyxys.int().tolist()):
                            cv2.rectangle(img=current_frame, pt1=(x1, y1), pt2=(x2, y2),
                                          color=blue, thickness=2)
                            cv2.putText(img=current_frame, text=f"ID:{p_track_ids[i]}",
                                        org=(x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=1.0, color=blue, thickness=2)

                    if d_boxes is not None and len(d_boxes) > 0 and d_xyxys is not None:
                        interacting_obj_ids = [interaction[2] for interaction in interactions]
                        for i, (x1, y1, x2, y2) in enumerate(d_xyxys.int().tolist()):
                            interacting_with_human = d_track_ids[i] in interacting_obj_ids
                            use_rgb = green if interacting_with_human else red
                            cv2.rectangle(img=current_frame, pt1=(x1, y1), pt2=(x2, y2),
                                          color=use_rgb, thickness=2)
                            text_x = max(0, x1)
                            text_y = y1 - 10 if y1 > 30 else y1 + 30
                            cv2.putText(img=current_frame, text=f"ID:{d_track_ids[i]}",
                                        org=(text_x, text_y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=1.0, color=red, thickness=2)
                            if interacting_with_human:
                                interacting_humans = [inter[1].item() for inter in interactions
                                                      if inter[2] == d_track_ids[i]]
                                humans_str = ",".join(map(str, interacting_humans))
                                cv2.putText(img=current_frame,
                                            text=f"Human:{humans_str} Object:{d_track_ids[i]}",
                                            org=(text_x, text_y + 35),
                                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                            fontScale=1.0, color=green, thickness=2)

                    if interactions:
                        hoid_log.writelines([
                            f"{f_index},{per_id},{o_id}\n"
                            for f_index, per_id, o_id, _ in interactions
                        ])
                        hoid_log_readable.writelines([
                            f"FRAME {frame_index:6d} : PERSON {per_id:4d} and {obj_class} {o_id:4d}\n"
                            for f_index, per_id, o_id, obj_class in interactions
                        ])

                # Interactions summary stitching (runs for stride AND non-stride;
                # interactions is [] off-stride so this is a no-op there — same as notebook)
                for f_index, per_id, o_id, _ in interactions:
                    key = (per_id.item(), o_id.item())
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
                    writer.write(output_frame)
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
                    f"{f1},{f2},{k1},{k2}\n"
                    for (k1, k2), v in interactions_summary.items()
                    for f1, f2 in v if (f2 - f1) >= 30
                ])
                hoid_log_summary_readable.writelines([
                    item
                    for (k1, k2), v in interactions_summary.items()
                    if any((f2 - f1) >= 30 for f1, f2 in v)
                    for item in [f"Human {k1} and Object {k2}\n"] +
                                [f"    Frame {f1:5} to Frame {f2:5}\n" for f1, f2 in v if (f2 - f1) >= 30]
                ])

            final_text = format_summary(interactions_summary)
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
            writer.release()
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
