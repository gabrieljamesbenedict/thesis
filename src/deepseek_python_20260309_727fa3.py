import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time

# ---------------------------
# Configuration parameters
# ---------------------------
IOU_THRESH = 0.15                # IoU threshold for candidate interaction
KEYPOINT_CONF_THRESH = 0.5       # Minimum confidence for a keypoint to be considered
MARGIN = 10                       # Pixel margin around object bbox for keypoint proximity
T_START = 5                       # Frames of sustained proximity to start interaction
T_STOP = 10                       # Frames without proximity to stop interaction
REID_DIST_THRESH = 50             # Max pixel distance for re-associating lost tracks
REID_TIME_THRESH = 2.0            # Max seconds to keep a cache entry
# ---------------------------------
# Load models
# ---------------------------------
det_model = YOLO('yolov11n.pt')          # detection model (replace with your trained model)
pose_model = YOLO('yolov11n-pose.pt')    # pose model

# ---------------------------------
# ByteTrack initialisation (simplified)
# We assume a tracker that returns tracks with id, bbox, class
# For demonstration, we use a placeholder class that mimics ByteTrack.
# In practice, install boxmot (pip install boxmot) and use:
# from boxmot import ByteTrack
# tracker = ByteTrack()
# ---------------------------------
class DummyTracker:
    def __init__(self):
        self.next_id = 0
        self.tracks = {}
    def update(self, detections, frame):
        # detections: list of [x1,y1,x2,y2,conf,class_id]
        # returns list of Track objects with .id, .bbox, .class_id
        tracks = []
        for det in detections:
            track = type('Track', (), {})()
            track.id = self.next_id
            track.bbox = det[:4]
            track.class_id = int(det[5])
            track.conf = det[4]
            self.tracks[self.next_id] = track
            tracks.append(track)
            self.next_id += 1
        return tracks

tracker = DummyTracker()   # replace with actual ByteTrack

# ---------------------------------
# Data structures for interaction tracking
# ---------------------------------
interactions = {}           # key = (person_id, object_id) -> state
                            # state = {'active': bool, 'start_frame': int, 'counter': int}
cache = {}                  # key = object_id (lost) -> {'bbox': last_bbox, 'class': class_id,
                            #                             'timestamp': time.time(), 'person_id': last_person_id?}
frame_count = 0

# ---------------------------------
# Helper functions
# ---------------------------------
def compute_iou(box1, box2):
    # box format: [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def is_keypoint_near_object(keypoints, obj_bbox, margin=MARGIN):
    """
    keypoints: list of (x, y, conf) for relevant joints (e.g., wrists, head, shoulders)
    returns True if any keypoint is inside or within margin of obj_bbox
    """
    x1, y1, x2, y2 = obj_bbox
    # expand bbox by margin
    x1 -= margin; y1 -= margin; x2 += margin; y2 += margin
    for x, y, conf in keypoints:
        if conf < KEYPOINT_CONF_THRESH:
            continue
        if x1 <= x <= x2 and y1 <= y <= y2:
            return True
    return False

def extract_relevant_keypoints(pose_result):
    """
    From YOLO pose result, extract keypoints for wrists, head, torso.
    COCO keypoint indices: 0=nose, 1=left_eye,2=right_eye,3=left_ear,4=right_ear,
    5=left_shoulder,6=right_shoulder,7=left_elbow,8=right_elbow,
    9=left_wrist,10=right_wrist,11=left_hip,12=right_hip,
    13=left_knee,14=right_knee,15=left_ankle,16=right_ankle.
    We'll use wrists (9,10), shoulders (5,6), head (0-4 roughly).
    """
    if pose_result[0].keypoints is None:
        return []
    kps = pose_result[0].keypoints.data.cpu().numpy()[0]  # shape (17,3) (x,y,conf)
    relevant_idx = [0,1,2,3,4,5,6,9,10]  # head + shoulders + wrists
    relevant = [(kps[i,0], kps[i,1], kps[i,2]) for i in relevant_idx]
    return relevant

# ---------------------------------
# Main loop (video capture)
# ---------------------------------
cap = cv2.VideoCapture(0)  # or video file

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    current_time = time.time()

    # 1. Run YOLOv11 detection on full frame
    det_results = det_model(frame, verbose=False)[0]
    detections = []
    for box in det_results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        detections.append([x1, y1, x2, y2, conf, cls])

    # 2. Update tracker with detections
    tracks = tracker.update(detections, frame)

    # Separate person tracks and object tracks
    person_tracks = [t for t in tracks if t.class_id == 0]   # assuming class 0 = person
    object_tracks = [t for t in tracks if t.class_id != 0]

    # 3. For each (person, object) pair, check IoU
    for person in person_tracks:
        for obj in object_tracks:
            iou = compute_iou(person.bbox, obj.bbox)
            if iou > IOU_THRESH:
                # Crop person from frame
                x1, y1, x2, y2 = map(int, person.bbox)
                person_crop = frame[y1:y2, x1:x2]
                if person_crop.size == 0:
                    continue
                # Run pose estimation on crop
                pose_results = pose_model(person_crop, verbose=False)
                keypoints = extract_relevant_keypoints(pose_results)
                # Check if any keypoint near object
                proximity = is_keypoint_near_object(keypoints, obj.bbox, margin=MARGIN)

                # Update interaction state
                key = (person.id, obj.id)
                if key not in interactions:
                    interactions[key] = {'active': False, 'counter': 0, 'start_frame': None}

                state = interactions[key]
                if proximity:
                    state['counter'] = min(state['counter'] + 1, T_START + 1)   # cap at T_START+1
                else:
                    state['counter'] = max(state['counter'] - 1, 0)

                # Check start condition
                if not state['active'] and state['counter'] >= T_START:
                    state['active'] = True
                    state['start_frame'] = frame_count
                    print(f"INTERACTION START: person {person.id}, object {obj.id}, "
                          f"class {det_model.names[obj.class_id]}, frame {frame_count}")

                # Check stop condition (only if active)
                elif state['active'] and state['counter'] == 0:
                    state['active'] = False
                    duration = frame_count - state['start_frame']
                    print(f"INTERACTION STOP: person {person.id}, object {obj.id}, "
                          f"duration {duration} frames, frame {frame_count}")

    # 4. Clean up interactions for tracks that no longer exist (optional)
    active_keys = set((p.id, o.id) for p in person_tracks for o in object_tracks)
    to_delete = [k for k in interactions if k not in active_keys]
    for k in to_delete:
        if interactions[k]['active']:
            # Interaction ended because track lost; log end?
            # You can decide based on your requirements.
            pass
        del interactions[k]

    # 5. Cache lost object tracks for re-identification
    # Get all current object IDs
    current_obj_ids = {obj.id for obj in object_tracks}
    # For each cached object, check if it reappears
    for obj_id, cached in list(cache.items()):
        # If too old, expire
        if current_time - cached['timestamp'] > REID_TIME_THRESH:
            del cache[obj_id]
            continue
        # Try to match with a new detection (not yet tracked)
        # In a real system, you'd compare bbox distance to new detections before they enter tracker.
        # Here we'll simulate by checking if any current object has a similar bbox.
        # For simplicity, we'll just check distance between cached bbox center and current object bbox centers.
        cx_cached = (cached['bbox'][0] + cached['bbox'][2]) / 2
        cy_cached = (cached['bbox'][1] + cached['bbox'][3]) / 2
        for obj in object_tracks:
            cx_obj = (obj.bbox[0] + obj.bbox[2]) / 2
            cy_obj = (obj.bbox[1] + obj.bbox[3]) / 2
            dist = np.hypot(cx_obj - cx_cached, cy_obj - cy_cached)
            if dist < REID_DIST_THRESH:
                # Re-assign ID: we could update tracker's ID mapping, but here we just print
                print(f"RE-IDENTIFIED object {obj_id} as new track {obj.id}")
                # Optionally update any state that relied on old ID
                # For simplicity, we remove from cache and break
                del cache[obj_id]
                break

    # Update cache: for each object track that is about to be lost? Actually we need to detect lost tracks.
    # In a real system, you'd compare previous frame tracks with current tracks.
    # We'll simulate by keeping all object tracks that are currently present, and later we'll see which ones disappeared.
    # For this example, we'll just store all objects that are present now, but we should only cache when a track is lost.
    # Proper implementation would keep a history of previous tracks.
    # Let's keep it simple: after processing each frame, we note which objects are present.
    # In a more robust solution, you'd have a separate step.

    # For demonstration, we'll just cache all object tracks at the end of the frame (this would be wrong in practice).
    # Instead, we should detect lost tracks by comparing with previous frame's objects.
    # We'll leave that out for brevity.

    # 6. Display frame (optional)
    cv2.imshow('Pipeline', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()