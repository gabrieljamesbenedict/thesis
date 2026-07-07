import streamlit as st
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO

uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    
    detect_model = YOLO("models/11L3.pt")
    pose_model = YOLO("yolo11l-pose")
    custom_tracker = "trackers/custom_botsort.yaml"
    
    processed_viewport = st.empty()
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        detect_results = detect_model.track(
            device=0,
            source=frame,
            tracker=custom_tracker,
            persist=True,
            conf=0.5,
            verbose=False
        )
        
        pose_results = pose_model.track(
            device=0,
            source=frame,
            tracker=custom_tracker,
            persist=True,
            conf=0.5,
            verbose=False
        )
        
        current_frame = pose_results[0].plot(boxes=False)
        d_boxes = detect_results[0].boxes
        
        if len(d_boxes) > 0:
            d_xyxys = d_boxes.xyxy.cpu().numpy()
            d_clss = d_boxes.cls.cpu().numpy().astype(int)
            d_confs = d_boxes.conf.cpu().numpy()
            d_track_ids = d_boxes.id.cpu().numpy().astype(int) if d_boxes.id is not None else [0] * len(d_boxes)
            
            for i, (x1, y1, x2, y2) in enumerate(d_xyxys):
                cv2.rectangle(current_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                label = f"ID:{d_track_ids[i]} CLS:{detect_model.names[d_clss[i]]} CONF:{round(float(d_confs[i]), 2)}"
                cv2.putText(current_frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
                
        p_boxes = pose_results[0].boxes
        p_xyxys = p_boxes.xyxy.cpu().numpy()
        p_confs = p_boxes.conf.cpu().numpy()
        p_track_ids = p_boxes.id.cpu().numpy().astype(int) if p_boxes.id is not None else [0] * len(p_boxes)
        
        for i, (x1, y1, x2, y2) in enumerate(p_xyxys):
            cv2.rectangle(current_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            label = f"ID:{p_track_ids[i]} CLS:person CONF:{round(float(p_confs[i]), 2)}"
            cv2.putText(current_frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
        annotated_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
        processed_viewport.image(annotated_rgb, channels="RGB")
        
    cap.release()