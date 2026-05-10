import sys
import cv2
import os
import glob

script_base_directory = os.path.dirname(os.path.abspath(__file__))

print("Reading input videos...")
matches = glob.glob(os.path.join(script_base_directory, "video", "input", "*.mp4"))

if len(matches) == 0:
    print("No MP4 files found!")
    print(matches)
    sys.exit()
else:
    for m in matches:
        print(f"Found: {m}")

log_frames = True           # Displays current frame
log_skip_stride = 100       # Amount of steps skipped in logging for clarity
sampling_stride = 30        # Amount of frames skipped to reduce overall dataset volume

print(f"Logging Frames: {log_frames}")
print(f"Skipped Log Amount: {log_skip_stride}")
print(f"Sampling Stride: {sampling_stride}")

video_output_path = os.path.join(script_base_directory, "video", "output")
os.makedirs(video_output_path, exist_ok=True)
for file in os.listdir(video_output_path):
    file_path = os.path.join(video_output_path, file)
    if os.path.isfile(file_path):
        os.remove(file_path)

total_frame_count = 0
total_extracted_frames = 0
total_dropped_frames = 0

for video in matches:
    video_input_path = video
    
    cap = cv2.VideoCapture(video_input_path)
    if not cap.isOpened():
        raise Exception("Error: Cannot open video file")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print()
    print("VIDEO DETAILS")
    print(f"Name: {video}")
    print("Width: "  + str(width))
    print("Height: "  + str(height))
    print("FPS: "  + str(fps))

    frame_count = 0
    extracted_frames = 0
    dropped_frames = 0

    while True:
        ret = cap.grab()
        if not ret:
            break

        if frame_count % sampling_stride != 0:
            dropped_frames += 1
            frame_count += 1
            continue

        ret, frame = cap.retrieve()
        if not ret:
            break
        
        frame_number = extracted_frames + total_extracted_frames
        filename = os.path.join(video_output_path, f"FRAME_{frame_number:010d}.jpg")
        cv2.imwrite(filename, frame)

        if log_frames and extracted_frames % log_skip_stride == 0:
            print(f"Converted: {filename}")

        extracted_frames += 1
        frame_count += 1

    total_frame_count += frame_count
    total_extracted_frames += extracted_frames
    total_dropped_frames += dropped_frames

    cap.release()

    print(f"Completed conversion of {video}")
    print(f"Found {frame_count} frames")
    print(f"Extracted {extracted_frames} frames")
    print(f"Dropped {dropped_frames} frames")
    print(f"{float(extracted_frames/frame_count*100)}% of frames have been extracted")

print(f"Completed conversion of all videos")
for video in matches:
    print(video)
print(f"Found {total_frame_count} total frames")
print(f"Extracted {total_extracted_frames} total frames")
print(f"Dropped {total_dropped_frames} total frames")