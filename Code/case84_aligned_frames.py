
### Purpose of program is to extract frames from video "boiling-84" which are aligned with temperature datapoints

import os
import cv2
import pandas as pd

VIDEO_PATH = "Boiling-84_video.mp4"
ALIGNMENT_PATH = "Boiling-84_aligned_frames.csv"
OUTPUT_DIR = "Boiling-84_frames"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(ALIGNMENT_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

saved = 0
for i, row in df.iterrows():
    frame_idx = int(row["frame_index_round"])
    frame_idx = max(0, min(frame_idx, n_frames - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        print(f"Failed at row {i}, frame {frame_idx}")
        continue

    # Keep filenames ordered and tied to the thermal sample
    filename = os.path.join(OUTPUT_DIR, f'frame_{i:04d}.png')
    cv2.imwrite(filename,frame)
    saved += 1

cap.release()
print(f"Saved {saved} aligned frames to {OUTPUT_DIR}")
