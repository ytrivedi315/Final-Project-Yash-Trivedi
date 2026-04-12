
### Purpose of program is to extract frames from video "boiling-91" which are aligned with temperature datapoints

import cv2
import pandas as pd
import os

video_path = 'Boiling-91_video.mp4'
csv_path = 'Boiling-91_aligned_frames.csv'
output_dir = 'Boiling-91_frames'

os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(csv_path)
cap = cv2.VideoCapture(video_path)

for i, row in df.iterrows():
    frame_idx = int(row['frame_index'])
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if ret:
        filename = os.path.join(output_dir, f'frame_{i:04d}.png')
        cv2.imwrite(filename, frame)

cap.release()
print('Done extracting frames.')
