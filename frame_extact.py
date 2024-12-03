import cv2
import os

# Path to your video file
video_path = r"C:\Users\KIIT\Desktop\CAR_vs_NYR_001.mp4"

# Directory to save extracted frames (same as video directory)
video_dir = os.path.dirname(video_path)
output_dir = os.path.join(video_dir, "frames")
os.makedirs(output_dir, exist_ok=True)

# Read the video
cap = cv2.VideoCapture(video_path)

# Get video metadata
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total frames
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
video_duration = total_frames / fps  # Video duration in seconds

print(f"Total frames: {total_frames}, FPS: {fps}, Video duration: {video_duration:.2f} seconds")

# Calculate frame interval to get exactly 100 frames
frame_interval = max(1, total_frames // 100)  # Ensure at least every frame is considered

frame_count = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break  # End of video

    # Save every calculated interval frame
    if frame_count % frame_interval == 0 and saved_count < 100:  # Ensure exactly 100 frames
        frame_filename = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        saved_count += 1

    frame_count += 1

cap.release()
print(f"Extracted {saved_count} frames and saved them in '{output_dir}'")
