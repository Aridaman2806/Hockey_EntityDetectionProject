import os
import cv2
from ultralytics import YOLO
import pytesseract
import numpy as np

# Specify the path to Tesseract executable if not in PATH
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Metadata dictionary for jersey numbers and corresponding player names
team_metadata = {
    str(i): f"Player {i}" for i in range(1, 89)  # Player names for jersey numbers 1–88
}

# Load the YOLO model
model = YOLO("runs/detect/hockey_detector/weights/best.pt")  # Path to your YOLO model

# Function to preprocess the ROI for better OCR results
def preprocess_roi(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Apply Gaussian Blur
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return binary

# Function to process the video and annotate frames
def process_video(input_video_path, output_video_path):
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file!")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Initialize the video writer
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection using the YOLO model
        results = model(frame)

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
            confs = result.boxes.conf.cpu().numpy()  # Confidence scores
            classes = result.boxes.cls.cpu().numpy()  # Class labels

            for box, conf, cls in zip(boxes, confs, classes):
                x1, y1, x2, y2 = map(int, box)
                label = int(cls)

                if label == 0:  # Assuming 0 is the label for players
                    # Crop the region of interest (ROI) for OCR
                    roi = frame[y1:y2, x1:x2]
                    
                    # Preprocess ROI for better OCR accuracy
                    preprocessed_roi = preprocess_roi(roi)

                    # Perform OCR using PyTesseract
                    ocr_result = pytesseract.image_to_string(preprocessed_roi, config="--psm 6").strip()

                    # Skip bounding box if OCR result is empty
                    if not ocr_result:
                        continue

                    # Extract the jersey number (if detected)
                    jersey_number = ''.join(filter(str.isdigit, ocr_result))  # Extract digits only
                    player_name = team_metadata.get(jersey_number, "Unknown Player")

                    # Annotate the frame with the player's name
                    color = (0, 255, 0)  # Green for bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{player_name} ({jersey_number})", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Write the annotated frame to the output video
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    print(f"Processed video saved to: {output_video_path}")

# Paths to the input and output videos
input_video_path = r"C:\Users\KIIT\Desktop\MEDI project\Task2\Testing_vid.mp4"  # Replace with your input video path
output_video_path = "tesseract_output_video.mp4"  # Replace with your desired output video path

# Process the video
process_video(input_video_path, output_video_path)
