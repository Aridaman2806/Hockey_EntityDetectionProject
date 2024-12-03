from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO("runs/detect/hockey_detector/weights/best.pt")  # Path to your trained weights

# Run inference on a video and save the output in the same directory as the script
results = model.predict(
    source=r"C:\Users\KIIT\Desktop\CAR_vs_NYR_001.mp4",  # Path to the input video
    conf=0.25,  # Confidence threshold
    save=True,  # Save the output video with detections
    save_dir=".",  # Save the output video in the current directory
    show=False  # Set to True if you want to display the video while processing
)

# Print the path to the saved video
print(f"Detection results saved at: {results[0].path}")
