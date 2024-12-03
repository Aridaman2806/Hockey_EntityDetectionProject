from ultralytics import YOLO

# Load the YOLOv8 pre-trained model
model = YOLO("yolov8n.pt")  # Use yolov8n.pt, yolov8s.pt, etc., based on your hardware

# Train the model
model.train(
    data="dataset.yaml",   # Path to the dataset YAML file
    epochs=50,             # Number of epochs
    imgsz=640,             # Image size (default is 640x640)
    batch=16,              # Batch size (adjust based on GPU memory)
    workers=4,             # Number of dataloader workers
    name="hockey_detector" # Name of the experiment
)

# Validate the model
metrics = model.val(data="dataset.yaml")  # Validate the model
print(metrics)  # Print validation metrics

# Run inference on an image
results = model.predict(source="dataset/val/images/img1.jpg", conf=0.25, show=True)

# Export the trained model to ONNX format
model.export(format="onnx")
