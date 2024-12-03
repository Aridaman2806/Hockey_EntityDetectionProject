from ultralytics import YOLO

model = YOLO("runs/detect/hockey_detector/weights/best.pt")

results = model.predict(
    source=r"C:\Users\KIIT\Desktop\CAR_vs_NYR_001.mp4",
    conf=0.5,
    save=True,
    save_dir=".",
    show=False
)

print(f"Detection results saved at: {results[0].path}")
