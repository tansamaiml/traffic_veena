from ultralytics import YOLO
from pathlib import Path

if __name__ == "__main__":
    # ===== CONFIG =====
    data_yaml_path = Path("D:/veena_stage1_srt_yolo/data.yaml")
    pretrained_model = "yolov8n.pt"
    epochs = 50
    img_size = 640

    # Create model
    model = YOLO(pretrained_model)

    # Start training
    model.train(
        data=str(data_yaml_path),
        epochs=epochs,
        imgsz=img_size,
        batch=16,  # adjust based on GPU
        workers=4, # number of CPU workers
        name="vehicle_yolov8"
    )

    print("✅ Training complete. Model saved in runs/detect/vehicle_yolov8/")
