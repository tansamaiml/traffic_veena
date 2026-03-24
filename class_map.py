from ultralytics import YOLO
model = YOLO(r"C:\train_trf\train_v2\weights\Traffic_new.pt")
print(model.names)
