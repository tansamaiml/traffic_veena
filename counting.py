import os
import cv2
import torch
from datetime import timedelta, datetime
from ultralytics import YOLO
from collections import defaultdict, deque
import numpy as np
import csv

# ===================== CONFIGURATION CONSTANTS =====================
DOT_TRACKING = True  # Toggle between Dot Tracking (True) or Box Tracking (False)
MIN_STABLE_FRAMES = 8  # Min frames an ID must be detected to be counted

# Counting zone (normalized 0.0 to 1.0)
COUNTING_ZONE_X_MIN = 0.005 #left
COUNTING_ZONE_X_MAX = 1.0   #right
COUNTING_ZONE_Y_MIN = 0.60  #top
COUNTING_ZONE_Y_MAX = 0.80   #bottom
CONFIDENCE_THRESHOLD = 0.50

def is_in_counting_zone(box_center_x, box_center_y, frame_w, frame_h):
    """Check if the object's center is within the counting zone."""
    norm_x = box_center_x / frame_w
    norm_y = box_center_y / frame_h
    return (COUNTING_ZONE_X_MIN <= norm_x <= COUNTING_ZONE_X_MAX and
            COUNTING_ZONE_Y_MIN <= norm_y <= COUNTING_ZONE_Y_MAX)

# ==================== MAIN PROCESS FUNCTION ====================
def process(input_video_path, output_folder, startDate, cam, phase, location):
    NEW_WIDTH, NEW_HEIGHT = 1280, 720

    if not isinstance(startDate, datetime):
        try:
            startDate = datetime.strptime(startDate, "%Y-%m-%d %H:%M:%S")
        except:
            startDate = datetime.now()

    video = cv2.VideoCapture(input_video_path)
    if not video.isOpened():
        raise Exception("Could not open video")

    video_name = os.path.splitext(os.path.basename(input_video_path))[0]
    video_folder = os.path.join(output_folder, video_name)
    os.makedirs(video_folder, exist_ok=True)

    # ==================== CUDA / DEVICE SETUP ====================
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.zeros(1).to(device)
        torch.cuda.synchronize()

    # ==================== MODEL LOADING ====================
    model_path = r"D:\Vehicle_new_traffic\weights\best_p2.pt"
    model = YOLO(model_path)
    model.to(device)

    fps = int(video.get(cv2.CAP_PROP_FPS))
    output_video_path = os.path.join(video_folder, f"{video_name}_labeled.mp4")
    writer = cv2.VideoWriter(
        output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (NEW_WIDTH, NEW_HEIGHT)
    )

    # ==================== TRACKING VARIABLES ====================
    total_counts = defaultdict(int)
    track_history = defaultdict(lambda: deque(maxlen=MIN_STABLE_FRAMES))
    stable_vehicle_ids = {}      # tracker_id -> stable vehicle ID
    vehicle_last_seen = {}       # stable vehicle ID -> last frame_idx seen
    counted_vehicle_signatures = set()
    InfoArray = []
    next_vehicle_id = 1

    PALETTE = [
        (52, 235, 229), (46, 204, 113), (231, 76, 60), (243, 156, 18),
        (155, 89, 182), (52, 152, 219), (241, 196, 15), (26, 188, 156),
        (230, 126, 34), (52, 73, 94)
    ]
    class_names = list(model.names.values())
    class_colors = {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(class_names)}

    def draw_modern_box(img, box, color, thickness=4):
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    def draw_label(img, text, pos, color, scale=0.6, thickness=1):
        x, y = pos
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
        cv2.rectangle(img, (x, y - text_height - baseline), (x + text_width, y + baseline), color, -1)
        cv2.putText(img, text, (x, y - baseline), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)

    frame_idx = 0

    # ==================== MAIN LOOP ====================
    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame_idx += 1
        resized_frame = cv2.resize(frame, (NEW_WIDTH, NEW_HEIGHT))
        annotated_frame = resized_frame.copy()
        milliseconds_timedelta = timedelta(milliseconds=video.get(cv2.CAP_PROP_POS_MSEC))
        updated_datetime = startDate + milliseconds_timedelta

        # ==================== Draw Counting Zone ====================
        zone_x1 = int(COUNTING_ZONE_X_MIN * NEW_WIDTH)
        zone_x2 = int(COUNTING_ZONE_X_MAX * NEW_WIDTH)
        zone_y1 = int(COUNTING_ZONE_Y_MIN * NEW_HEIGHT)
        zone_y2 = int(COUNTING_ZONE_Y_MAX * NEW_HEIGHT)

        # Semi-transparent fill
        zone_overlay = annotated_frame.copy()
        cv2.rectangle(zone_overlay, (zone_x1, zone_y1), (zone_x2, zone_y2), (0, 255, 255), -1)
        cv2.addWeighted(zone_overlay, 0.2, annotated_frame, 0.8, 0, annotated_frame)

        # Border and label
        cv2.rectangle(annotated_frame, (zone_x1, zone_y1), (zone_x2, zone_y2), (0, 255, 255), 2)
        cv2.putText(annotated_frame, "Counting Zone", (zone_x1 + 5, zone_y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # ==================== YOLOv8 Tracking ====================
        results = model.track(
            resized_frame,
            persist=True,
            tracker="botsort.yaml",
            conf=CONFIDENCE_THRESHOLD,
            imgsz=640,
            device=device
        )
        result = results[0]

        current_frame_ids = set([int(i) for i in result.boxes.id.cpu().numpy()] if result.boxes.id is not None else [])

        # Handle disappeared trackers
        for track_id in list(stable_vehicle_ids.keys()):
            if track_id not in current_frame_ids:
                stable_id = stable_vehicle_ids[track_id]
                vehicle_last_seen[stable_id] = frame_idx

        # Remove long-unseen vehicles
        for stable_id, last_seen in list(vehicle_last_seen.items()):
            if frame_idx - last_seen > 50:
                counted_vehicle_signatures.discard(stable_id)
                del vehicle_last_seen[stable_id]

        if result.boxes is not None and result.boxes.id is not None:
            ids = result.boxes.id.cpu().numpy().astype(int)
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            xyxys = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()

            for track_id, cls_id, box, conf in zip(ids, cls_ids, xyxys, confs):
                class_name = model.names[cls_id]
                if class_name.lower() == "person":
                    continue

                x1, y1, x2, y2 = map(int, box)
                center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)

                history = track_history[track_id]
                history.append({
                    "center": (center_x, center_y),
                    "timestamp": updated_datetime,
                    "confidence": conf,
                    "class_name": class_name
                })

                stable_id = stable_vehicle_ids.get(track_id)
                if stable_id is None:
                    stable_id = next_vehicle_id
                    next_vehicle_id += 1
                    stable_vehicle_ids[track_id] = stable_id
                vehicle_last_seen[stable_id] = frame_idx

                # ==================== Counting Logic ====================
                if stable_id not in counted_vehicle_signatures:
                    if is_in_counting_zone(center_x, center_y, NEW_WIDTH, NEW_HEIGHT):
                        counted_vehicle_signatures.add(stable_id)
                        total_counts[class_name] += 1
                        InfoArray.append([
                            updated_datetime, cam, class_name, phase, location,
                            "Stable Entry", "N/A"
                        ])
                        print(f"✅ COUNTED: {class_name} → StableID {stable_id} at {updated_datetime}")

                # ==================== Visualization ====================
                draw_color = class_colors.get(class_name, (255, 255, 255))
                label_text = f"{class_name} | ID {stable_id} | {conf:.2f}"
                if DOT_TRACKING:
                    cv2.circle(annotated_frame, (center_x, center_y), 6, draw_color, -1)
                    cv2.putText(
                        annotated_frame,
                        label_text,
                        (center_x + 10, center_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        draw_color,
                        2,
                        cv2.LINE_AA
                    )
                    for pt in [h["center"] for h in history]:
                        cv2.circle(annotated_frame, pt, 2, draw_color, -1)
                else:
                    draw_modern_box(annotated_frame, (x1, y1, x2, y2), draw_color, 4)
                    draw_label(annotated_frame, label_text, (x1, y1 - 8), draw_color)

        # ==================== Overlay Count Display ====================
        overlay = annotated_frame.copy()
        alpha = 0.7
        y_offset, x_offset, item_spacing = 30, 10, 28

        cv2.rectangle(overlay, (5, 5), (320, 35 + len(total_counts) * 25), (0, 0, 0), -1)
        cv2.putText(overlay, "TOTAL VEHICLE COUNT", (x_offset, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += item_spacing

        total_vehicles = 0
        for vehicle_type, count in sorted(total_counts.items()):
            cv2.putText(overlay, f"{vehicle_type}: {count}", (x_offset + 10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += item_spacing
            total_vehicles += count

        cv2.putText(overlay, f"TOTAL: {total_vehicles}", (x_offset + 10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)

        writer.write(annotated_frame)
        cv2.imshow("YOLOv8 Stable Counting (Label + ID + Confidence)", annotated_frame)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    video.release()
    writer.release()
    cv2.destroyAllWindows()

    print("\nFinal counts (Total Entry):")
    print(dict(total_counts))

    csv_path = os.path.join(video_folder, f"{video_name}_count.csv")
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Camera", "Class", "Phase", "Location", "Event", "Extra"])
        writer.writerows(InfoArray)

    print(f"\n✅ Count data saved to: {csv_path}")
    return InfoArray


# ==================== RUN ====================
if __name__ == "__main__":
    process(
        input_video_path=r"D:\vandalur\230 mins.mp4",
        output_folder=r"D:\CODE\outputs_vandalur",
        startDate="2025-10-01 08:00:00",
        cam="Cam_03",
        phase="Morning",
        location="Main_Road"
    )
