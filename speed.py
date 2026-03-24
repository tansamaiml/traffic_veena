import os
import cv2
import torch
from datetime import timedelta, datetime
from ultralytics import YOLO
from collections import defaultdict, deque
import numpy as np
import csv
import time


# ===================== CONFIGURATION CONSTANTS =====================
DOT_TRACKING = True
SHOW_WINDOWS = True   # show video while processing
MIN_STABLE_FRAMES = 5
MIN_DISPLACEMENT = 5

COUNTING_ZONE_X_MIN = 0.005
COUNTING_ZONE_X_MAX = 1.0
COUNTING_ZONE_Y_MIN = 0.60
COUNTING_ZONE_Y_MAX = 0.99
CONFIDENCE_THRESHOLD = 0.40  # ↓ slightly lower to detect fast/blurred bikes better

# Count display customization
COUNT_BG_COLOR = (20, 20, 40)
COUNT_TEXT_COLOR = (0, 255, 0)
COUNT_HEADER_COLOR = (0, 255, 255)
COUNT_TOTAL_COLOR = (0, 255, 0)
COUNT_HEADER_SCALE = 1.0
COUNT_ITEM_SCALE = 0.6
COUNT_TOTAL_SCALE = 0.5
COUNT_BOX_ALPHA = 0.35

DOT_SIZE = 4
DOT_BORDER_SIZE = 9

# ==================== PERFORMANCE SETTINGS ====================
NEW_WIDTH, NEW_HEIGHT = 1280, 720
#SHOW_WINDOWS = False
SKIP_FRAMES = 0                  # process every frame for smoother tracking
USE_HALF_PRECISION = True        # enable FP16 inference if supported


def is_in_counting_zone(box_center_x, box_center_y, frame_w, frame_h):
    norm_x = box_center_x / frame_w
    norm_y = box_center_y / frame_h
    return (COUNTING_ZONE_X_MIN <= norm_x <= COUNTING_ZONE_X_MAX and
            COUNTING_ZONE_Y_MIN <= norm_y <= COUNTING_ZONE_Y_MAX)


def draw_counting_zone(img, frame_w, frame_h):
    x1 = int(COUNTING_ZONE_X_MIN * frame_w)
    y1 = int(COUNTING_ZONE_Y_MIN * frame_h)
    x2 = int(COUNTING_ZONE_X_MAX * frame_w)
    y2 = int(COUNTING_ZONE_Y_MAX * frame_h)
    corner_length = 50
    corner_thickness = 6
    corner_color = (0, 255, 255)
    # four corners
    cv2.line(img, (x1, y1), (x1 + corner_length, y1), corner_color, corner_thickness)
    cv2.line(img, (x1, y1), (x1, y1 + corner_length), corner_color, corner_thickness)
    cv2.line(img, (x2, y1), (x2 - corner_length, y1), corner_color, corner_thickness)
    cv2.line(img, (x2, y1), (x2, y1 + corner_length), corner_color, corner_thickness)
    cv2.line(img, (x1, y2), (x1 + corner_length, y2), corner_color, corner_thickness)
    cv2.line(img, (x1, y2), (x1, y2 - corner_length), corner_color, corner_thickness)
    cv2.line(img, (x2, y2), (x2 - corner_length, y2), corner_color, corner_thickness)
    cv2.line(img, (x2, y2), (x2, y2 - corner_length), corner_color, corner_thickness)


def draw_rounded_rectangle(img, pt1, pt2, color, thickness, radius=15):
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)


def process(input_video_path, output_folder):
    video = cv2.VideoCapture(input_video_path)
    if not video.isOpened():
        raise Exception("Could not open video")

    video_name = os.path.splitext(os.path.basename(input_video_path))[0]
    video_folder = os.path.join(output_folder, video_name)
    os.makedirs(video_folder, exist_ok=True)

    print("🔍 Checking CUDA availability...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ CUDA not available — using CPU.")

    model_path = r"C:\train_trf\train_v2\weights\Traffic_new.pt"
    #model_path = r"Z:\Highways\pt\best_blnc.pt"
    model = YOLO(model_path).to(device)

    # === Precision handling ===
    if USE_HALF_PRECISION and device.type == "cuda":
       # model.model.half()
        print("🧮 Using FP16 for faster inference.")
    else:
        model.model.float()
        print("⚙️ Using FP32 precision.")

    fps = int(video.get(cv2.CAP_PROP_FPS))
    writer = cv2.VideoWriter(
        os.path.join(video_folder, f"{video_name}_stable.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (NEW_WIDTH, NEW_HEIGHT)
    )

    total_counts = defaultdict(int)
    counted_vehicles = {}
    track_persistence = defaultdict(int)
    track_history = defaultdict(lambda: deque(maxlen=MIN_STABLE_FRAMES))
    counted_vehicle_signatures = set()
    InfoArray = []
    next_vehicle_id = 1

    PALETTE = [
        (255, 100, 100), (100, 255, 100), (100, 100, 255),
        (255, 255, 100), (255, 100, 255), (100, 255, 255),
        (255, 200, 100), (200, 100, 255), (100, 255, 200), (255, 150, 150)
    ]
    class_names = list(model.names.values())
    class_colors = {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(class_names)}

    frame_idx = 0
    print("🚀 Starting video processing...")

    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx % (SKIP_FRAMES + 1) != 0:
            continue

        start_time = time.time()

        resized_frame = cv2.resize(frame, (NEW_WIDTH, NEW_HEIGHT))
        annotated_frame = resized_frame.copy()
        draw_counting_zone(annotated_frame, NEW_WIDTH, NEW_HEIGHT)

        results = model.track(
            resized_frame,
            persist=True,
            tracker=r"D:\PythonProject\PythonProject\PythonProject\PythonProject\traffice_testing\botsort2.yaml",
            conf=CONFIDENCE_THRESHOLD,
            imgsz=960,
            device=device
        )

        result = results[0]
        current_frame_ids = set(
            [int(i) for i in result.boxes.id.cpu().numpy()] if result.boxes.id is not None else []
        )
        for track_id in list(track_persistence.keys()):
            if track_id not in current_frame_ids:
                track_persistence[track_id] = 0

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
                stable_id = counted_vehicles.get(track_id)

                track_history[track_id].append((center_x, center_y))
                if len(track_history[track_id]) >= MIN_STABLE_FRAMES:
                    dx = track_history[track_id][-1][0] - track_history[track_id][0][0]
                    dy = track_history[track_id][-1][1] - track_history[track_id][0][1]
                    displacement = np.sqrt(dx ** 2 + dy ** 2)
                else:
                    displacement = 0

                sig = (class_name, round(center_x / 10), round(center_y / 10))

                if stable_id is None and displacement >= MIN_DISPLACEMENT and sig not in counted_vehicle_signatures:
                    if is_in_counting_zone(center_x, center_y, NEW_WIDTH, NEW_HEIGHT):
                        track_persistence[track_id] += 1
                        if track_persistence[track_id] >= MIN_STABLE_FRAMES:
                            stable_id = next_vehicle_id
                            next_vehicle_id += 1
                            counted_vehicles[track_id] = stable_id
                            counted_vehicle_signatures.add(sig)
                            total_counts[class_name] += 1
                            print(f"STABLE COUNT: {class_name} → StableID {stable_id}")
                    else:
                        track_persistence[track_id] = 0

                draw_color = class_colors[class_name]
                label_text = f"{class_name} {conf:.2f}" if stable_id is None else f"ID:{stable_id} {class_name} {conf:.2f}"

                if is_in_counting_zone(center_x, center_y, NEW_WIDTH, NEW_HEIGHT):
                    if DOT_TRACKING:
                        cv2.circle(annotated_frame, (center_x, center_y), DOT_SIZE, draw_color, -1)
                        cv2.circle(annotated_frame, (center_x, center_y), DOT_BORDER_SIZE, (255, 255, 255), 1)
                    else:
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), draw_color, 2)
                    # ======== DRAW LABEL TEXT ========
                    text_scale = 0.6
                    text_thickness = 2
                    (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, text_scale,
                                                          text_thickness)
                    label_bg_top = max(y1 - 10, text_h + 10)
                    cv2.rectangle(annotated_frame, (x1, y1 - text_h - 8), (x1 + text_w + 4, y1), draw_color, -1)
                    cv2.putText(
                        annotated_frame,
                        label_text,
                        (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        text_scale,
                        (255, 255, 255),
                        text_thickness
                    )

        # ======= Count panel =======
        overlay = annotated_frame.copy()
        header_height = int(40 * COUNT_HEADER_SCALE)
        item_spacing = int(32 * COUNT_ITEM_SCALE)
        num_items = len(total_counts) if len(total_counts) > 0 else 1
        box_height = header_height + (num_items + 1) * item_spacing + 30
        draw_rounded_rectangle(overlay, (10, 10), (380, box_height), COUNT_BG_COLOR, -1, radius=20)
        cv2.addWeighted(overlay, COUNT_BOX_ALPHA, annotated_frame, 1 - COUNT_BOX_ALPHA, 0, annotated_frame)

        y = 45
        cv2.putText(annotated_frame, "VEHICLE COUNT", (20, y),
                    cv2.FONT_HERSHEY_DUPLEX, COUNT_HEADER_SCALE, COUNT_HEADER_COLOR, 2)
        y += item_spacing
        total_vehicles = sum(total_counts.values())
        for vehicle_type, count in sorted(total_counts.items()):
            cv2.putText(annotated_frame, f"{vehicle_type}: {count}", (25, y),
                        cv2.FONT_HERSHEY_SIMPLEX, COUNT_ITEM_SCALE, COUNT_TEXT_COLOR, 2)
            y += item_spacing
        cv2.putText(annotated_frame, f"TOTAL: {total_vehicles}", (25, y + 10),
                    cv2.FONT_HERSHEY_DUPLEX, COUNT_TOTAL_SCALE, COUNT_TOTAL_COLOR, 2)

        writer.write(annotated_frame)

        if SHOW_WINDOWS:
            cv2.imshow("YOLOv8 Stable Counting", annotated_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("🛑 Stopping early by user request.")
                break

        print(f"⏱ Frame {frame_idx} processed in {time.time() - start_time:.2f}s")

    video.release()
    writer.release()
    cv2.destroyAllWindows()

    print("\nFinal counts (Total Entry):")
    print(dict(total_counts))
    print(f"\n✅ Count data saved to: {video_folder}")
    return InfoArray


if __name__ == "__main__":
    process(
        #input_video_path=r"Z:\5_min_kancheepuram.mp4",
        input_video_path=r"D:\test_vedio_30 min.mp4",
        output_folder=r"D:\traffic\kundardhur_day_30hr_new"
    )