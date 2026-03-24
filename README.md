# 🚗 Traffic Vehicle Detection & Counting using YOLOv8

## 📌 Project Overview

This project performs **real-time vehicle detection, tracking, and counting** using the powerful **YOLOv8 model**.
It processes video input and counts vehicles based on a defined **counting zone**, ensuring accurate and stable counting using tracking algorithms.

---

## 🚀 Features

* 🔍 Vehicle Detection using YOLOv8
* 🎯 Object Tracking using BoT-SORT / DeepSORT
* 📊 Real-time Vehicle Counting
* 📍 Custom Counting Zone
* 🟢 Stable ID assignment to avoid double counting
* 🎥 Output video with annotations
* ⚡ GPU acceleration support (CUDA)

---

## 🛠️ Tech Stack

* Python
* OpenCV
* PyTorch
* YOLOv8 (Ultralytics)
* NumPy

---

## 📂 Project Structure

```
traffic_veena/
│
├── counting.py          # Main vehicle detection & counting script
├── speed.py             # Speed-related logic (optional)
├── class_map.py         # Class mapping
│
├── botsort.yaml         # Tracker configuration
├── deepsort.yaml        # Tracker configuration
├── data.yaml            # Dataset config
├── ylo.yaml             # Model config
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/tansamaiml/traffic_veena.git
cd traffic_veena
```

### 2️⃣ Create virtual environment (optional)

```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

Update the video path inside `counting.py`:

```python
input_video_path = "your_video.mp4"
```

Run the script:

```bash
python counting.py
```

---

## 📊 Output

* Annotated video with:

  * Bounding boxes / tracking dots
  * Vehicle IDs
  * Vehicle count display panel
* Final vehicle count printed in console

---

## ⚠️ Note

* Model weights (`.pt` files) are not included due to size limitations.
* Download YOLOv8 weights from:
  👉 https://github.com/ultralytics/ultralytics

---

## 🔮 Future Improvements

* Web app deployment (Flask / Streamlit)
* Real-time CCTV integration
* Dashboard visualization
* Multi-lane traffic analysis
* Speed estimation module



Give it a ⭐ on GitHub!
Update this directory using maint_tools/vendor_array_api_compat.sh
