# Line Crossing RPi

Real-time people `IN/OUT` counting from an RTSP stream using YOLOv8n + tracking, optimized for Raspberry Pi 4 (CPU only).

## Project Structure

```text
line-trip/
├── models/
│   └── yolov8n.pt
├── src/
│   ├── detector.py
│   ├── tracker.py
│   └── counter.py
├── main.py
├── config.py
└── requirements.txt
```

## Setup (Raspberry Pi)

1. Install system packages:

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv libatlas-base-dev libopenblas-dev
```

2. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install Python dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Download YOLOv8 Nano model into `models/`:

```bash
mkdir -p models
wget -O models/yolov8n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt
```

## Run

Default run (uses RTSP URL in `config.py`):

```bash
python3 main.py
```

Example with CLI overrides:

```bash
python3 main.py \
  --rtsp "rtsp://user:Iam_User1@10.129.4.100:554/cam/realmonitor?channel=21&subtype=1" \
  --line-y 260 \
  --conf 0.35 \
  --width 640 \
  --save-snapshots \
  --log-counts
```

Press `ESC` to exit when display is enabled.
