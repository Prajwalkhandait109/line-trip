# Line Crossing (YOLOv5 ONNX, Raspberry Pi, Python 3.7)

Real-time RTSP people counting on Raspberry Pi CPU using:
- YOLOv5n ONNX detection (OpenCV DNN, no PyTorch on target)
- Lightweight centroid tracking
- Line crossing count (IN/OUT)
- Headless logging (`counts.log`)

## Project Structure

```text
line-trip/
├── main.py
├── tracker.py
├── counter.py
├── config.py
├── requirements.txt
└── models/
    └── yolov5n.onnx
```

## 1) Raspberry Pi Setup (Buster / Python 3.7)

Use system OpenCV + NumPy:

```bash
sudo apt update
sudo apt install -y python3-opencv python3-numpy python3-venv
```

Create venv with system packages visible:

```bash
python3 -m venv --system-site-packages venv
source venv/bin/activate
```

Optional pip path (if you prefer pip wheels):

```bash
python -m pip install -r requirements.txt
```

## 2) Get YOLOv5 ONNX Model

Place model at:

```text
models/yolov5n.onnx
```

If you only have `.pt`, export ONNX on another machine (with torch):

```bash
python export.py --weights yolov5n.pt --include onnx --img 640
```

Then copy `yolov5n.onnx` to Pi.

## 3) Run

Headless + logging (recommended):

```bash
python main.py --model "./models/yolov5n.onnx" --no-display --log-counts --width 512 --skip-frames 1
```

No-tracking fallback:

```bash
python main.py --model "./models/yolov5n.onnx" --no-display --log-counts --no-tracking
```

## CLI

- `--rtsp` RTSP URL override
- `--line-y` absolute line Y
- `--conf` confidence threshold
- `--nms` NMS threshold
- `--width` frame width
- `--model` path to `.onnx`
- `--skip-frames` process one frame every N+1 frames
- `--no-display` headless mode
- `--log-counts` write `counts.log`
- `--no-tracking` disable centroid tracking

## Verify Counting

Run app:

```bash
python main.py --model "./models/yolov5n.onnx" --no-display --log-counts
```

Watch logs:

```bash
tail -f counts.log
```

Expected line format:

```text
2026-04-10 15:40:11,track_id=4,direction=IN,in=1,out=0
2026-04-10 15:41:03,track_id=7,direction=OUT,in=1,out=1
```
