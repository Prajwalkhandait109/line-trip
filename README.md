# Line Crossing (YOLOv5 ONNX, Raspberry Pi, Python 3.7)

Real-time RTSP people counting on Raspberry Pi CPU using:
- YOLOv5n ONNX detection (OpenCV DNN, no PyTorch on target)
- Lightweight centroid tracking
- Line crossing count (IN/OUT)
- Headless logging (`counts.log`)

## Project Structure

```text
line-trip/
|- main.py
|- tracker.py
|- counter.py
|- config.py
|- requirements.txt
`- models/
   |- yolov5n_320.onnx
   `- yolov5n_416.onnx (optional, accuracy profile)
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

## 2) Get YOLOv5 ONNX Models

Place model files in:

```text
models/yolov5n_320.onnx
models/yolov5n_416.onnx   (optional)
```

If you only have `.pt`, export ONNX on another machine (with torch):

```bash
python export.py --weights yolov5n.pt --include onnx --img 320 --opset 12
python export.py --weights yolov5n.pt --include onnx --img 416 --opset 12
```

## 3) Run

Max-FPS profile:

```bash
python main.py --model "./models/yolov5n_320.onnx" --input-size 320 --no-display --log-counts --width 320 --skip-frames 2
```

Accuracy profile:

```bash
python main.py --profile accuracy --model "./models/yolov5n_416.onnx" --input-size 416 --no-display --log-counts
```

No-tracking fallback:

```bash
python main.py --model "./models/yolov5n_320.onnx" --input-size 320 --no-display --log-counts --no-tracking
```

## CLI

- `--rtsp` RTSP URL override
- `--line-y` absolute line Y
- `--conf` confidence threshold
- `--nms` NMS threshold
- `--width` frame width
- `--input-size` ONNX input size (must match model export)
- `--model` path to `.onnx`
- `--skip-frames` process one frame every N+1 frames
- `--no-display` headless mode
- `--log-counts` write `counts.log`
- `--no-tracking` disable centroid tracking
- `--profile` `max_fps` or `accuracy`
- `--hysteresis-px` line hysteresis band
- `--count-cooldown` minimum frames between counts per ID
- `--debug` and `--debug-interval` runtime timing diagnostics

## Verify Counting

Run app:

```bash
python main.py --model "./models/yolov5n_320.onnx" --input-size 320 --no-display --log-counts --debug --debug-interval 1.0
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
