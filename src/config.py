# config.py

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

MODEL_PATH = MODELS_DIR / "yolov8n_256.onnx"

RTSP_URL = "rtsp://user:Iam_User1@10.129.4.100:554/cam/realmonitor?channel=24&subtype=1"

INPUT_SIZE = 256
CONF_THRESHOLD = 0.25

LINE_Y = 200

FRAME_SKIP = 1

# ---------------------------------------------------------------------------
# ByteTrack settings
# ---------------------------------------------------------------------------
# Minimum confidence to activate a new track (first association pass).
BYTETRACK_TRACK_HIGH_THRESH = 0.50
# Minimum confidence for detections to enter the second association pass.
BYTETRACK_TRACK_LOW_THRESH = 0.10
# IoU threshold used when matching detections to existing tracks.
BYTETRACK_MATCH_THRESH = 0.80
# Number of frames a lost track is kept alive before being dropped.
BYTETRACK_TRACK_BUFFER = 30

FRAME_READ_TIMEOUT_SEC = 2.0
STREAM_RECONNECT_DELAY_SEC = 2.0

SHOW_PREVIEW = False
STATUS_LOG_INTERVAL_SEC = 10

ORT_INTRA_OP_THREADS = 3
ORT_INTER_OP_THREADS = 1

LOG_FILE = LOGS_DIR / "line_cross.log"

# ---------------------------------------------------------------------------
# FFmpeg ingestion settings  (used by FFmpegPipeReader / LatestFrameCapture)
# ---------------------------------------------------------------------------

FFMPEG_BIN = "ffmpeg"
FFMPEG_RTSP_TRANSPORT = "tcp"

FFMPEG_PIPE_WIDTH = 320
FFMPEG_PIPE_HEIGHT = 240

# Frame rate throttle applied inside FFmpeg's filter graph.
# Also used as ByteTrack's frame_rate for Kalman filter tuning.
FFMPEG_INPUT_FPS = 7

FFMPEG_USE_HW_DECODE = False

# ---------------------------------------------------------------------------
# TSSegmentReader settings
# ---------------------------------------------------------------------------
TS_SEGMENT_DIR = Path("/tmp/segments")
TS_SEGMENT_POLL_INTERVAL_SEC = 0.5
TS_SEGMENT_MIN_AGE_SEC = 1.0