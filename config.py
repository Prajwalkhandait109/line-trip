from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Settings:
    rtsp_url: str = (
        "rtsp://user:Iam_User1@10.129.4.100:554/"
        "cam/realmonitor?channel=23&subtype=1"
    )
    model_path: str = "models/yolov5n_320.onnx"
    person_class_id: int = 0
    confidence_threshold: float = 0.05
    nms_threshold: float = 0.45
    model_input_size: int = 320
    frame_width: int = 320
    line_y: Optional[int] = None
    line_y_ratio: float = 0.45
    reconnect_delay_sec: float = 2.0
    stale_track_frames: int = 180
    tracker_max_distance: int = 260
    show_window: bool = True
    use_latest_frame_reader: bool = True
    skip_frames: int = 2
    use_tracking: bool = True
    line_hysteresis_px: int = 4
    count_cooldown_frames: int = 3
    log_counts: bool = False
    log_file: str = "counts.log"
    cpu_threads: int = 4


DEFAULT_SETTINGS = Settings()


def resolve_line_y(frame_height, explicit_line_y, ratio):
    if explicit_line_y is not None:
        return max(0, min(frame_height - 1, int(explicit_line_y)))
    return max(0, min(frame_height - 1, int(frame_height * ratio)))


def ensure_runtime_dirs(settings):
    if settings.log_counts:
        Path(settings.log_file).parent.mkdir(parents=True, exist_ok=True)
