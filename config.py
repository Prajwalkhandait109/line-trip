from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    rtsp_url: str = (
        "rtsp://user:Iam_User1@10.129.4.100:554/"
        "cam/realmonitor?channel=23&subtype=1"
    )
    model_path: str = "models/yolov8n.pt"
    person_class_id: int = 0
    confidence_threshold: float = 0.35
    frame_width: int = 640
    line_y: int | None = None
    line_y_ratio: float = 0.45
    reconnect_delay_sec: float = 2.0
    stale_track_frames: int = 60
    history_length: int = 12
    show_window: bool = True
    save_snapshots: bool = False
    snapshot_dir: str = "snapshots"
    log_counts: bool = False
    log_file: str = "counts.log"
    cpu_threads: int = 2
    tracker_config: str = "bytetrack.yaml"


DEFAULT_SETTINGS = Settings()


def resolve_line_y(frame_height: int, explicit_line_y: int | None, ratio: float) -> int:
    if explicit_line_y is not None:
        return max(0, min(frame_height - 1, int(explicit_line_y)))
    return max(0, min(frame_height - 1, int(frame_height * ratio)))


def ensure_runtime_dirs(settings: Settings) -> None:
    if settings.save_snapshots:
        Path(settings.snapshot_dir).mkdir(parents=True, exist_ok=True)
