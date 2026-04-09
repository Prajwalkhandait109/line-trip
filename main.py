from __future__ import annotations

import argparse
import threading
import time

import cv2

from config import DEFAULT_SETTINGS, Settings, ensure_runtime_dirs, resolve_line_y
from src.counter import LineCrossCounter
from src.detector import PersonDetector
from src.tracker import TrackHistoryManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RTSP line crossing people counter")
    parser.add_argument("--rtsp", type=str, default=None, help="RTSP URL")
    parser.add_argument("--line-y", type=int, default=None, help="Absolute line Y position")
    parser.add_argument("--conf", type=float, default=None, help="Detection confidence")
    parser.add_argument("--width", type=int, default=None, help="Resize width")
    parser.add_argument("--model", type=str, default=None, help="YOLO model path")
    parser.add_argument(
        "--save-snapshots",
        action="store_true",
        help="Save snapshot on each crossing event",
    )
    parser.add_argument(
        "--log-counts", action="store_true", help="Log crossing events to file"
    )
    parser.add_argument(
        "--no-display", action="store_true", help="Disable display window (headless mode)"
    )
    parser.add_argument(
        "--low-latency",
        action="store_true",
        help="Enable low-latency mode (latest frame reader + frame skipping)",
    )
    parser.add_argument(
        "--latest-frame",
        action="store_true",
        help="Read only latest RTSP frame via background thread",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=None,
        help="Process one frame every N+1 frames (N=0 means process all)",
    )
    parser.add_argument(
        "--no-tracker",
        action="store_true",
        help="Disable YOLO tracker and use lightweight centroid ID assignment",
    )
    return parser.parse_args()


def build_settings(args: argparse.Namespace) -> Settings:
    base = DEFAULT_SETTINGS
    low_latency = bool(args.low_latency)
    skip_frames = args.skip_frames if args.skip_frames is not None else base.skip_frames
    use_latest = args.latest_frame or base.use_latest_frame_reader
    if low_latency:
        use_latest = True
        skip_frames = max(1, skip_frames)

    return Settings(
        rtsp_url=args.rtsp if args.rtsp else base.rtsp_url,
        model_path=args.model if args.model else base.model_path,
        person_class_id=base.person_class_id,
        confidence_threshold=args.conf if args.conf is not None else base.confidence_threshold,
        frame_width=args.width if args.width else base.frame_width,
        line_y=args.line_y if args.line_y is not None else base.line_y,
        line_y_ratio=base.line_y_ratio,
        reconnect_delay_sec=base.reconnect_delay_sec,
        stale_track_frames=base.stale_track_frames,
        history_length=base.history_length,
        show_window=False if args.no_display else base.show_window,
        save_snapshots=args.save_snapshots or base.save_snapshots,
        snapshot_dir=base.snapshot_dir,
        log_counts=args.log_counts or base.log_counts,
        log_file=base.log_file,
        cpu_threads=base.cpu_threads,
        tracker_config=base.tracker_config,
        use_latest_frame_reader=use_latest,
        skip_frames=max(0, int(skip_frames)),
        use_tracker=False if args.no_tracker else base.use_tracker,
        no_tracker_max_distance=base.no_tracker_max_distance,
    )


def configure_runtime(cpu_threads: int) -> None:
    cv2.setNumThreads(max(1, int(cpu_threads)))
    try:
        import torch

        torch.set_num_threads(max(1, int(cpu_threads)))
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def open_capture(rtsp_url: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class LatestFrameReader:
    def __init__(self, rtsp_url: str, reconnect_delay_sec: float) -> None:
        self.rtsp_url = rtsp_url
        self.reconnect_delay_sec = reconnect_delay_sec
        self._cap: cv2.VideoCapture | None = None
        self._latest_frame = None
        self._latest_index = -1
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                self._cap = open_capture(self.rtsp_url)
                if not self._cap.isOpened():
                    time.sleep(self.reconnect_delay_sec)
                    continue

            ok, frame = self._cap.read()
            if not ok or frame is None:
                self._cap.release()
                self._cap = None
                time.sleep(self.reconnect_delay_sec)
                continue

            with self._lock:
                self._latest_frame = frame
                self._latest_index += 1

    def get_latest(self):
        with self._lock:
            if self._latest_frame is None:
                return None, -1
            return self._latest_frame.copy(), self._latest_index

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()


def resize_keep_aspect(frame, target_width: int):
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    scale = target_width / float(w)
    target_height = int(h * scale)
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)


def draw_overlay(frame, line_y: int, in_count: int, out_count: int, fps: float) -> None:
    h, w = frame.shape[:2]
    cv2.line(frame, (0, line_y), (w, line_y), (0, 255, 255), 2)
    cv2.putText(
        frame,
        f"IN: {in_count}   OUT: {out_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (50, 255, 50),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def run() -> None:
    args = parse_args()
    settings = build_settings(args)

    configure_runtime(settings.cpu_threads)
    ensure_runtime_dirs(settings)

    detector = PersonDetector(
        model_path=settings.model_path,
        confidence_threshold=settings.confidence_threshold,
        person_class_id=settings.person_class_id,
        tracker_config=settings.tracker_config,
        use_tracker=settings.use_tracker,
        no_tracker_max_distance=settings.no_tracker_max_distance,
    )
    track_history = TrackHistoryManager(
        history_length=settings.history_length,
        stale_track_frames=settings.stale_track_frames,
    )

    cap: cv2.VideoCapture | None = None
    reader: LatestFrameReader | None = None
    counter: LineCrossCounter | None = None
    frame_index = 0
    source_frame_index = -1
    processed_frames = 0
    last_tracks = []
    prev_time = time.perf_counter()
    fps = 0.0

    try:
        if settings.use_latest_frame_reader:
            reader = LatestFrameReader(
                rtsp_url=settings.rtsp_url,
                reconnect_delay_sec=settings.reconnect_delay_sec,
            )
            reader.start()

        while True:
            if settings.use_latest_frame_reader:
                frame, latest_idx = reader.get_latest() if reader is not None else (None, -1)
                if frame is None or latest_idx == source_frame_index:
                    time.sleep(0.005)
                    continue
                source_frame_index = latest_idx
            else:
                if cap is None or not cap.isOpened():
                    cap = open_capture(settings.rtsp_url)
                    if not cap.isOpened():
                        time.sleep(settings.reconnect_delay_sec)
                        continue

                ok, frame = cap.read()
                if not ok or frame is None:
                    cap.release()
                    cap = None
                    time.sleep(settings.reconnect_delay_sec)
                    continue
                source_frame_index += 1

            frame = resize_keep_aspect(frame, settings.frame_width)
            line_y = resolve_line_y(
                frame_height=frame.shape[0],
                explicit_line_y=settings.line_y,
                ratio=settings.line_y_ratio,
            )

            if counter is None:
                counter = LineCrossCounter(
                    line_y=line_y,
                    save_snapshots=settings.save_snapshots,
                    snapshot_dir=settings.snapshot_dir,
                    log_counts=settings.log_counts,
                    log_file=settings.log_file,
                )
            else:
                counter.line_y = line_y

            frame_index += 1
            run_detection = (frame_index - 1) % (settings.skip_frames + 1) == 0
            if run_detection:
                tracks = detector.detect_and_track(frame)
                last_tracks = tracks
                processed_frames += 1
            else:
                tracks = last_tracks

            for item in tracks:
                track_id = item["track_id"]
                x1, y1, x2, y2 = item["bbox"]
                cx, cy = item["centroid"]
                conf = item["confidence"]

                prev_centroid, curr_centroid = track_history.update(
                    track_id=track_id, centroid=(cx, cy), frame_index=frame_index
                )

                if counter is not None:
                    event = counter.update(
                        track_id=track_id,
                        prev_centroid=prev_centroid,
                        curr_centroid=curr_centroid,
                        frame=frame,
                    )
                    if event is not None:
                        track_history.mark_counted(track_id, event)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    f"ID {track_id} {conf:.2f}",
                    (x1, max(18, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 220, 0),
                    2,
                    cv2.LINE_AA,
                )

            track_history.cleanup(frame_index)

            now = time.perf_counter()
            dt = max(now - prev_time, 1e-6)
            inst_fps = 1.0 / dt
            fps = inst_fps if fps == 0.0 else (0.9 * fps + 0.1 * inst_fps)
            prev_time = now

            draw_overlay(
                frame=frame,
                line_y=line_y,
                in_count=counter.in_count if counter is not None else 0,
                out_count=counter.out_count if counter is not None else 0,
                fps=fps,
            )
            cv2.putText(
                frame,
                f"PROC: {processed_frames}/{frame_index} "
                f"SKIP:{settings.skip_frames} "
                f"{'BYTE' if settings.use_tracker else 'CENTROID'}",
                (10, 88),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 0),
                2,
                cv2.LINE_AA,
            )

            if settings.show_window:
                cv2.imshow("Line Crossing Counter", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        if reader is not None:
            reader.stop()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
