from __future__ import division

import argparse
import os
import threading
import time

import cv2
import numpy as np

from config import DEFAULT_SETTINGS, Settings, ensure_runtime_dirs, resolve_line_y
from counter import LineCrossCounter
from tracker import CentroidTracker


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOv5 ONNX RTSP line crossing people counter (CPU)"
    )
    parser.add_argument("--rtsp", type=str, default=None, help="RTSP URL")
    parser.add_argument("--line-y", type=int, default=None, help="Absolute line Y")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold")
    parser.add_argument("--nms", type=float, default=None, help="NMS threshold")
    parser.add_argument("--width", type=int, default=None, help="Resize width")
    parser.add_argument("--input-size", type=int, default=None, help="ONNX model input size (must match export, e.g. 640)")
    parser.add_argument("--model", type=str, default=None, help="Path to yolov5n.onnx")
    parser.add_argument("--skip-frames", type=int, default=None, help="Process one frame every N+1 frames")
    parser.add_argument("--no-display", action="store_true", help="Disable GUI window (headless mode)")
    parser.add_argument("--log-counts", action="store_true", help="Write crossing events to counts.log")
    parser.add_argument("--no-tracking", action="store_true", help="Disable centroid tracking")
    parser.add_argument("--debug", action="store_true", help="Print runtime debug diagnostics")
    parser.add_argument("--debug-interval", type=float, default=2.0, help="Seconds between debug status prints")
    return parser.parse_args()


def build_settings(args):
    base = DEFAULT_SETTINGS
    return Settings(
        rtsp_url=args.rtsp if args.rtsp else base.rtsp_url,
        model_path=args.model if args.model else base.model_path,
        person_class_id=base.person_class_id,
        confidence_threshold=args.conf if args.conf is not None else base.confidence_threshold,
        nms_threshold=args.nms if args.nms is not None else base.nms_threshold,
        model_input_size=args.input_size if args.input_size else base.model_input_size,
        frame_width=args.width if args.width else base.frame_width,
        line_y=args.line_y if args.line_y is not None else base.line_y,
        line_y_ratio=base.line_y_ratio,
        reconnect_delay_sec=base.reconnect_delay_sec,
        stale_track_frames=base.stale_track_frames,
        tracker_max_distance=base.tracker_max_distance,
        show_window=False if args.no_display else base.show_window,
        use_latest_frame_reader=base.use_latest_frame_reader,
        skip_frames=args.skip_frames if args.skip_frames is not None else base.skip_frames,
        use_tracking=False if args.no_tracking else base.use_tracking,
        log_counts=args.log_counts or base.log_counts,
        log_file=base.log_file,
        cpu_threads=base.cpu_threads,
    )


def configure_runtime(cpu_threads):
    cv2.setNumThreads(max(1, int(cpu_threads)))


class YoloV5OnnxPersonDetector(object):
    def __init__(self, model_path, conf_threshold, nms_threshold, person_class_id, input_size):
        if not os.path.exists(model_path):
            raise FileNotFoundError("ONNX model not found: {}".format(model_path))
        self.person_class_id = int(person_class_id)
        self.conf_threshold = float(conf_threshold)
        self.nms_threshold = float(nms_threshold)
        self.input_size = int(input_size)

        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def detect(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1.0 / 255.0,
            size=(self.input_size, self.input_size),
            mean=(0, 0, 0),
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)
        preds = self.net.forward()
        preds = np.squeeze(preds)
        if preds.ndim == 1:
            preds = np.expand_dims(preds, axis=0)

        # Handle both common YOLOv5 ONNX formats:
        # 1) raw output: [N, 85] => [cx, cy, w, h, obj, cls...]
        # 2) post-NMS output: [N, 6] => [x1, y1, x2, y2, score, class_id]
        if preds.shape[-1] <= 7:
            return self._detect_from_post_nms(preds, w, h)
        return self._detect_from_raw(preds, w, h)

    def _detect_from_raw(self, preds, frame_w, frame_h):
        scale_x = float(frame_w) / float(self.input_size)
        scale_y = float(frame_h) / float(self.input_size)

        boxes = []
        confidences = []
        centroids = []

        for row in preds:
            obj_conf = float(row[4])
            if obj_conf <= 0.0:
                continue

            class_scores = row[5:]
            class_id = int(np.argmax(class_scores))
            class_conf = float(class_scores[class_id])
            score = obj_conf * class_conf

            if class_id != self.person_class_id or score < self.conf_threshold:
                continue

            cx = float(row[0]) * scale_x
            cy = float(row[1]) * scale_y
            bw = float(row[2]) * scale_x
            bh = float(row[3]) * scale_y

            x1 = max(0, int(cx - bw / 2.0))
            y1 = max(0, int(cy - bh / 2.0))
            x2 = min(frame_w - 1, int(x1 + max(1, bw)))
            y2 = min(frame_h - 1, int(y1 + max(1, bh)))

            boxes.append([x1, y1, max(1, x2 - x1), max(1, y2 - y1)])
            confidences.append(score)
            centroids.append(((x1 + x2) // 2, (y1 + y2) // 2))

        return self._nms_to_detections(boxes, confidences, centroids)

    def _detect_from_post_nms(self, preds, frame_w, frame_h):
        detections = []
        for row in preds:
            if len(row) < 6:
                continue
            x1, y1, x2, y2, score, class_id = row[:6]
            if int(class_id) != self.person_class_id:
                continue
            if float(score) < self.conf_threshold:
                continue

            # Some exports produce normalized xyxy; auto-scale if values look normalized.
            if x2 <= 2.0 and y2 <= 2.0:
                x1 *= frame_w
                x2 *= frame_w
                y1 *= frame_h
                y2 *= frame_h

            x1i = max(0, min(frame_w - 1, int(x1)))
            y1i = max(0, min(frame_h - 1, int(y1)))
            x2i = max(0, min(frame_w - 1, int(x2)))
            y2i = max(0, min(frame_h - 1, int(y2)))

            if x2i <= x1i or y2i <= y1i:
                continue

            detections.append(
                {
                    "bbox": (x1i, y1i, x2i, y2i),
                    "centroid": ((x1i + x2i) // 2, (y1i + y2i) // 2),
                    "confidence": float(score),
                }
            )
        return detections

    def _nms_to_detections(self, boxes, confidences, centroids):
        detections = []
        if not boxes:
            return detections

        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, self.conf_threshold, self.nms_threshold
        )
        if len(indices) == 0:
            return detections

        for idx in indices:
            i = int(idx[0]) if isinstance(idx, (tuple, list, np.ndarray)) else int(idx)
            x, y, bw, bh = boxes[i]
            detections.append(
                {
                    "bbox": (x, y, x + bw, y + bh),
                    "centroid": centroids[i],
                    "confidence": float(confidences[i]),
                }
            )
        return detections


def open_capture(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class LatestFrameReader(object):
    def __init__(self, rtsp_url, reconnect_delay_sec):
        self.rtsp_url = rtsp_url
        self.reconnect_delay_sec = reconnect_delay_sec
        self._cap = None
        self._latest_frame = None
        self._latest_index = -1
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop)
        self._thread.daemon = True
        self._thread.start()

    def _loop(self):
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

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()


def resize_keep_aspect(frame, target_width):
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    scale = float(target_width) / float(w)
    target_height = int(h * scale)
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)


def draw_overlay(frame, line_y, in_count, out_count, fps):
    h, w = frame.shape[:2]
    cv2.line(frame, (0, line_y), (w, line_y), (0, 255, 255), 2)
    cv2.putText(frame, "IN: {}   OUT: {}".format(in_count, out_count), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 255, 50), 2, cv2.LINE_AA)
    cv2.putText(frame, "FPS: {:.1f}".format(fps), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def attach_temp_ids(detections):
    tracked = []
    for i, det in enumerate(detections):
        tracked.append({"track_id": int(i + 1), "bbox": det["bbox"], "centroid": det["centroid"], "confidence": det["confidence"]})
    return tracked


def debug_print(enabled, message):
    if enabled:
        print("[DEBUG] {}".format(message), flush=True)


def run():
    args = parse_args()
    settings = build_settings(args)
    debug_enabled = bool(args.debug)
    debug_interval = max(0.5, float(args.debug_interval))

    configure_runtime(settings.cpu_threads)
    ensure_runtime_dirs(settings)

    detector = YoloV5OnnxPersonDetector(
        model_path=settings.model_path,
        conf_threshold=settings.confidence_threshold,
        nms_threshold=settings.nms_threshold,
        person_class_id=settings.person_class_id,
        input_size=settings.model_input_size,
    )
    tracker = CentroidTracker(max_distance=settings.tracker_max_distance, max_missed_frames=settings.stale_track_frames)

    reader = None
    cap = None
    if settings.use_latest_frame_reader:
        reader = LatestFrameReader(
            rtsp_url=settings.rtsp_url,
            reconnect_delay_sec=settings.reconnect_delay_sec,
        )
        reader.start()

    counter = None
    frame_index = 0
    source_frame_index = -1
    processed_frames = 0
    last_tracked_objects = []
    prev_time = time.perf_counter()
    fps = 0.0
    last_debug_time = 0.0
    last_detection_count = 0
    last_tracked_count = 0

    debug_print(
        debug_enabled,
        "start model={} conf={} nms={} width={} skip={} tracking={} no_display={} line_y={}".format(
            settings.model_path,
            settings.confidence_threshold,
            settings.nms_threshold,
            settings.frame_width,
            settings.skip_frames,
            settings.use_tracking,
            not settings.show_window,
            settings.line_y if settings.line_y is not None else "ratio({})".format(settings.line_y_ratio),
        ),
    )

    try:
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
            line_y = resolve_line_y(frame.shape[0], settings.line_y, settings.line_y_ratio)

            if counter is None:
                counter = LineCrossCounter(line_y=line_y, log_counts=settings.log_counts, log_file=settings.log_file)
            else:
                counter.set_line(line_y)

            frame_index += 1
            should_detect = (frame_index - 1) % (max(0, settings.skip_frames) + 1) == 0

            if should_detect:
                detections = detector.detect(frame)
                tracked_objects = tracker.update(detections) if settings.use_tracking else attach_temp_ids(detections)
                last_tracked_objects = tracked_objects
                processed_frames += 1
                last_detection_count = len(detections)
                last_tracked_count = len(tracked_objects)
            else:
                tracked_objects = last_tracked_objects

            events = counter.update(tracked_objects)
            if events:
                for event_track_id, direction in events:
                    debug_print(
                        debug_enabled,
                        "crossing track_id={} direction={} in={} out={}".format(
                            event_track_id, direction, counter.in_count, counter.out_count
                        ),
                    )

            for obj in tracked_objects:
                x1, y1, x2, y2 = obj["bbox"]
                cx, cy = obj["centroid"]
                track_id = obj["track_id"]
                conf = obj["confidence"]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, "ID {} {:.2f}".format(track_id, conf), (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2, cv2.LINE_AA)

            for event_track_id, direction in events:
                cv2.putText(frame, "{} ID {}".format(direction, event_track_id), (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2, cv2.LINE_AA)

            now = time.perf_counter()
            dt = max(now - prev_time, 1e-6)
            inst_fps = 1.0 / dt
            fps = inst_fps if fps == 0.0 else (0.9 * fps + 0.1 * inst_fps)
            prev_time = now

            draw_overlay(frame=frame, line_y=line_y, in_count=counter.in_count, out_count=counter.out_count, fps=fps)
            cv2.putText(
                frame,
                "PROC: {}/{} SKIP:{} {}".format(processed_frames, frame_index, settings.skip_frames, "TRACK" if settings.use_tracking else "NO-TRACK"),
                (10, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 0),
                2,
                cv2.LINE_AA,
            )

            if settings.show_window:
                cv2.imshow("Line Crossing Counter (YOLOv5-ONNX)", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            now_debug = time.perf_counter()
            if debug_enabled and (now_debug - last_debug_time) >= debug_interval:
                debug_print(
                    True,
                    "status fps={:.2f} src_idx={} frame_idx={} detect_now={} tracked_now={} in={} out={} line_y={}".format(
                        fps,
                        source_frame_index,
                        frame_index,
                        last_detection_count,
                        last_tracked_count,
                        counter.in_count,
                        counter.out_count,
                        line_y,
                    ),
                )
                last_debug_time = now_debug
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
