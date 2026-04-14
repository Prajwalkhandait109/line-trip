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
    parser.add_argument("--profile", type=str, choices=["accuracy", "max_fps"], default=None, help="Runtime profile preset")
    parser.add_argument("--hysteresis-px", type=int, default=None, help="Line hysteresis band in pixels")
    parser.add_argument("--count-cooldown", type=int, default=None, help="Minimum frames between counts for same ID")
    return parser.parse_args()


def build_settings(args):
    base = DEFAULT_SETTINGS
    model_path = args.model if args.model else base.model_path
    confidence_threshold = (
        args.conf if args.conf is not None else base.confidence_threshold
    )
    nms_threshold = args.nms if args.nms is not None else base.nms_threshold
    model_input_size = args.input_size if args.input_size else base.model_input_size
    frame_width = args.width if args.width else base.frame_width
    use_latest_frame_reader = base.use_latest_frame_reader
    skip_frames = args.skip_frames if args.skip_frames is not None else base.skip_frames
    tracker_max_distance = base.tracker_max_distance
    stale_track_frames = base.stale_track_frames

    if args.profile == "accuracy":
        model_path = args.model if args.model else "models/yolov5n_416.onnx"
        confidence_threshold = 0.15 if args.conf is None else args.conf
        nms_threshold = 0.45 if args.nms is None else args.nms
        model_input_size = 416 if args.input_size is None else args.input_size
        frame_width = 416 if args.width is None else args.width
        use_latest_frame_reader = False
        skip_frames = 0 if args.skip_frames is None else args.skip_frames
        tracker_max_distance = 180
        stale_track_frames = 120
    elif args.profile == "max_fps":
        model_path = args.model if args.model else "models/yolov5n_320.onnx"
        confidence_threshold = 0.15 if args.conf is None else args.conf
        nms_threshold = 0.45 if args.nms is None else args.nms
        model_input_size = 320 if args.input_size is None else args.input_size
        frame_width = 320 if args.width is None else args.width
        use_latest_frame_reader = True
        skip_frames = 2 if args.skip_frames is None else args.skip_frames
        tracker_max_distance = 260
        stale_track_frames = 180

    return Settings(
        rtsp_url=args.rtsp if args.rtsp else base.rtsp_url,
        model_path=model_path,
        person_class_id=base.person_class_id,
        confidence_threshold=confidence_threshold,
        nms_threshold=nms_threshold,
        model_input_size=model_input_size,
        frame_width=frame_width,
        line_y=args.line_y if args.line_y is not None else base.line_y,
        line_y_ratio=base.line_y_ratio,
        reconnect_delay_sec=base.reconnect_delay_sec,
        stale_track_frames=stale_track_frames,
        tracker_max_distance=tracker_max_distance,
        show_window=False if args.no_display else base.show_window,
        use_latest_frame_reader=use_latest_frame_reader,
        skip_frames=skip_frames,
        use_tracking=False if args.no_tracking else base.use_tracking,
        line_hysteresis_px=args.hysteresis_px if args.hysteresis_px is not None else base.line_hysteresis_px,
        count_cooldown_frames=args.count_cooldown if args.count_cooldown is not None else base.count_cooldown_frames,
        log_counts=args.log_counts or base.log_counts,
        log_file=base.log_file,
        cpu_threads=base.cpu_threads,
    )


def configure_runtime(cpu_threads):
    cv2.setNumThreads(max(1, int(cpu_threads)))


def validate_model_input_size(settings):
    model_name = os.path.basename(settings.model_path).lower()
    expected = None
    for size in (320, 416, 640):
        if "_{}".format(size) in model_name:
            expected = size
            break
    if expected is not None and int(settings.model_input_size) != expected:
        raise ValueError(
            "Model/input-size mismatch: model '{}' suggests input-size {}, but got {}".format(
                model_name, expected, settings.model_input_size
            )
        )


class YoloV5OnnxPersonDetector(object):
    def __init__(self, model_path, conf_threshold, nms_threshold, person_class_id, input_size, cpu_threads=2):
        if not os.path.exists(model_path):
            raise FileNotFoundError("ONNX model not found: {}".format(model_path))
        self.person_class_id = int(person_class_id)
        self.conf_threshold = float(conf_threshold)
        self.nms_threshold = float(nms_threshold)
        self.input_size = int(input_size)
        self.last_preprocess_ms = 0.0
        self.last_forward_ms = 0.0
        self.last_postprocess_ms = 0.0

        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        cv2.setNumThreads(max(1, int(cpu_threads)))

    @staticmethod
    def _letterbox(frame, new_shape=640, color=(114, 114, 114)):
        h, w = frame.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        r = min(float(new_shape[0]) / float(h), float(new_shape[1]) / float(w))
        new_unpad = (int(round(w * r)), int(round(h * r)))
        dw = float(new_shape[1] - new_unpad[0]) / 2.0
        dh = float(new_shape[0] - new_unpad[1]) / 2.0

        if (w, h) != new_unpad:
            resized = cv2.resize(frame, new_unpad, interpolation=cv2.INTER_LINEAR)
        else:
            resized = frame

        top = int(round(dh - 0.1))
        bottom = int(round(dh + 0.1))
        left = int(round(dw - 0.1))
        right = int(round(dw + 0.1))
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )
        return padded, r, dw, dh

    def detect(self, frame):
        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        padded, ratio, dw, dh = self._letterbox(frame, self.input_size)
        blob = cv2.dnn.blobFromImage(
            padded,
            scalefactor=1.0 / 255.0,
            size=(self.input_size, self.input_size),
            mean=(0, 0, 0),
            swapRB=True,
            crop=False,
        )
        t1 = time.perf_counter()
        self.net.setInput(blob)
        preds = self.net.forward()
        t2 = time.perf_counter()
        preds = np.squeeze(preds)
        if preds.ndim == 1:
            preds = np.expand_dims(preds, axis=0)

        # Handle both common YOLOv5 ONNX formats:
        # 1) raw output: [N, 85] => [cx, cy, w, h, obj, cls...]
        # 2) post-NMS output: [N, 6] => [x1, y1, x2, y2, score, class_id]
        if preds.shape[-1] <= 7:
            detections = self._detect_from_post_nms(preds, w, h, ratio, dw, dh)
        else:
            detections = self._detect_from_raw(preds, w, h, ratio, dw, dh)
        t3 = time.perf_counter()
        self.last_preprocess_ms = (t1 - t0) * 1000.0
        self.last_forward_ms = (t2 - t1) * 1000.0
        self.last_postprocess_ms = (t3 - t2) * 1000.0
        return detections

    def _detect_from_raw(self, preds, frame_w, frame_h, ratio, dw, dh):
        # Vectorised numpy filtering - avoids slow Python loop over ~25 200 candidates
        obj_conf = preds[:, 4]
        class_scores = preds[:, 5:]
        class_ids = np.argmax(class_scores, axis=1)
        class_conf = class_scores[np.arange(len(preds)), class_ids]
        scores = obj_conf * class_conf

        mask = (class_ids == self.person_class_id) & (scores >= self.conf_threshold) & (obj_conf > 0.0)
        if not np.any(mask):
            return []

        pf = preds[mask]
        sf = scores[mask]
        inv_ratio = 1.0 / max(ratio, 1e-9)

        x1 = np.clip(((pf[:, 0] - pf[:, 2] / 2.0 - dw) * inv_ratio).astype(np.int32), 0, frame_w - 1)
        y1 = np.clip(((pf[:, 1] - pf[:, 3] / 2.0 - dh) * inv_ratio).astype(np.int32), 0, frame_h - 1)
        x2 = np.clip(((pf[:, 0] + pf[:, 2] / 2.0 - dw) * inv_ratio).astype(np.int32), 0, frame_w - 1)
        y2 = np.clip(((pf[:, 1] + pf[:, 3] / 2.0 - dh) * inv_ratio).astype(np.int32), 0, frame_h - 1)

        valid = (x2 > x1) & (y2 > y1)
        x1, y1, x2, y2, sf = x1[valid], y1[valid], x2[valid], y2[valid], sf[valid]
        if len(x1) == 0:
            return []

        boxes = [[int(x1[i]), int(y1[i]), max(1, int(x2[i]) - int(x1[i])), max(1, int(y2[i]) - int(y1[i]))] for i in range(len(x1))]
        confidences = sf.tolist()
        centroids = [((int(x1[i]) + int(x2[i])) // 2, (int(y1[i]) + int(y2[i])) // 2) for i in range(len(x1))]

        return self._nms_to_detections(boxes, confidences, centroids)

    def _detect_from_post_nms(self, preds, frame_w, frame_h, ratio, dw, dh):
        detections = []
        for row in preds:
            if len(row) < 6:
                continue
            x1, y1, x2, y2, score, class_id = row[:6]
            if int(class_id) != self.person_class_id:
                continue
            if float(score) < self.conf_threshold:
                continue

            # Some exports produce normalized xyxy on model input space.
            if x2 <= 2.0 and y2 <= 2.0:
                x1 *= self.input_size
                x2 *= self.input_size
                y1 *= self.input_size
                y2 *= self.input_size

            x1 = (x1 - dw) / max(ratio, 1e-9)
            x2 = (x2 - dw) / max(ratio, 1e-9)
            y1 = (y1 - dh) / max(ratio, 1e-9)
            y2 = (y2 - dh) / max(ratio, 1e-9)

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


class YoloV5TFLitePersonDetector(object):
    """TFLite backend - 2-3x faster than cv2.dnn on ARM. Requires tflite-runtime."""

    def __init__(self, model_path, conf_threshold, nms_threshold, person_class_id, input_size, cpu_threads=2):
        if not os.path.exists(model_path):
            raise FileNotFoundError("TFLite model not found: {}".format(model_path))
        self.person_class_id = int(person_class_id)
        self.conf_threshold = float(conf_threshold)
        self.nms_threshold = float(nms_threshold)
        self.input_size = int(input_size)
        self.last_preprocess_ms = 0.0
        self.last_forward_ms = 0.0
        self.last_postprocess_ms = 0.0

        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            import tensorflow.lite as tflite
            Interpreter = tflite.Interpreter

        self.interpreter = Interpreter(model_path=model_path, num_threads=max(1, int(cpu_threads)))
        self.interpreter.allocate_tensors()
        inp = self.interpreter.get_input_details()[0]
        out = self.interpreter.get_output_details()[0]
        self.input_index = inp["index"]
        self.output_index = out["index"]
        self.input_dtype = inp["dtype"]
        # Input quantization (INT8 models)
        inp_qp = inp.get("quantization_parameters", {})
        self.input_scale = float((inp_qp.get("scales", [1.0]) or [1.0])[0] or 1.0)
        self.input_zero_point = int((inp_qp.get("zero_points", [0]) or [0])[0])
        self.is_int8 = (self.input_dtype == np.uint8 or self.input_dtype == np.int8)
        # Output quantization (must dequantize raw int8 output to float)
        out_qp = out.get("quantization_parameters", {})
        self.output_scale = float((out_qp.get("scales", [1.0]) or [1.0])[0] or 1.0)
        self.output_zero_point = int((out_qp.get("zero_points", [0]) or [0])[0])
        self.output_dtype = out["dtype"]
        print(("[INFO] TFLite loaded: {}\n"
               "  input : dtype={} scale={:.8f} zp={} shape={}\n"
               "  output: dtype={} scale={:.8f} zp={} shape={}").format(
            model_path,
            getattr(self.input_dtype, '__name__', str(self.input_dtype)),
            self.input_scale, self.input_zero_point, inp["shape"],
            getattr(self.output_dtype, '__name__', str(self.output_dtype)),
            self.output_scale, self.output_zero_point, out["shape"]), flush=True)

    @staticmethod
    def _letterbox(frame, new_shape=640, color=(114, 114, 114)):
        h, w = frame.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        r = min(float(new_shape[0]) / float(h), float(new_shape[1]) / float(w))
        new_unpad = (int(round(w * r)), int(round(h * r)))
        dw = float(new_shape[1] - new_unpad[0]) / 2.0
        dh = float(new_shape[0] - new_unpad[1]) / 2.0
        if (w, h) != new_unpad:
            resized = cv2.resize(frame, new_unpad, interpolation=cv2.INTER_LINEAR)
        else:
            resized = frame
        top = int(round(dh - 0.1))
        bottom = int(round(dh + 0.1))
        left = int(round(dw - 0.1))
        right = int(round(dw + 0.1))
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return padded, r, dw, dh

    def detect(self, frame):
        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        padded, ratio, dw, dh = self._letterbox(frame, self.input_size)
        # TFLite expects BHWC (BGR->RGB)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        if self.is_int8:
            # Normalize to [0,1] first, THEN quantize: q = f/scale + zero_point
            float_img = np.expand_dims(rgb, 0).astype(np.float32) / 255.0
            if self.input_dtype == np.int8:
                inp = np.clip(np.round(float_img / self.input_scale + self.input_zero_point), -128, 127).astype(np.int8)
            else:  # uint8
                inp = np.clip(np.round(float_img / self.input_scale + self.input_zero_point), 0, 255).astype(np.uint8)
        else:
            inp = np.expand_dims(rgb, 0).astype(np.float32) / 255.0
        t1 = time.perf_counter()

        self.interpreter.set_tensor(self.input_index, inp)
        self.interpreter.invoke()
        raw = self.interpreter.get_tensor(self.output_index)  # [1, N, 85]
        # Dequantize INT8/UINT8 output to float32
        if self.output_dtype in (np.int8, np.uint8):
            preds = (raw.astype(np.float32) - self.output_zero_point) * self.output_scale
        else:
            preds = raw.astype(np.float32)

        t2 = time.perf_counter()
        preds = np.squeeze(preds)
        if preds.ndim == 1:
            preds = np.expand_dims(preds, 0)

        detections = self._detect_from_raw(preds, w, h, ratio, dw, dh)
        t3 = time.perf_counter()
        self.last_preprocess_ms = (t1 - t0) * 1000.0
        self.last_forward_ms = (t2 - t1) * 1000.0
        self.last_postprocess_ms = (t3 - t2) * 1000.0
        return detections

    def _detect_from_raw(self, preds, frame_w, frame_h, ratio, dw, dh):
        obj_conf = preds[:, 4]
        class_scores = preds[:, 5:]
        class_ids = np.argmax(class_scores, axis=1)
        class_conf = class_scores[np.arange(len(preds)), class_ids]
        scores = obj_conf * class_conf

        mask = (class_ids == self.person_class_id) & (scores >= self.conf_threshold) & (obj_conf > 0.0)
        if not np.any(mask):
            return []

        pf = preds[mask]
        sf = scores[mask]
        inv_ratio = 1.0 / max(ratio, 1e-9)
        # TFLite outputs coordinates normalized to [0,1]; scale to letterbox pixel space
        input_sz = float(self.input_size)
        x1 = np.clip(((pf[:, 0] * input_sz - pf[:, 2] * input_sz / 2.0 - dw) * inv_ratio).astype(np.int32), 0, frame_w - 1)
        y1 = np.clip(((pf[:, 1] * input_sz - pf[:, 3] * input_sz / 2.0 - dh) * inv_ratio).astype(np.int32), 0, frame_h - 1)
        x2 = np.clip(((pf[:, 0] * input_sz + pf[:, 2] * input_sz / 2.0 - dw) * inv_ratio).astype(np.int32), 0, frame_w - 1)
        y2 = np.clip(((pf[:, 1] * input_sz + pf[:, 3] * input_sz / 2.0 - dh) * inv_ratio).astype(np.int32), 0, frame_h - 1)

        valid = (x2 > x1) & (y2 > y1)
        x1, y1, x2, y2, sf = x1[valid], y1[valid], x2[valid], y2[valid], sf[valid]
        if len(x1) == 0:
            return []

        boxes = [[int(x1[i]), int(y1[i]), max(1, int(x2[i] - x1[i])), max(1, int(y2[i] - y1[i]))] for i in range(len(x1))]
        confidences = sf.tolist()
        centroids = [((int(x1[i]) + int(x2[i])) // 2, (int(y1[i]) + int(y2[i])) // 2) for i in range(len(x1))]

        detections = []
        if not boxes:
            return detections
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        if len(indices) == 0:
            return detections
        for idx in indices:
            i = int(idx[0]) if isinstance(idx, (tuple, list, np.ndarray)) else int(idx)
            bx, by, bw, bh = boxes[i]
            detections.append({"bbox": (bx, by, bx + bw, by + bh), "centroid": centroids[i], "confidence": float(confidences[i])})
        return detections


def build_detector(model_path, conf_threshold, nms_threshold, person_class_id, input_size, cpu_threads):
    """Auto-select TFLite or ONNX backend based on file extension."""
    if model_path.endswith(".tflite"):
        return YoloV5TFLitePersonDetector(
            model_path=model_path,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            person_class_id=person_class_id,
            input_size=input_size,
            cpu_threads=cpu_threads,
        )
    return YoloV5OnnxPersonDetector(
        model_path=model_path,
        conf_threshold=conf_threshold,
        nms_threshold=nms_threshold,
        person_class_id=person_class_id,
        input_size=input_size,
        cpu_threads=cpu_threads,
    )


class AsyncInferenceWorker(object):
    """Runs detection + tracking in a background thread so the display loop never blocks on inference.

    The worker always processes the *latest* submitted frame; intermediate frames
    are silently dropped if inference is still busy (drop-frame semantics).
    """

    def __init__(self, detector, tracker):
        self._detector = detector
        self._tracker = tracker
        self._pending_frame = None
        self._pending_id = 0
        self._result = []
        self._result_id = 0
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._thread = threading.Thread(target=self._loop)
        self._thread.daemon = True
        self._thread.start()

    def submit(self, frame):
        """Non-blocking. Overwrites any pending frame not yet processed."""
        with self._lock:
            self._pending_frame = frame
            self._pending_id += 1
        self._event.set()

    def get_result(self):
        """Returns (tracked_objects, result_id). result_id increments each time inference completes."""
        with self._lock:
            return list(self._result), self._result_id

    def _loop(self):
        while True:
            self._event.wait()
            self._event.clear()  # clear BEFORE reading pending to avoid race
            with self._lock:
                frame = self._pending_frame
                pending_id = self._pending_id
            if frame is None:
                continue
            detections = self._detector.detect(frame)
            if self._tracker is not None:
                tracked = self._tracker.update(detections)
            else:
                tracked = [
                    {"track_id": i + 1, "bbox": d["bbox"], "centroid": d["centroid"], "confidence": d["confidence"]}
                    for i, d in enumerate(detections)
                ]
            with self._lock:
                self._result = tracked
                self._result_id = pending_id
                # If a new frame arrived during inference, wake up immediately
                if self._pending_id != pending_id:
                    self._event.set()


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
    validate_model_input_size(settings)

    detector = build_detector(
        model_path=settings.model_path,
        conf_threshold=settings.confidence_threshold,
        nms_threshold=settings.nms_threshold,
        person_class_id=settings.person_class_id,
        input_size=settings.model_input_size,
        cpu_threads=settings.cpu_threads,
    )
    tracker = CentroidTracker(max_distance=settings.tracker_max_distance, max_missed_frames=settings.stale_track_frames) if settings.use_tracking else None
    worker = AsyncInferenceWorker(detector, tracker)

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
    last_result_id = -1
    last_events = []
    event_display_countdown = 0
    EVENT_DISPLAY_FRAMES = 20
    prev_time = time.perf_counter()
    fps = 0.0
    last_debug_time = 0.0
    last_detection_count = 0
    last_tracked_count = 0
    detect_calls = 0
    avg_pre_ms = 0.0
    avg_fwd_ms = 0.0
    avg_post_ms = 0.0
    detect_start_time = time.perf_counter()

    debug_print(
        debug_enabled,
        "start model={} input_size={} conf={} nms={} width={} skip={} tracking={} latest_frame={} no_display={} line_y={}".format(
            settings.model_path,
            settings.model_input_size,
            settings.confidence_threshold,
            settings.nms_threshold,
            settings.frame_width,
            settings.skip_frames,
            settings.use_tracking,
            settings.use_latest_frame_reader,
            not settings.show_window,
            settings.line_y if settings.line_y is not None else "ratio({})".format(settings.line_y_ratio),
        ),
    )
    if not settings.use_tracking:
        print("[WARN] fast mode without tracking enabled: count accuracy may drop", flush=True)

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
                counter = LineCrossCounter(
                    line_y=line_y,
                    log_counts=settings.log_counts,
                    log_file=settings.log_file,
                    hysteresis_px=settings.line_hysteresis_px,
                    cooldown_frames=settings.count_cooldown_frames,
                )
            else:
                counter.set_line(line_y)

            frame_index += 1
            worker.submit(frame)

            tracked_objects, result_id = worker.get_result()
            if result_id != last_result_id and result_id > 0:
                last_result_id = result_id
                processed_frames += 1
                last_detection_count = len(tracked_objects)
                last_tracked_count = len(tracked_objects)
                detect_calls += 1
                alpha = 0.2
                avg_pre_ms = detector.last_preprocess_ms if detect_calls == 1 else (1 - alpha) * avg_pre_ms + alpha * detector.last_preprocess_ms
                avg_fwd_ms = detector.last_forward_ms if detect_calls == 1 else (1 - alpha) * avg_fwd_ms + alpha * detector.last_forward_ms
                avg_post_ms = detector.last_postprocess_ms if detect_calls == 1 else (1 - alpha) * avg_post_ms + alpha * detector.last_postprocess_ms

                # Update counter only when real detections exist (confidence>0).
                # Frozen tracker positions (confidence=0.0) never move and should not
                # trigger counter logic - only new inference hits should.
                has_real_detections = any(o.get("confidence", 0.0) > 0.0 for o in tracked_objects)
                events = []
                if has_real_detections:
                    events = counter.update(tracked_objects, frame_index=detect_calls)
                if events:
                    last_events = events
                    event_display_countdown = EVENT_DISPLAY_FRAMES
                    for event_track_id, direction in events:
                        debug_print(
                            debug_enabled,
                            "crossing track_id={} direction={} in={} out={}".format(
                                event_track_id, direction, counter.in_count, counter.out_count
                            ),
                        )

                # Debug: show where tracked centroids are relative to the line
                if debug_enabled and tracked_objects:
                    ys = [o["centroid"][1] for o in tracked_objects]
                    debug_print(
                        debug_enabled,
                        "centroid_y min={} max={} line_y={} band={}-{}".format(
                            min(ys), max(ys), line_y,
                            line_y - settings.line_hysteresis_px,
                            line_y + settings.line_hysteresis_px,
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

            if event_display_countdown > 0:
                for event_track_id, direction in last_events:
                    cv2.putText(frame, "{} ID {}".format(direction, event_track_id), (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2, cv2.LINE_AA)
                event_display_countdown -= 1

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
                detect_elapsed = max(now_debug - detect_start_time, 1e-6)
                detect_fps = float(detect_calls) / detect_elapsed
                debug_print(
                    True,
                    "status fps={:.2f} detect_fps={:.2f} pre={:.1f}ms fwd={:.1f}ms post={:.1f}ms src_idx={} frame_idx={} detect_now={} tracked_now={} in={} out={} line_y={}".format(
                        fps,
                        detect_fps,
                        avg_pre_ms,
                        avg_fwd_ms,
                        avg_post_ms,
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
