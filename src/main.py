import logging
import time

import cv2
import onnxruntime as ort

from config import (
    BYTETRACK_MATCH_THRESH,
    BYTETRACK_TRACK_BUFFER,
    BYTETRACK_TRACK_HIGH_THRESH,
    CONF_THRESHOLD,
    FFMPEG_INPUT_FPS,
    FRAME_READ_TIMEOUT_SEC,
    FRAME_SKIP,
    INPUT_SIZE,
    LINE_Y,
    LOG_FILE,
    MODEL_PATH,
    ORT_INTER_OP_THREADS,
    ORT_INTRA_OP_THREADS,
    RTSP_URL,
    SHOW_PREVIEW,
    STREAM_RECONNECT_DELAY_SEC,
)
from stream import LatestFrameCapture
from tracker import Tracker
from utils import postprocess, preprocess


def build_session():
    session_options = ort.SessionOptions()
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.intra_op_num_threads = max(1, ORT_INTRA_OP_THREADS)
    session_options.inter_op_num_threads = max(1, ORT_INTER_OP_THREADS)
    return ort.InferenceSession(
        str(MODEL_PATH),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )


def build_logger():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("line_cross")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(file_handler)
    return logger


def draw_preview(frame, boxes, tracks):
    for box in boxes:
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    for track in tracks:
        cx, cy = track["center"]
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    cv2.line(frame, (0, LINE_Y), (frame.shape[1], LINE_Y), (255, 0, 0), 2)
    cv2.imshow("Line Crossing", frame)


def log_event_snapshot(logger, processed_frames, started_at, in_count, out_count):
    elapsed = max(time.perf_counter() - started_at, 1e-6)
    fps = processed_frames / elapsed
    logger.info("fps=%.2f in=%s out=%s", fps, in_count, out_count)


def main():
    cv2.setUseOptimized(True)
    cv2.setNumThreads(1)

    logger = build_logger()
    session = build_session()
    input_name = session.get_inputs()[0].name

    capture = LatestFrameCapture(RTSP_URL, reconnect_delay=STREAM_RECONNECT_DELAY_SEC).start()
    tracker = Tracker(
        track_activation_threshold=BYTETRACK_TRACK_HIGH_THRESH,
        lost_track_buffer=BYTETRACK_TRACK_BUFFER,
        minimum_matching_threshold=BYTETRACK_MATCH_THRESH,
        frame_rate=FFMPEG_INPUT_FPS,
    )

    frame_count = 0
    processed_frames = 0
    in_count = 0
    out_count = 0
    started_at = time.perf_counter()

    try:
        while True:
            ret, frame = capture.read(timeout=FRAME_READ_TIMEOUT_SEC)
            if not ret:
                continue

            frame_count += 1
            if frame_count % FRAME_SKIP != 0:
                continue

            input_tensor = preprocess(frame, INPUT_SIZE)
            outputs = session.run(None, {input_name: input_tensor})
            boxes = postprocess(outputs, frame, CONF_THRESHOLD, INPUT_SIZE)
            tracks = tracker.update(boxes)
            processed_frames += 1

            for track in tracks:
                if track["previous"] is None or track["id"] in tracker.crossed:
                    continue

                _, previous_y = track["previous"]
                current_center = track["center"]
                _, current_y = current_center

                if previous_y < LINE_Y <= current_y:
                    in_count += 1
                    tracker.crossed.add(track["id"])
                    log_event_snapshot(logger, processed_frames, started_at, in_count, out_count)
                elif previous_y > LINE_Y >= current_y:
                    out_count += 1
                    tracker.crossed.add(track["id"])
                    log_event_snapshot(logger, processed_frames, started_at, in_count, out_count)

            if SHOW_PREVIEW:
                draw_preview(frame, boxes, tracks)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    except KeyboardInterrupt:
        pass
    finally:
        capture.stop()

        if SHOW_PREVIEW:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()