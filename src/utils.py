# utils.py

import cv2
import numpy as np


def preprocess(frame, size):
    resized = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    img = np.ascontiguousarray(img)
    return np.expand_dims(img, axis=0)


def _flatten_nms_indices(indices):
    if indices is None:
        return []

    if isinstance(indices, np.ndarray):
        return indices.flatten().tolist()

    flat = []
    for index in indices:
        if isinstance(index, (list, tuple, np.ndarray)):
            flat.append(int(index[0]))
        else:
            flat.append(int(index))
    return flat


def _postprocess_yolov8_raw(raw_output, frame_height, frame_width, conf_threshold, input_size):
    preds = raw_output
    if preds.shape[0] < preds.shape[1]:
        # Typical YOLOv8 ONNX output is (84, num_preds), transpose to (num_preds, 84)
        preds = preds.T

    x_scale = frame_width / input_size
    y_scale = frame_height / input_size

    boxes_xyxy = []
    boxes_for_nms = []
    scores = []

    for p in preds:
        class_scores = p[4:]
        cls = int(np.argmax(class_scores))
        conf = float(class_scores[cls])

        if cls != 0 or conf <= conf_threshold:
            continue

        x, y, box_width, box_height = p[:4]
        x1 = max(0, int((x - box_width / 2) * x_scale))
        y1 = max(0, int((y - box_height / 2) * y_scale))
        x2 = min(frame_width - 1, int((x + box_width / 2) * x_scale))
        y2 = min(frame_height - 1, int((y + box_height / 2) * y_scale))

        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        if width == 0 or height == 0:
            continue

        boxes_xyxy.append((x1, y1, x2, y2, conf))
        boxes_for_nms.append([x1, y1, width, height])
        scores.append(conf)

    if not boxes_xyxy:
        return []

    selected = cv2.dnn.NMSBoxes(boxes_for_nms, scores, conf_threshold, 0.45)
    selected_indices = _flatten_nms_indices(selected)
    return [boxes_xyxy[idx] for idx in selected_indices]


def _postprocess_legacy(raw_output, frame_height, frame_width, conf_threshold, input_size):
    x_scale = frame_width / input_size
    y_scale = frame_height / input_size
    boxes = []

    for p in raw_output:
        conf = float(p[4])
        cls = int(p[5])

        if conf <= conf_threshold or cls != 0:
            continue

        x, y, box_width, box_height = p[:4]
        x1 = max(0, int((x - box_width / 2) * x_scale))
        y1 = max(0, int((y - box_height / 2) * y_scale))
        x2 = min(frame_width - 1, int((x + box_width / 2) * x_scale))
        y2 = min(frame_height - 1, int((y + box_height / 2) * y_scale))
        boxes.append((x1, y1, x2, y2, conf))

    return boxes


def postprocess(outputs, frame, conf_threshold, input_size):
    frame_height, frame_width = frame.shape[:2]
    raw_output = outputs[0][0]

    # YOLOv8 raw ONNX format: (84, num_preds) or (num_preds, 84)
    if raw_output.ndim == 2 and max(raw_output.shape) > 100:
        return _postprocess_yolov8_raw(raw_output, frame_height, frame_width, conf_threshold, input_size)

    # Legacy or NMS-exported format: (num_preds, 6)
    if raw_output.ndim == 2 and raw_output.shape[1] >= 6:
        return _postprocess_legacy(raw_output, frame_height, frame_width, conf_threshold, input_size)

    return []

def get_center(box):
    x1, y1, x2, y2 = box[:4]
    return (x1 + x2)//2, (y1 + y2)//2