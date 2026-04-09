from __future__ import annotations

from typing import Any

from ultralytics import YOLO


class PersonDetector:
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float,
        person_class_id: int = 0,
        tracker_config: str = "bytetrack.yaml",
    ) -> None:
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.person_class_id = person_class_id
        self.tracker_config = tracker_config

    def detect_and_track(self, frame: Any) -> list[dict[str, Any]]:
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_config,
            conf=self.confidence_threshold,
            classes=[self.person_class_id],
            device="cpu",
            verbose=False,
            stream=False,
        )
        if not results:
            return []

        result = results[0]
        if result.boxes is None or result.boxes.id is None:
            return []

        xyxy = result.boxes.xyxy
        confs = result.boxes.conf
        ids = result.boxes.id

        tracks: list[dict[str, Any]] = []
        for i in range(len(ids)):
            track_id = ids[i]
            if track_id is None:
                continue

            x1, y1, x2, y2 = xyxy[i].tolist()
            x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
            centroid = ((x1i + x2i) // 2, (y1i + y2i) // 2)
            confidence = float(confs[i])

            tracks.append(
                {
                    "track_id": int(track_id),
                    "bbox": (x1i, y1i, x2i, y2i),
                    "centroid": centroid,
                    "confidence": confidence,
                }
            )
        return tracks
