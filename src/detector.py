from __future__ import annotations

from collections import defaultdict
from typing import Any

from ultralytics import YOLO


class _CentroidAssigner:
    def __init__(self, max_distance: int = 70, max_missed_frames: int = 15) -> None:
        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames
        self.next_id = 1
        self.active: dict[int, tuple[int, int]] = {}
        self.missed = defaultdict(int)

    def assign(self, centroids: list[tuple[int, int]]) -> list[int]:
        if not self.active:
            ids = []
            for c in centroids:
                track_id = self.next_id
                self.next_id += 1
                self.active[track_id] = c
                self.missed[track_id] = 0
                ids.append(track_id)
            return ids

        available_ids = set(self.active.keys())
        assigned_ids: list[int] = []

        for c in centroids:
            best_id = None
            best_dist = float("inf")
            for track_id in available_ids:
                old = self.active[track_id]
                dist = ((old[0] - c[0]) ** 2 + (old[1] - c[1]) ** 2) ** 0.5
                if dist < best_dist and dist <= self.max_distance:
                    best_dist = dist
                    best_id = track_id

            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
                self.active[best_id] = c
                self.missed[best_id] = 0
            else:
                self.active[best_id] = c
                self.missed[best_id] = 0
                available_ids.remove(best_id)

            assigned_ids.append(best_id)

        for track_id in list(self.active.keys()):
            if track_id not in assigned_ids:
                self.missed[track_id] += 1
                if self.missed[track_id] > self.max_missed_frames:
                    self.active.pop(track_id, None)
                    self.missed.pop(track_id, None)

        return assigned_ids


class PersonDetector:
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float,
        person_class_id: int = 0,
        tracker_config: str = "bytetrack.yaml",
        use_tracker: bool = True,
        no_tracker_max_distance: int = 70,
    ) -> None:
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.person_class_id = person_class_id
        self.tracker_config = tracker_config
        self.use_tracker = use_tracker
        self._centroid_assigner = _CentroidAssigner(max_distance=no_tracker_max_distance)

    def detect_and_track(self, frame: Any) -> list[dict[str, Any]]:
        if self.use_tracker:
            return self._detect_with_tracker(frame)
        return self._detect_without_tracker(frame)

    def _detect_with_tracker(self, frame: Any) -> list[dict[str, Any]]:
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

    def _detect_without_tracker(self, frame: Any) -> list[dict[str, Any]]:
        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            classes=[self.person_class_id],
            device="cpu",
            verbose=False,
            stream=False,
        )
        if not results:
            return []

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        xyxy = result.boxes.xyxy
        confs = result.boxes.conf
        centroids: list[tuple[int, int]] = []
        bboxes: list[tuple[int, int, int, int]] = []
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i].tolist()
            x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
            bboxes.append((x1i, y1i, x2i, y2i))
            centroids.append(((x1i + x2i) // 2, (y1i + y2i) // 2))

        assigned_ids = self._centroid_assigner.assign(centroids)
        tracks: list[dict[str, Any]] = []
        for i, track_id in enumerate(assigned_ids):
            tracks.append(
                {
                    "track_id": int(track_id),
                    "bbox": bboxes[i],
                    "centroid": centroids[i],
                    "confidence": float(confs[i]),
                }
            )
        return tracks
