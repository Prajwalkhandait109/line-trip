from __future__ import division

from collections import OrderedDict
from math import sqrt


class CentroidTracker(object):
    def __init__(self, max_distance=70, max_missed_frames=45):
        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames
        self.next_object_id = 1
        self.objects = OrderedDict()  # object_id -> (cx, cy)
        self.bboxes = OrderedDict()   # object_id -> (x1, y1, x2, y2)
        self.disappeared = OrderedDict()  # object_id -> missed frame count

    def _register(self, centroid, bbox=(0, 0, 0, 0)):
        object_id = self.next_object_id
        self.next_object_id += 1
        self.objects[object_id] = centroid
        self.bboxes[object_id] = bbox
        self.disappeared[object_id] = 0

    def _deregister(self, object_id):
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.bboxes:
            del self.bboxes[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]

    @staticmethod
    def _distance(a, b):
        return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def update(self, detections):
        """
        detections: list of dict with keys:
            bbox -> (x1, y1, x2, y2)
            centroid -> (cx, cy)
            confidence -> float
        returns: list of dict with track_id attached
        """
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_missed_frames:
                    self._deregister(object_id)
            # Return still-live tracks at their last known position
            return [
                {"track_id": oid, "bbox": self.bboxes[oid], "centroid": self.objects[oid], "confidence": 0.0}
                for oid in self.objects
            ]

        input_centroids = [d["centroid"] for d in detections]

        if len(self.objects) == 0:
            for i, c in enumerate(input_centroids):
                self._register(c, detections[i]["bbox"])
            return self._attach_ids(detections, list(self.objects.keys()))

        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        matches = []
        used_rows = set()
        used_cols = set()

        distance_matrix = []
        for i in range(len(object_centroids)):
            row = []
            for j in range(len(input_centroids)):
                row.append(self._distance(object_centroids[i], input_centroids[j]))
            distance_matrix.append(row)

        while True:
            best = None
            for r in range(len(distance_matrix)):
                if r in used_rows:
                    continue
                for c in range(len(distance_matrix[r])):
                    if c in used_cols:
                        continue
                    d = distance_matrix[r][c]
                    if d <= self.max_distance:
                        if best is None or d < best[2]:
                            best = (r, c, d)
            if best is None:
                break

            row, col, _ = best
            used_rows.add(row)
            used_cols.add(col)
            matches.append((row, col))

        for row, col in matches:
            object_id = object_ids[row]
            self.objects[object_id] = input_centroids[col]
            self.bboxes[object_id] = detections[col]["bbox"]
            self.disappeared[object_id] = 0

        unmatched_rows = set(range(len(object_ids))) - used_rows
        for row in unmatched_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_missed_frames:
                self._deregister(object_id)

        unmatched_cols = set(range(len(input_centroids))) - used_cols
        for col in unmatched_cols:
            self._register(input_centroids[col], detections[col]["bbox"])

        # Build mapping from centroid to recently updated IDs.
        assigned_ids = []
        remaining_ids = list(self.objects.keys())
        for det in detections:
            c = det["centroid"]
            best_id = None
            best_dist = None
            for object_id in remaining_ids:
                d = self._distance(self.objects[object_id], c)
                if best_dist is None or d < best_dist:
                    best_dist = d
                    best_id = object_id
            if best_id is None:
                self._register(c)
                best_id = self.next_object_id - 1
            if best_id in remaining_ids:
                remaining_ids.remove(best_id)
            assigned_ids.append(best_id)

        return self._attach_ids(detections, assigned_ids)

    @staticmethod
    def _attach_ids(detections, assigned_ids):
        output = []
        for i, det in enumerate(detections):
            item = {
                "track_id": int(assigned_ids[i]),
                "bbox": det["bbox"],
                "centroid": det["centroid"],
                "confidence": float(det["confidence"]),
            }
            output.append(item)
        return output
