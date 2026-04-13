from __future__ import division

from datetime import datetime
from pathlib import Path


class LineCrossCounter(object):
    def __init__(self, line_y, log_counts=False, log_file="counts.log"):
        self.line_y = line_y
        self.log_counts = log_counts
        self.log_file = Path(log_file)

        self.in_count = 0
        self.out_count = 0
        self.prev_centroids = {}  # track_id -> (cx, cy)
        self.counted_in_ids = set()
        self.counted_out_ids = set()

        if self.log_counts:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def set_line(self, line_y):
        self.line_y = line_y

    def update(self, tracked_objects):
        """
        tracked_objects: list of dict with track_id and centroid
        returns: list of crossing events [(track_id, direction), ...]
        """
        events = []

        for obj in tracked_objects:
            track_id = obj["track_id"]
            curr_centroid = obj["centroid"]

            prev_centroid = self.prev_centroids.get(track_id)
            if prev_centroid is not None:
                prev_y = prev_centroid[1]
                curr_y = curr_centroid[1]

                if prev_y < self.line_y <= curr_y and track_id not in self.counted_in_ids:
                    self.in_count += 1
                    self.counted_in_ids.add(track_id)
                    events.append((track_id, "IN"))
                    self._log_event(track_id, "IN")

                elif (
                    prev_y > self.line_y >= curr_y
                    and track_id not in self.counted_out_ids
                ):
                    self.out_count += 1
                    self.counted_out_ids.add(track_id)
                    events.append((track_id, "OUT"))
                    self._log_event(track_id, "OUT")

            self.prev_centroids[track_id] = curr_centroid

        active_ids = set([obj["track_id"] for obj in tracked_objects])
        stale_ids = set(self.prev_centroids.keys()) - active_ids
        for stale_id in stale_ids:
            del self.prev_centroids[stale_id]

        return events

    def _log_event(self, track_id, direction):
        if not self.log_counts:
            return
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = "{},track_id={},direction={},in={},out={}\n".format(
            timestamp, track_id, direction, self.in_count, self.out_count
        )
        with self.log_file.open("a") as fp:
            fp.write(line)
