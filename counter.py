from __future__ import division

from datetime import datetime
from pathlib import Path


class LineCrossCounter(object):
    def __init__(
        self,
        line_y,
        log_counts=False,
        log_file="counts.log",
        hysteresis_px=8,
        cooldown_frames=12,
    ):
        self.line_y = line_y
        self.log_counts = log_counts
        self.log_file = Path(log_file)
        self.hysteresis_px = int(hysteresis_px)
        self.cooldown_frames = int(cooldown_frames)

        self.in_count = 0
        self.out_count = 0
        self.prev_centroids = {}  # track_id -> (cx, cy)
        self.counted_in_ids = set()
        self.counted_out_ids = set()
        self.last_side = {}  # track_id -> -1 (above), 0 (band), +1 (below)
        self.last_count_frame = {}  # track_id -> frame index

        if self.log_counts:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def set_line(self, line_y):
        self.line_y = line_y

    def _side(self, y):
        if y < (self.line_y - self.hysteresis_px):
            return -1
        if y > (self.line_y + self.hysteresis_px):
            return 1
        return 0

    def update(self, tracked_objects, frame_index=0):
        """
        tracked_objects: list of dict with track_id and centroid
        returns: list of crossing events [(track_id, direction), ...]
        """
        events = []

        for obj in tracked_objects:
            track_id = obj["track_id"]
            curr_centroid = obj["centroid"]

            prev_side = self.last_side.get(track_id, 0)
            curr_side = self._side(curr_centroid[1])
            last_count_at = self.last_count_frame.get(track_id, -10**9)
            cooldown_ok = (frame_index - last_count_at) >= self.cooldown_frames

            if prev_side == -1 and curr_side == 1 and cooldown_ok:
                self.in_count += 1
                self.last_count_frame[track_id] = frame_index
                events.append((track_id, "IN"))
                self._log_event(track_id, "IN")
            elif prev_side == 1 and curr_side == -1 and cooldown_ok:
                self.out_count += 1
                self.last_count_frame[track_id] = frame_index
                events.append((track_id, "OUT"))
                self._log_event(track_id, "OUT")

            self.prev_centroids[track_id] = curr_centroid
            if curr_side != 0:
                self.last_side[track_id] = curr_side

        active_ids = set([obj["track_id"] for obj in tracked_objects])
        stale_ids = set(self.prev_centroids.keys()) - active_ids
        for stale_id in stale_ids:
            del self.prev_centroids[stale_id]
            if stale_id in self.last_side:
                del self.last_side[stale_id]
            if stale_id in self.last_count_frame:
                del self.last_count_frame[stale_id]

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
