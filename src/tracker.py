# tracker.py

import warnings

import numpy as np
import supervision as sv

# Suppress the FutureWarning about ByteTrack being deprecated in sv 0.28.
# The underlying class is stable; only the top-level alias is deprecated.
warnings.filterwarnings("ignore", category=FutureWarning, module="supervision")

from supervision.tracker.byte_tracker.core import (
    ByteTrack as _ByteTrack,
)

# Access the real class that sits behind the deprecation proxy so we can
# instantiate it directly without triggering the warning every frame.
_RealByteTrack = _ByteTrack._DeprecatedProxy__config.obj


class Tracker:
    """
    ByteTrack-based multi-object tracker with the same public interface as the
    previous centroid tracker:

    Input  (update):  list of (x1, y1, x2, y2, score) from postprocess()
    Output (update):  list of {"id": int, "center": (cx, cy), "previous": (cx, cy) | None}
    State:            self.crossed — set of track IDs that already crossed the line
    """

    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.80,
        frame_rate: float = 30,
    ):
        self._bt = _RealByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )
        # Maps track_id → last known centroid; used to provide "previous" field.
        self._prev_centers: dict[int, tuple[int, int]] = {}
        # Set of track IDs that have already been counted for line crossing.
        self.crossed: set[int] = set()

    def update(self, detections: list) -> list:
        """
        Parameters
        ----------
        detections : list of (x1, y1, x2, y2, score)
            Output of postprocess().  Empty list is handled gracefully.

        Returns
        -------
        list of {"id": int, "center": (cx, cy), "previous": (cx, cy) | None}
            Only currently active tracks are returned (same behaviour as the
            previous centroid tracker).
        """
        if detections:
            xyxy = np.array([[d[0], d[1], d[2], d[3]] for d in detections], dtype=np.float32)
            scores = np.array([d[4] for d in detections], dtype=np.float32)
        else:
            xyxy = np.empty((0, 4), dtype=np.float32)
            scores = np.empty((0,), dtype=np.float32)

        sv_dets = sv.Detections(xyxy=xyxy, confidence=scores)
        tracked = self._bt.update_with_detections(sv_dets)

        active_tracks = []
        new_prev: dict[int, tuple[int, int]] = {}

        if tracked.tracker_id is not None:
            for box, tid in zip(tracked.xyxy, tracked.tracker_id):
                x1, y1, x2, y2 = box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                center = (cx, cy)
                previous = self._prev_centers.get(int(tid))
                new_prev[int(tid)] = center
                active_tracks.append({
                    "id": int(tid),
                    "center": center,
                    "previous": previous,
                })

        self._prev_centers = new_prev
        return active_tracks