from __future__ import annotations

from collections import defaultdict, deque


class TrackHistoryManager:
    def __init__(self, history_length: int = 12, stale_track_frames: int = 60) -> None:
        self._history: dict[int, deque[tuple[int, int]]] = defaultdict(
            lambda: deque(maxlen=history_length)
        )
        self._last_seen: dict[int, int] = {}
        self._counted: dict[int, dict[str, bool]] = defaultdict(
            lambda: {"IN": False, "OUT": False}
        )
        self._stale_track_frames = stale_track_frames

    def update(
        self, track_id: int, centroid: tuple[int, int], frame_index: int
    ) -> tuple[tuple[int, int] | None, tuple[int, int]]:
        points = self._history[track_id]
        prev = points[-1] if points else None
        points.append(centroid)
        self._last_seen[track_id] = frame_index
        return prev, points[-1]

    def is_counted(self, track_id: int, direction: str) -> bool:
        return self._counted[track_id].get(direction, False)

    def mark_counted(self, track_id: int, direction: str) -> None:
        self._counted[track_id][direction] = True

    def cleanup(self, frame_index: int) -> None:
        stale_ids = [
            track_id
            for track_id, last_frame in self._last_seen.items()
            if frame_index - last_frame > self._stale_track_frames
        ]
        for track_id in stale_ids:
            self._history.pop(track_id, None)
            self._last_seen.pop(track_id, None)
            self._counted.pop(track_id, None)
