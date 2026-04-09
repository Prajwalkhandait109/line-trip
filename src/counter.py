from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2


class LineCrossCounter:
    def __init__(
        self,
        line_y: int,
        save_snapshots: bool = False,
        snapshot_dir: str = "snapshots",
        log_counts: bool = False,
        log_file: str = "counts.log",
    ) -> None:
        self.line_y = line_y
        self.in_count = 0
        self.out_count = 0
        self._counted = defaultdict(lambda: {"IN": False, "OUT": False})
        self.save_snapshots = save_snapshots
        self.snapshot_dir = Path(snapshot_dir)
        self.log_counts = log_counts
        self.log_file = Path(log_file)

        if self.save_snapshots:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        if self.log_counts:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def update(
        self,
        track_id: int,
        prev_centroid: tuple[int, int] | None,
        curr_centroid: tuple[int, int],
        frame: Any,
    ) -> str | None:
        if prev_centroid is None:
            return None

        prev_y = prev_centroid[1]
        curr_y = curr_centroid[1]
        direction: str | None = None

        if prev_y < self.line_y <= curr_y and not self._counted[track_id]["IN"]:
            self.in_count += 1
            self._counted[track_id]["IN"] = True
            direction = "IN"
        elif prev_y > self.line_y >= curr_y and not self._counted[track_id]["OUT"]:
            self.out_count += 1
            self._counted[track_id]["OUT"] = True
            direction = "OUT"

        if direction is not None:
            self._on_cross_event(track_id=track_id, direction=direction, frame=frame)
        return direction

    def _on_cross_event(self, track_id: int, direction: str, frame: Any) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if self.log_counts:
            line = (
                f"{timestamp},track_id={track_id},direction={direction},"
                f"in={self.in_count},out={self.out_count}\n"
            )
            with self.log_file.open("a", encoding="utf-8") as f:
                f.write(line)

        if self.save_snapshots:
            fname = datetime.now().strftime(
                f"{direction.lower()}_id{track_id}_%Y%m%d_%H%M%S_%f.jpg"
            )
            cv2.imwrite(str(self.snapshot_dir / fname), frame)
