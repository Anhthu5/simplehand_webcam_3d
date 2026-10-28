"""Export measured session statistics and an optional side-by-side demo video."""
from datetime import datetime
from pathlib import Path
import json
import cv2
import numpy as np


class SessionStats:
    def __init__(self):
        self.frames = self.predictions = 0
        self.total_seconds = self.active_seconds = 0.
        self.detection_ms = self.inference_ms = self.render_ms = 0.

    def add(self, seconds, detection_ms, inference_ms, render_ms, predicted):
        self.frames += 1
        self.total_seconds += seconds
        self.detection_ms += detection_ms
        self.render_ms += render_ms
        if predicted:
            self.predictions += 1
            self.active_seconds += seconds
            self.inference_ms += inference_ms

    def summary(self):
        return {
            "frames": self.frames,
            "frames_with_simplehand_prediction": self.predictions,
            "tracked_frame_fraction": self.predictions/max(self.frames, 1),
            "overall_loop_fps": self.frames/max(self.total_seconds, 1e-9),
            "loop_fps_when_prediction_present": self.predictions/self.active_seconds if self.predictions else None,
            "mean_mediapipe_ms": self.detection_ms/max(self.frames, 1),
            "mean_simplehand_ms_when_present": self.inference_ms/self.predictions if self.predictions else None,
            "mean_render_ms": self.render_ms/max(self.frames, 1),
            "measured_loop_seconds": self.total_seconds,
            "note": "Tracking fraction is not accuracy. No 3D error without ground truth. "
                    "Frame time includes camera, processing, optional recording and display, "
                    "but excludes one-off snapshot export after keypress."
        }

    def save(self, directory, configuration):
        if not self.frames:
            return None
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"session_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
        with path.open("w", encoding="utf-8") as stream:
            json.dump({"configuration": configuration, "measurements": self.summary()},
                      stream, ensure_ascii=False, indent=2)
        return path


def compose_demo(webcam, mesh):
    """Resize to a fixed canvas without distorting either aspect ratio."""
    canvas = np.full((620, 1360, 3), 16, np.uint8)
    for image, x, width in [(webcam, 0, 640), (mesh, 640, 720)]:
        if image is None:
            continue
        h, w = image.shape[:2]
        scale = min(width/w, 620/h)
        new_w, new_h = max(1, int(w*scale)), max(1, int(h*scale))
        resized = cv2.resize(image, (new_w, new_h))
        left, top = x+(width-new_w)//2, (620-new_h)//2
        canvas[top:top+new_h, left:left+new_w] = resized
    return canvas


class DemoRecorder:
    """Resample output timestamps to keep the MP4 close to actual elapsed time."""
    def __init__(self, path, fps=20.):
        self.path, self.fps = Path(path), fps
        if self.path.exists():
            raise FileExistsError(f"Recording already exists: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1360, 620))
        if not self.writer.isOpened():
            raise RuntimeError(f"Cannot create demo MP4: {self.path}")
        self.start = None
        self.count = 0
        self.previous = None

    def write(self, webcam, mesh, timestamp):
        frame = compose_demo(webcam, mesh)
        if self.start is None:
            self.start = timestamp
        desired_count = int(max(0., timestamp-self.start)*self.fps)+1
        while self.count < desired_count-1:
            self.writer.write(self.previous if self.previous is not None else frame)
            self.count += 1
        if self.count < desired_count:
            self.writer.write(frame)
            self.count += 1
        self.previous = frame

    def close(self):
        self.writer.release()
