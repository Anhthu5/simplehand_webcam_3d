from dataclasses import dataclass
import cv2
import mediapipe as mp
import numpy as np


@dataclass
class Detection:
    points: np.ndarray
    label: str
    score: float


class HandDetector:
    """MediaPipe is used for ROI only; displayed skeleton comes from SimpleHand."""
    def __init__(self, confidence=0.6):
        if not hasattr(mp, "solutions"):
            raise RuntimeError("Legacy Hands API missing. Install mediapipe==0.10.14 in Python 3.10.")
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=1, model_complexity=0,
            min_detection_confidence=confidence, min_tracking_confidence=confidence)

    def detect(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self.hands.process(rgb)
        if not result.multi_hand_landmarks:
            return None
        h, w = bgr.shape[:2]
        points = np.array([(p.x*w, p.y*h) for p in result.multi_hand_landmarks[0].landmark],
                          dtype=np.float32)
        if not np.isfinite(points).all():
            return None
        label, score = "Unknown", 0.0
        if result.multi_handedness:
            item = result.multi_handedness[0].classification[0]
            label, score = item.label, item.score
        # MediaPipe assumes mirrored input. Camera frames here are unmirrored.
        label = {"Left": "Right", "Right": "Left"}.get(label, label)
        return Detection(points, label, score)

    def close(self):
        self.hands.close()
