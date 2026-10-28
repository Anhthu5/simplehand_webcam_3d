import cv2
import numpy as np

EDGES = [(0, s) for s in (1, 5, 9, 13, 17)] + [
    (s+i, s+i+1) for s in (1, 5, 9, 13, 17) for i in range(3)]


def skeleton(image, points, color=(30, 230, 255)):
    h, w = image.shape[:2]
    valid = np.isfinite(points).all(1) & (np.abs(points).max(1) < 100000)
    for a, b in EDGES:
        if valid[a] and valid[b]:
            ok, p1, p2 = cv2.clipLine((0, 0, w, h), tuple(points[a].astype(int)), tuple(points[b].astype(int)))
            if ok:
                cv2.line(image, p1, p2, color, 2, cv2.LINE_AA)
    for p, v in zip(points, valid):
        if v and 0 <= p[0] < w and 0 <= p[1] < h:
            cv2.circle(image, tuple(p.astype(int)), 4, (50, 70, 255), -1, cv2.LINE_AA)


