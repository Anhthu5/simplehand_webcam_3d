from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class Crop:
    image: np.ndarray
    inverse: np.ndarray
    box: tuple


def square_crop(frame, points, padding=1.35, size=224, flip=False):
    """Square affine crop with black padding outside image, and exact inverse."""
    low, high = points.min(0), points.max(0)
    side = float(max(high - low) * padding)
    if side < 16 or not np.isfinite(side):
        return None
    cx, cy = (low + high) / 2
    x, y = float(cx-side/2), float(cy-side/2)
    scale = size / side
    matrix = np.array([[scale, 0, -x*scale], [0, scale, -y*scale]], dtype=np.float64)
    image = cv2.warpAffine(frame, matrix, (size, size), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT)
    # Optional left-hand canonicalization. Include reflection in inverse geometry.
    if flip:
        image = cv2.flip(image, 1)
        reflection = np.array([[-1., 0, size-1], [0, 1, 0], [0, 0, 1]])
        matrix = (reflection @ np.vstack([matrix, [0, 0, 1]]))[:2]
    return Crop(image, cv2.invertAffineTransform(matrix), (x, y, x+side, y+side))


def uv_to_frame(normalized_uv, crop, size=224):
    pixels = np.asarray(normalized_uv, dtype=np.float64) * size
    homogeneous = np.column_stack([pixels, np.ones(len(pixels))])
    return (homogeneous @ crop.inverse.T).astype(np.float32)
