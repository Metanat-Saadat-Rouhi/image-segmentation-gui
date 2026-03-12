from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class LayerMask:
    name: str
    mask: np.ndarray
    color: tuple[int, int, int]


def otsu_segmentation(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def adaptive_threshold_segmentation(gray: np.ndarray, block_size: int = 31, c: int = 2) -> np.ndarray:
    if block_size % 2 == 0:
        block_size += 1
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c,
    )


def watershed_segmentation(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.3 * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    cv2.watershed(image.copy(), markers)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    mask[markers > 1] = 255
    return mask


def region_growing(gray: np.ndarray, seed: tuple[int, int], tolerance: int = 15) -> np.ndarray:
    h, w = gray.shape
    sx, sy = seed
    if not (0 <= sx < w and 0 <= sy < h):
        return np.zeros_like(gray, dtype=np.uint8)

    seed_value = int(gray[sy, sx])
    visited = np.zeros_like(gray, dtype=bool)
    mask = np.zeros_like(gray, dtype=np.uint8)
    queue: deque[tuple[int, int]] = deque([(sx, sy)])

    while queue:
        x, y = queue.popleft()
        if x < 0 or y < 0 or x >= w or y >= h or visited[y, x]:
            continue
        visited[y, x] = True

        if abs(int(gray[y, x]) - seed_value) <= tolerance:
            mask[y, x] = 255
            queue.extend([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])

    return mask


def apply_morphology(mask: np.ndarray, operation: str, kernel_size: int = 3) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    operations = {
        "erode": cv2.MORPH_ERODE,
        "dilate": cv2.MORPH_DILATE,
        "open": cv2.MORPH_OPEN,
        "close": cv2.MORPH_CLOSE,
    }
    if operation not in operations:
        return mask
    if operation == "erode":
        return cv2.erode(mask, kernel, iterations=1)
    if operation == "dilate":
        return cv2.dilate(mask, kernel, iterations=1)
    return cv2.morphologyEx(mask, operations[operation], kernel)


def colorize_mask(mask: np.ndarray) -> tuple[int, int, int]:
    rng = np.random.default_rng()
    return tuple(int(v) for v in rng.integers(60, 255, size=3))
