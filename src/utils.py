from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtGui import QImage


def ensure_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)


def ndarray_to_qimage(image: np.ndarray) -> QImage:
    image = ensure_uint8(image)
    if image.ndim == 2:
        h, w = image.shape
        return QImage(image.data, w, h, w, QImage.Format.Format_Grayscale8).copy()

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()


def save_image(path: str, image: np.ndarray) -> None:
    cv2.imwrite(path, image)
