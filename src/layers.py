from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class Layer:
    name: str
    mask: np.ndarray
    color: tuple[int, int, int]
    visible: bool = True


class LayerManager:
    def __init__(self) -> None:
        self.layers: list[Layer] = []
        self._active_index: int = -1

    def add_layer(self, name: str, mask: np.ndarray, color: tuple[int, int, int]) -> Layer:
        layer = Layer(name=name, mask=mask, color=color)
        self.layers.append(layer)
        self._active_index = len(self.layers) - 1
        return layer

    def clear(self) -> None:
        self.layers.clear()
        self._active_index = -1

    def set_active_index(self, index: int) -> None:
        self._active_index = index

    def active_layer(self) -> Optional[Layer]:
        if 0 <= self._active_index < len(self.layers):
            return self.layers[self._active_index]
        return None

    def compose_overlay(self, shape: tuple[int, int], opacity: float = 0.45) -> Optional[np.ndarray]:
        if not self.layers:
            return None
        h, w = shape
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        for layer in self.layers:
            if not layer.visible:
                continue
            alpha = np.zeros((h, w), dtype=np.uint8)
            alpha[layer.mask > 0] = int(255 * opacity)
            color_img = np.zeros((h, w, 3), dtype=np.uint8)
            color_img[layer.mask > 0] = np.array(layer.color, dtype=np.uint8)
            rgba = np.dstack((color_img, alpha))
            overlay = alpha_blend_rgba(overlay, rgba)
        return overlay


def alpha_blend_rgba(base: np.ndarray, top: np.ndarray) -> np.ndarray:
    base_rgb = base[..., :3].astype(np.float32)
    top_rgb = top[..., :3].astype(np.float32)
    base_a = base[..., 3:4].astype(np.float32) / 255.0
    top_a = top[..., 3:4].astype(np.float32) / 255.0

    out_a = top_a + base_a * (1.0 - top_a)
    out_rgb = np.zeros_like(base_rgb)
    valid = out_a > 0
    out_rgb[valid[..., 0]] = (
        top_rgb[valid[..., 0]] * top_a[valid[..., 0]]
        + base_rgb[valid[..., 0]] * base_a[valid[..., 0]] * (1.0 - top_a[valid[..., 0]])
    ) / out_a[valid[..., 0]]

    out = np.zeros_like(base)
    out[..., :3] = np.clip(out_rgb, 0, 255).astype(np.uint8)
    out[..., 3] = np.clip(out_a[..., 0] * 255, 0, 255).astype(np.uint8)
    return out
