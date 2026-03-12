from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QMouseEvent, QPixmap
from PyQt6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView

from .utils import ndarray_to_qimage


class ImageViewer(QGraphicsView):
    seed_selected = pyqtSignal(int, int)

    def __init__(self) -> None:
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.overlay_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        self.scene.addItem(self.overlay_item)
        self.image: np.ndarray | None = None
        self.overlay: np.ndarray | None = None
        self.seed_mode = False
        self.setRenderHints(self.renderHints())
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

    def set_image(self, image: np.ndarray) -> None:
        self.image = image.copy()
        self.pixmap_item.setPixmap(QPixmap.fromImage(ndarray_to_qimage(image)))
        self.overlay_item.setPixmap(QPixmap())
        self.setSceneRect(self.pixmap_item.boundingRect())
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def set_overlay(self, overlay: np.ndarray | None) -> None:
        self.overlay = overlay
        if overlay is None:
            self.overlay_item.setPixmap(QPixmap())
            return
        rgba = cv2.cvtColor(overlay, cv2.COLOR_BGRA2RGBA)
        h, w, _ = rgba.shape
        qimage = QImage(rgba.data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()
        self.overlay_item.setPixmap(QPixmap.fromImage(qimage))

    def enable_seed_mode(self, enabled: bool) -> None:
        self.seed_mode = enabled
        self.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self.seed_mode and self.image is not None:
            scene_pos = self.mapToScene(event.pos())
            x = int(scene_pos.x())
            y = int(scene_pos.y())
            if 0 <= x < self.image.shape[1] and 0 <= y < self.image.shape[0]:
                self.seed_selected.emit(x, y)
                return
        super().mousePressEvent(event)

    def render_current_view_numpy(self) -> np.ndarray:
        if self.image is None:
            raise ValueError("No image loaded")
        result = self.image.copy()
        if self.overlay is not None:
            alpha = self.overlay[..., 3:4].astype(np.float32) / 255.0
            color = self.overlay[..., :3].astype(np.float32)
            result = (result.astype(np.float32) * (1 - alpha) + color * alpha).astype(np.uint8)
        return result
