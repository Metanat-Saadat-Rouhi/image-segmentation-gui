from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QAction, QImage, QMouseEvent, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .utils import ndarray_to_qimage


class Canvas(QLabel):
    def __init__(self, image: np.ndarray, mask: np.ndarray) -> None:
        super().__init__()
        self.image = image.copy()
        self.mask = mask.copy()
        self.brush_size = 8
        self.mode = "pen"
        self.last_pos: QPoint | None = None
        self._refresh()

    def set_mode(self, mode: str) -> None:
        self.mode = mode

    def set_brush_size(self, size: int) -> None:
        self.brush_size = size

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_pos = event.pos()
            self._draw_at(event.pos())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.last_pos is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self._draw_line(self.last_pos, event.pos())
            self.last_pos = event.pos()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self.last_pos = None

    def _draw_at(self, pos: QPoint) -> None:
        value = 255 if self.mode == "pen" else 0
        cv2.circle(self.mask, (pos.x(), pos.y()), self.brush_size, value, -1)
        self._refresh()

    def _draw_line(self, start: QPoint, end: QPoint) -> None:
        value = 255 if self.mode == "pen" else 0
        cv2.line(self.mask, (start.x(), start.y()), (end.x(), end.y()), value, self.brush_size * 2)
        self._refresh()

    def _refresh(self) -> None:
        overlay = self.image.copy()
        overlay[self.mask > 0] = (0.35 * overlay[self.mask > 0] + 0.65 * np.array([0, 0, 255])).astype(np.uint8)
        self.setPixmap(QPixmap.fromImage(ndarray_to_qimage(overlay)))
        self.adjustSize()


class MaskEditorDialog(QDialog):
    def __init__(self, image: np.ndarray, mask: np.ndarray, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Mask Editor")
        self.resize(1100, 800)
        layout = QHBoxLayout(self)

        self.canvas = Canvas(image, mask)
        layout.addWidget(self.canvas, 1)

        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.addWidget(QLabel("Tools"))

        pen_button = QPushButton("Pen")
        pen_button.clicked.connect(lambda: self.canvas.set_mode("pen"))
        side_layout.addWidget(pen_button)

        eraser_button = QPushButton("Eraser")
        eraser_button.clicked.connect(lambda: self.canvas.set_mode("eraser"))
        side_layout.addWidget(eraser_button)

        side_layout.addWidget(QLabel("Brush size"))
        brush_slider = QSlider(Qt.Orientation.Horizontal)
        brush_slider.setRange(1, 40)
        brush_slider.setValue(8)
        brush_slider.valueChanged.connect(self.canvas.set_brush_size)
        side_layout.addWidget(brush_slider)

        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self.accept)
        side_layout.addWidget(apply_button)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        side_layout.addWidget(cancel_button)

        side_layout.addStretch(1)
        layout.addWidget(side)

    def get_mask(self) -> np.ndarray:
        return self.canvas.mask.copy()
