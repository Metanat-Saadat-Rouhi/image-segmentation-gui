import os
import sys
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
    QComboBox,
)

from src.editor import MaskEditorDialog
from src.layers import LayerManager
from src.segmentation import (
    adaptive_threshold_segmentation,
    apply_morphology,
    colorize_mask,
    otsu_segmentation,
    region_growing,
    watershed_segmentation,
)
from src.utils import ensure_uint8, ndarray_to_qimage, save_image
from src.viewer import ImageViewer


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Medical Image Segmentation GUI")
        self.resize(1400, 900)

        self.image: Optional[np.ndarray] = None
        self.image_path: Optional[str] = None
        self.active_mask: Optional[np.ndarray] = None
        self.overlay_opacity: float = 0.45
        self.layer_manager = LayerManager()

        self._build_ui()
        self._build_menu()

    def _build_ui(self) -> None:
        central = QWidget()
        root_layout = QHBoxLayout(central)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root_layout.addWidget(splitter)
        self.setCentralWidget(central)

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setSpacing(10)

        form = QFormLayout()
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(["Otsu", "Adaptive Threshold", "Watershed", "Region Growing"])
        form.addRow("Algorithm", self.algorithm_combo)

        self.block_size_spin = QSpinBox()
        self.block_size_spin.setRange(3, 101)
        self.block_size_spin.setSingleStep(2)
        self.block_size_spin.setValue(31)
        form.addRow("Adaptive block size", self.block_size_spin)

        self.region_tol_spin = QSpinBox()
        self.region_tol_spin.setRange(1, 100)
        self.region_tol_spin.setValue(15)
        form.addRow("Region tolerance", self.region_tol_spin)

        self.morph_combo = QComboBox()
        self.morph_combo.addItems(["None", "erode", "dilate", "open", "close"])
        form.addRow("Morphology", self.morph_combo)

        self.kernel_spin = QSpinBox()
        self.kernel_spin.setRange(1, 25)
        self.kernel_spin.setValue(3)
        form.addRow("Kernel size", self.kernel_spin)

        controls_layout.addLayout(form)

        self.run_button = QPushButton("Run Segmentation")
        self.run_button.clicked.connect(self.run_segmentation)
        controls_layout.addWidget(self.run_button)

        self.edit_button = QPushButton("Open Mask Editor")
        self.edit_button.clicked.connect(self.open_editor)
        controls_layout.addWidget(self.edit_button)

        self.export_mask_button = QPushButton("Export Active Mask")
        self.export_mask_button.clicked.connect(self.export_mask)
        controls_layout.addWidget(self.export_mask_button)

        self.export_overlay_button = QPushButton("Export Overlay")
        self.export_overlay_button.clicked.connect(self.export_overlay)
        controls_layout.addWidget(self.export_overlay_button)

        controls_layout.addWidget(QLabel("Overlay opacity"))
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(45)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        controls_layout.addWidget(self.opacity_slider)

        controls_layout.addWidget(QLabel("Layers"))
        self.layer_list = QListWidget()
        self.layer_list.currentRowChanged.connect(self._on_layer_selected)
        controls_layout.addWidget(self.layer_list)

        self.status_label = QLabel("Load an image to begin.")
        self.status_label.setWordWrap(True)
        controls_layout.addWidget(self.status_label)
        controls_layout.addStretch(1)

        self.viewer = ImageViewer()
        self.viewer.seed_selected.connect(self._handle_seed_selected)

        splitter.addWidget(controls)
        splitter.addWidget(self.viewer)
        splitter.setSizes([280, 1120])

    def _build_menu(self) -> None:
        menu = self.menuBar()
        file_menu = menu.addMenu("File")

        open_action = QAction("Open Image", self)
        open_action.triggered.connect(self.open_image)
        file_menu.addAction(open_action)

        clear_action = QAction("Clear Layers", self)
        clear_action.triggered.connect(self.clear_layers)
        file_menu.addAction(clear_action)

        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

    def open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if not path:
            return

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            QMessageBox.warning(self, "Open image", "Could not load the selected image.")
            return

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        self.image = img
        self.image_path = path
        self.active_mask = None
        self.layer_manager.clear()
        self.layer_list.clear()
        self.viewer.set_image(self.image)
        self.status_label.setText(f"Loaded: {os.path.basename(path)}")

    def run_segmentation(self) -> None:
        if self.image is None:
            QMessageBox.information(self, "Run segmentation", "Load an image first.")
            return

        algorithm = self.algorithm_combo.currentText()
        gray = cv2.cvtColor(ensure_uint8(self.image), cv2.COLOR_BGR2GRAY)

        if algorithm == "Otsu":
            mask = otsu_segmentation(gray)
        elif algorithm == "Adaptive Threshold":
            mask = adaptive_threshold_segmentation(gray, self.block_size_spin.value(), 2)
        elif algorithm == "Watershed":
            mask = watershed_segmentation(self.image)
        elif algorithm == "Region Growing":
            self.status_label.setText("Click on the image to choose a seed point for region growing.")
            self.viewer.enable_seed_mode(True)
            return
        else:
            mask = otsu_segmentation(gray)

        morph = self.morph_combo.currentText()
        if morph != "None":
            mask = apply_morphology(mask, morph, self.kernel_spin.value())

        self._store_result(mask, algorithm)

    def _handle_seed_selected(self, x: int, y: int) -> None:
        if self.image is None:
            return
        gray = cv2.cvtColor(ensure_uint8(self.image), cv2.COLOR_BGR2GRAY)
        mask = region_growing(gray, (x, y), self.region_tol_spin.value())
        morph = self.morph_combo.currentText()
        if morph != "None":
            mask = apply_morphology(mask, morph, self.kernel_spin.value())
        self._store_result(mask, f"Region Growing ({x}, {y})")
        self.viewer.enable_seed_mode(False)

    def _store_result(self, mask: np.ndarray, layer_name: str) -> None:
        self.active_mask = (mask > 0).astype(np.uint8) * 255
        color = colorize_mask(self.active_mask)
        layer = self.layer_manager.add_layer(layer_name, self.active_mask.copy(), color)
        self.layer_list.addItem(layer.name)
        self.layer_list.setCurrentRow(self.layer_list.count() - 1)
        self.status_label.setText(f"Segmentation completed: {layer_name}")
        self._update_view()

    def _update_view(self) -> None:
        if self.image is None:
            return
        overlay = self.layer_manager.compose_overlay(self.image.shape[:2], self.overlay_opacity)
        self.viewer.set_overlay(overlay)

    def _on_layer_selected(self, row: int) -> None:
        self.layer_manager.set_active_index(row)
        layer = self.layer_manager.active_layer()
        self.active_mask = None if layer is None else layer.mask.copy()
        self._update_view()

    def _on_opacity_changed(self, value: int) -> None:
        self.overlay_opacity = value / 100.0
        self._update_view()

    def open_editor(self) -> None:
        if self.image is None or self.active_mask is None:
            QMessageBox.information(self, "Mask editor", "Run a segmentation first.")
            return
        dialog = MaskEditorDialog(self.image, self.active_mask, self)
        if dialog.exec():
            edited = dialog.get_mask()
            layer = self.layer_manager.active_layer()
            if layer is not None:
                layer.mask = edited.copy()
                layer.color = colorize_mask(edited)
            self.active_mask = edited
            self._update_view()
            self.status_label.setText("Mask updated from editor.")

    def export_mask(self) -> None:
        if self.active_mask is None:
            QMessageBox.information(self, "Export mask", "No active mask available.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save mask", "mask.png", "PNG Image (*.png)")
        if not path:
            return
        save_image(path, self.active_mask)
        self.status_label.setText(f"Mask exported to {path}")

    def export_overlay(self) -> None:
        if self.image is None:
            QMessageBox.information(self, "Export overlay", "Load an image first.")
            return
        overlay = self.layer_manager.compose_overlay(self.image.shape[:2], self.overlay_opacity)
        if overlay is None:
            QMessageBox.information(self, "Export overlay", "No overlay to export.")
            return
        composed = self.viewer.render_current_view_numpy()
        path, _ = QFileDialog.getSaveFileName(self, "Save overlay", "overlay.png", "PNG Image (*.png)")
        if not path:
            return
        save_image(path, composed)
        self.status_label.setText(f"Overlay exported to {path}")

    def clear_layers(self) -> None:
        self.layer_manager.clear()
        self.layer_list.clear()
        self.active_mask = None
        self._update_view()
        self.status_label.setText("Layers cleared.")


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
