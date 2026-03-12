# Medical Image Segmentation GUI

A public portfolio version of an image segmentation desktop tool built with **PyQt6**, **OpenCV**, and **NumPy**.

## Application Preview

![Segmentation Result](assets/screenshots/segmentation-result.png)

## Features

- Load grayscale or RGB images
- Otsu thresholding
- Adaptive thresholding
- Watershed segmentation
- Region growing from a seed point
- Morphology tools: erode, dilate, open, close
- Layer manager with visibility toggles
- Manual mask editor with brush + eraser
- Overlay preview with opacity control
- Export masks and overlay images

## Project structure

```text
medical-image-segmentation-gui/
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
├── assets/
│   └── screenshots/
└── src/
    ├── editor.py
    ├── layers.py
    ├── segmentation.py
    ├── utils.py
    └── viewer.py
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
![Segmentation Result](assets/screenshots/Screenshot 2026-03-12 141024.png)

## Usage

1. Open an image.
2. Choose a segmentation algorithm.
3. Run segmentation on the full image.
4. Edit the result in the manual editor if needed.
5. Export the active mask or overlay.

## Notes

This repository is a standalone public demo intended to showcase GUI engineering, image processing, and segmentation workflow design.
