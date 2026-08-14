# YOLO11L Screen Trainer

A desktop app that lets you capture your screen, annotate objects with
bounding boxes or free-form polygons, train YOLO11L, and test the model
in real time — all from one window.

---

## Prerequisites

| Requirement        | Notes                                      |
|--------------------|--------------------------------------------|
| Python 3.10+       | 3.11 or 3.12 recommended                  |
| NVIDIA GPU + CUDA  | Strongly recommended for training          |
| 8 GB+ VRAM         | YOLO11L is a large model                   |

> CPU-only training *works* but is extremely slow for YOLO11L.
> Consider `yolo11n` or `yolo11s` if you have no GPU.

---

## Installation

```bash
# fresh environment
rmdir /s /q venv
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip

# 1) PyTorch FIRST, from the CUDA 12.8 index (matches your RTX 5080)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 2) everything else SECOND, with headless OpenCV (NOT opencv-python)
pip install ultralytics opencv-python-headless mss screeninfo PyQt5 PyYAML Pillow numpy

## Running the app
```bash
venv\Scripts\activate
python main.py
```