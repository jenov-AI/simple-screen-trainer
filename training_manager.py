"""Manages YOLO dataset on disk and runs training in a background thread."""

import shutil
import random
import yaml
import numpy as np
import cv2
from pathlib import Path
from PyQt5.QtCore import QThread, pyqtSignal
from ultralytics import YOLO


class DatasetManager:
    """Saves images + labels in YOLO format, split into train/val."""

    def __init__(self, project_dir, val_split=0.2):
        self.project_dir = Path(project_dir)
        self.val_split = val_split
        self.img_train_dir = self.project_dir / "images" / "train"
        self.img_val_dir = self.project_dir / "images" / "val"
        self.lbl_train_dir = self.project_dir / "labels" / "train"
        self.lbl_val_dir = self.project_dir / "labels" / "val"
        for d in (self.img_train_dir, self.img_val_dir,
                  self.lbl_train_dir, self.lbl_val_dir):
            d.mkdir(parents=True, exist_ok=True)
        self._counter = len(list(self.img_train_dir.glob("*.png")))

    def save_annotation(self, frame_bgr, annotations, classes):
        if not annotations:
            return
        fname = "frame_%06d" % self._counter
        self._counter += 1

        use_val = random.random() < self.val_split
        img_dir = self.img_val_dir if use_val else self.img_train_dir
        lbl_dir = self.lbl_val_dir if use_val else self.lbl_train_dir

        cv2.imwrite(str(img_dir / (fname + ".png")), frame_bgr)

        lines = []
        for ann in annotations:
            pts = ann.points
            if len(pts) == 4:
                xs = [p.x() for p in pts]; ys = [p.y() for p in pts]
                cx = (min(xs) + max(xs)) / 2; cy = (min(ys) + max(ys)) / 2
                bw = max(xs) - min(xs); bh = max(ys) - min(ys)
                lines.append("%d %.6f %.6f %.6f %.6f" % (ann.class_id, cx, cy, bw, bh))
            else:
                coords = " ".join("%.6f %.6f" % (p.x(), p.y()) for p in pts)
                lines.append("%d %s" % (ann.class_id, coords))
        with open(lbl_dir / (fname + ".txt"), "w") as f:
            f.write("\n".join(lines) + "\n")

        self._write_yaml(classes)

    def save_duplicates(self, frame_bgr, annotations, classes, count=1, vary=True):
        """Save the same frame `count` times; extra copies get mild augmentation."""
        saved = 0
        for i in range(max(1, int(count))):
            f, anns = frame_bgr, annotations
            if vary and i > 0:
                f, anns = self._augment(frame_bgr, annotations)
                if not anns:
                    f, anns = frame_bgr, annotations
            self.save_annotation(f, anns, classes)
            saved += 1
        return saved

    def _augment(self, frame, annotations):
        img = frame.copy()
        H, W = img.shape[:2]

        # Photometric jitter (never moves the labels)
        img = cv2.convertScaleAbs(img, alpha=random.uniform(0.9, 1.1),
                                  beta=random.uniform(-15, 15))
        if random.random() < 0.35:
            img = cv2.GaussianBlur(img, (3, 3), 0)

        # Safe geometric jitter: zoom-in crop (stays inside the image, no gray borders)
        z = random.uniform(1.0, 1.12)          # zoom in up to ~12%
        win = 1.0 / z
        ox = random.uniform(0.0, 1.0 - win)    # random window top-left (normalized)
        oy = random.uniform(0.0, 1.0 - win)
        x0 = int(ox * W); y0 = int(oy * H)
        x1 = max(x0 + 1, min(W, int((ox + win) * W)))
        y1 = max(y0 + 1, min(H, int((oy + win) * H)))
        img = cv2.resize(img[y0:y1, x0:x1], (W, H), interpolation=cv2.INTER_LINEAR)

        # Transform labels with the exact same crop+zoom math
        out = []
        for ann in annotations:
            pts, xs, ys = [], [], []
            for p in ann.points:
                nx = max(0.0, min(1.0, (p.x() - ox) * z))
                ny = max(0.0, min(1.0, (p.y() - oy) * z))
                pts.append(type(p)(nx, ny)); xs.append(nx); ys.append(ny)
            if (max(xs) - min(xs)) <= 1e-4 or (max(ys) - min(ys)) <= 1e-4:
                continue   # fell outside the crop
            out.append(type(ann)(pts, ann.class_id, ann.class_name))
        return img, out

    def _write_yaml(self, classes):
        data = {
            "path": str(self.project_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": {k: v for k, v in sorted(classes.items())},
        }
        with open(self.project_dir / "data.yaml", "w") as f:
            yaml.dump(data, f, default_flow_style=False)

    @property
    def num_images(self):
        return (len(list(self.img_train_dir.glob("*.png"))) +
                len(list(self.img_val_dir.glob("*.png"))))

    @property
    def yaml_path(self):
        return str(self.project_dir / "data.yaml")


class TrainingThread(QThread):
    progress = pyqtSignal(str)
    finished_ok = pyqtSignal(str)
    finished_err = pyqtSignal(str)

    def __init__(self, dataset, epochs=50, imgsz=1280,
                 model_size="n", use_segmentation=False):
        super().__init__()
        self.dataset = dataset
        self.epochs = epochs
        self.imgsz = imgsz
        self.model_size = model_size
        self.use_segmentation = use_segmentation

    def run(self):
        try:
            import torch
            if torch.cuda.is_available():
                self.progress.emit("Using GPU: %s" % torch.cuda.get_device_name(0))
            else:
                self.progress.emit("WARNING: CUDA not available, using CPU.")

            self.progress.emit("Loading model...")
            base = "yolo11%s%s.pt" % (self.model_size,
                                      "-seg" if self.use_segmentation else "")
            model = YOLO(base)

            self.progress.emit("Training yolo11%s @ %dpx, %d epochs on %d images..."
                               % (self.model_size, self.imgsz, self.epochs,
                                  self.dataset.num_images))
            model.train(
                data=self.dataset.yaml_path,
                epochs=self.epochs,
                imgsz=self.imgsz,
                patience=20,
                batch=8,
                project=str(self.dataset.project_dir / "runs"),
                name="train",
                exist_ok=True,
                verbose=True,
            )

            sd = getattr(model.trainer, "save_dir", None)
            if sd is not None:
                best = str(Path(sd) / "weights" / "best.pt")
            else:
                best = str(self.dataset.project_dir / "runs" / "train" /
                           "weights" / "best.pt")
            self.finished_ok.emit(best)
        except Exception as e:
            self.finished_err.emit(str(e))


def export_pt(source_path, dest_path):
    shutil.copy2(source_path, dest_path)