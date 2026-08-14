"""Live test window - threaded inference with a focusable region of interest."""

import time
import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSlider, QSizePolicy, QFrame,
)
from PyQt5.QtCore import Qt, QObject, QThread, pyqtSignal, pyqtSlot, QPointF, QRectF
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QBrush, QColor

from ultralytics import YOLO
from screen_capture import ScreenCaptureThread

CLASS_COLORS = [
    (46, 204, 113), (52, 152, 219), (231, 76, 60), (241, 196, 15),
    (155, 89, 182), (26, 188, 156), (230, 126, 34), (236, 240, 241),
    (149, 165, 166), (255, 107, 129),
]

HUD_STYLE = """
    QWidget#hud { background: #15171c; }
    QLabel#pill { background:#1f2430; color:#cfd6e4; border-radius:10px; padding:4px 12px; font-weight:600; }
    QLabel#pillOk { background:#123524; color:#4ade80; border-radius:10px; padding:4px 12px; font-weight:700; }
    QLabel#pillErr{ background:#3a1414; color:#f87171; border-radius:10px; padding:4px 12px; font-weight:700; }
    QLabel#metric { color:#8b93a7; font-size:11px; }
    QLabel#metricVal{ color:#e6e9f0; font-size:15px; font-weight:700; }
    QLabel#legendTxt{ color:#aab2c5; font-size:11px; }
    QPushButton#closeBtn{ background:#c0392b; color:white; border:none; border-radius:8px; padding:6px 16px; font-weight:700; }
    QPushButton#closeBtn:hover{ background:#e74c3c; }
    QPushButton#toolBtn{ background:#23272f; color:#c7cdd9; border:1px solid #333a45; border-radius:8px; padding:5px 12px; }
    QPushButton#toolBtn:hover{ background:#2c323c; }
    QSlider::groove:horizontal{ height:4px; background:#2a2f3a; border-radius:2px; }
    QSlider::handle:horizontal{ width:14px; margin:-5px 0; background:#4ade80; border-radius:7px; }
"""


def _bgr_to_pixmap(bgr):
    h, w, ch = bgr.shape
    qimg = QImage(bgr.tobytes(), w, h, ch * w, QImage.Format_BGR888).copy()
    return QPixmap.fromImage(qimg)


class RoiLabel(QLabel):
    """Displays frames and lets the user drag a normalized region of interest."""
    roi_changed = pyqtSignal(object)   # QRectF (normalized) or None

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background:#0c0d11; color:#5a6172; font-size:14px;")
        self.setText("Waiting for first frame...  (drag to focus a region)")
        self._roi = None
        self._dragging = False
        self._start = QPointF(); self._end = QPointF()

    def _img_rect(self):
        pm = self.pixmap()
        if pm is None:
            return QRectF()
        pw, ph = pm.width(), pm.height()
        ww, wh = self.width(), self.height()
        s = min(ww / pw, wh / ph)
        iw, ih = pw * s, ph * s
        return QRectF((ww - iw) / 2, (wh - ih) / 2, iw, ih)

    def _to_norm(self, pt):
        r = self._img_rect()
        if r.width() == 0:
            return QPointF(0, 0)
        return QPointF(max(0.0, min(1.0, (pt.x() - r.x()) / r.width())),
                       max(0.0, min(1.0, (pt.y() - r.y()) / r.height())))

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._dragging = True
            self._start = e.pos(); self._end = e.pos()
            self.update()

    def mouseMoveEvent(self, e):
        if self._dragging:
            self._end = e.pos(); self.update()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self._dragging:
            self._dragging = False
            r = QRectF(self._start, self._end).normalized()
            if r.width() > 12 and r.height() > 12:
                tl = self._to_norm(r.topLeft()); br = self._to_norm(r.bottomRight())
                self._roi = QRectF(tl.x(), tl.y(), br.x() - tl.x(), br.y() - tl.y())
                self.roi_changed.emit(self._roi)
            else:
                self._roi = None
                self.roi_changed.emit(None)
            self.update()

    def paintEvent(self, e):
        super().paintEvent(e)
        if self._dragging:
            p = QPainter(self)
            p.setPen(QPen(QColor(45, 212, 191), 2, Qt.DashLine))
            p.setBrush(QBrush(QColor(45, 212, 191, 25)))
            p.drawRect(QRectF(self._start, self._end).normalized())
            p.end()


class InferenceWorker(QObject):
    result_ready = pyqtSignal(object, int, str)

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.names = model.names
        self._busy = False
        self.conf = 0.35
        self.paint = True
        self.roi = None

    @pyqtSlot(float)
    def set_conf(self, v): self.conf = v

    @pyqtSlot(bool)
    def set_paint(self, v): self.paint = v

    @pyqtSlot(object)
    def set_roi(self, roi): self.roi = roi

    @pyqtSlot(np.ndarray)
    def process(self, frame):
        if self._busy or frame is None or frame.size == 0 or frame.ndim < 3:
            return
        self._busy = True
        try:
            H, W = frame.shape[:2]
            roi = self.roi
            use_roi = (roi is not None and roi.width() > 0.01 and roi.height() > 0.01)
            if use_roi:
                x0 = max(0, min(W, int(roi.x() * W)))
                x1 = max(0, min(W, int((roi.x() + roi.width()) * W)))
                y0 = max(0, min(H, int(roi.y() * H)))
                y1 = max(0, min(H, int((roi.y() + roi.height()) * H)))
                crop = frame[y0:y1, x0:x1].copy()
                if crop.size == 0:
                    crop = frame; use_roi = False
            else:
                crop = frame

            results = self.model(crop, verbose=False, conf=self.conf)
            painted = (self._paint(crop, results) if self.paint
                       else self._boxes(crop, results))

            vis = frame.copy()
            if use_roi:
                vis[y0:y1, x0:x1] = painted
                self._veil(vis, roi)
            else:
                vis = painted

            n = len(results[0].boxes) if results and results[0].boxes is not None else 0
            self.result_ready.emit(_bgr_to_pixmap(vis), n, "")
        except Exception as e:
            self.result_ready.emit(None, 0, "%s: %s" % (type(e).__name__, e))
        finally:
            self._busy = False

    def _veil(self, img, roi):
        H, W = img.shape[:2]
        x0 = int(roi.x() * W); y0 = int(roi.y() * H)
        x1 = int((roi.x() + roi.width()) * W); y1 = int((roi.y() + roi.height()) * H)
        ov = img.copy()
        cv2.rectangle(ov, (0, 0), (W, y0), (6, 7, 10), -1)
        cv2.rectangle(ov, (0, y1), (W, H), (6, 7, 10), -1)
        cv2.rectangle(ov, (0, y0), (x0, y1), (6, 7, 10), -1)
        cv2.rectangle(ov, (x1, y0), (W, y1), (6, 7, 10), -1)
        cv2.addWeighted(ov, 0.75, img, 0.25, 0, img)
        cv2.rectangle(img, (x0, y0), (x1, y1), (45, 212, 191), 2)

    def _paint(self, img, results):
        overlay = img.copy()
        r = results[0]; boxes = r.boxes
        masks = getattr(r, "masks", None)
        for i, cid in enumerate(boxes.cls):
            col = CLASS_COLORS[int(cid) % len(CLASS_COLORS)]
            if masks is not None and i < len(masks.data):
                m = cv2.resize(masks.data[i].cpu().numpy(), (img.shape[1], img.shape[0])) > 0.5
                overlay[m] = col
            else:
                x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
                cv2.rectangle(overlay, (x1, y1), (x2, y2), col, -1)
        cv2.addWeighted(overlay, 0.40, img, 0.60, 0, img)
        return self._boxes(img, results)

    def _boxes(self, img, results):
        r = results[0]
        for i, cid in enumerate(r.boxes.cls):
            cid = int(cid); col = CLASS_COLORS[cid % len(CLASS_COLORS)]
            x1, y1, x2, y2 = map(int, r.boxes.xyxy[i].tolist())
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            txt = "%s %.2f" % (self.names.get(cid, str(cid)), float(r.boxes.conf[i]))
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), col, -1)
            cv2.putText(img, txt, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return img


class TestWindow(QWidget):
    window_closed = pyqtSignal()

    def __init__(self, model_path, monitor_index, parent=None, initial_roi=None):
        super().__init__(parent)
        self.setWindowTitle("YOLO11 - Live Test")
        self.resize(960, 680)
        self.setStyleSheet(HUD_STYLE)
        self._last_err = ""; self._times = []
        self._roi = initial_roi

        root = QVBoxLayout(self); root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)

        hud = QFrame(); hud.setObjectName("hud")
        hb = QHBoxLayout(hud); hb.setContentsMargins(12, 8, 12, 8)
        self.pill = QLabel("WARMING UP GPU..."); self.pill.setObjectName("pillErr")
        hb.addWidget(self.pill); hb.addSpacing(14)
        self.lbl_fps = self._metric(hb, "FPS", "-")
        self.lbl_det = self._metric(hb, "OBJECTS", "0")
        hb.addSpacing(10)

        cw = QVBoxLayout(); cw.setSpacing(2)
        cl = QLabel("confidence"); cl.setObjectName("metric")
        self.conf_slider = QSlider(Qt.Horizontal); self.conf_slider.setRange(5, 95)
        self.conf_slider.setValue(35); self.conf_slider.setFixedWidth(110)
        cw.addWidget(cl); cw.addWidget(self.conf_slider); hb.addLayout(cw); hb.addSpacing(8)

        self.btn_paint = QPushButton("PAINT: ON"); self.btn_paint.setObjectName("toolBtn")
        self.btn_paint.setFixedWidth(90); self.btn_paint.clicked.connect(self._toggle_paint)
        hb.addWidget(self.btn_paint); hb.addSpacing(6)

        self.btn_clear_roi = QPushButton("Full screen"); self.btn_clear_roi.setObjectName("toolBtn")
        self.btn_clear_roi.clicked.connect(lambda: self._on_roi(None))
        hb.addWidget(self.btn_clear_roi); hb.addStretch()

        btn_close = QPushButton("X  Close"); btn_close.setObjectName("closeBtn")
        btn_close.clicked.connect(self.close); hb.addWidget(btn_close)
        root.addWidget(hud)

        self.lbl_region = QLabel(""); self.lbl_region.setObjectName("legendTxt")
        self.lbl_region.setContentsMargins(12, 4, 12, 2); self.lbl_region.setWordWrap(True)
        root.addWidget(self.lbl_region)

        self.view = RoiLabel()
        self.view.roi_changed.connect(self._on_roi)
        root.addWidget(self.view)

        if monitor_index < 0:
            self.pill.setText("NO MONITOR"); return
        try:
            model = YOLO(model_path)
        except Exception as e:
            self.pill.setText("MODEL LOAD FAILED"); self.view.setText(str(e)); return

        self._build_legend(model.names)
        self.worker = InferenceWorker(model)
        self.worker.set_roi(self._roi)
        self.worker_thread = QThread(); self.worker.moveToThread(self.worker_thread)
        self.worker.result_ready.connect(self._on_result)
        self.conf_slider.valueChanged.connect(lambda v: self.worker.set_conf(v / 100.0))
        self.worker_thread.start()

        self.cap = ScreenCaptureThread(monitor_index, fps=20)
        self.cap.frame_ready.connect(self.worker.process)
        self.cap.start()
        self.pill.setText("SCANNING..."); self.pill.setObjectName("pill")
        self._update_region_text()

    def _metric(self, layout, title, val):
        w = QVBoxLayout(); w.setSpacing(0)
        t = QLabel(title); t.setObjectName("metric")
        v = QLabel(val); v.setObjectName("metricVal")
        w.addWidget(t); w.addWidget(v); layout.addLayout(w); layout.addSpacing(10)
        return v

    def _build_legend(self, names):
        sw = "   ".join("<span style='color:rgb(%d,%d,%d)'>#</span> %s"
                        % (c[2], c[1], c[0], names[i]) for i, c in enumerate(CLASS_COLORS) if i in names)
        self.lbl_region.setText("Drag on the preview to focus a region; click to reset.   " + (sw or ""))

    def _update_region_text(self):
        if self._roi is None:
            base = "Region: full screen"
        else:
            base = "Region: %.0f%% x %.0f%% of screen (focused)" % (
                self._roi.width() * 100, self._roi.height() * 100)
        self.lbl_region.setText(base + "  -  drag to move/refocus, click to reset.")

    def _on_roi(self, roi):
        self._roi = roi
        try:
            self.worker.set_roi(roi)
        except Exception:
            pass
        self._update_region_text()

    def _toggle_paint(self):
        on = self.btn_paint.text().endswith("ON")
        self.worker.set_paint(not on)
        self.btn_paint.setText("PAINT: OFF" if on else "PAINT: ON")

    @pyqtSlot(object, int, str)
    def _on_result(self, pixmap, count, err):
        if err:
            if err != self._last_err:
                self._last_err = err
                self.pill.setText("INFERENCE ERROR"); self.pill.setObjectName("pillErr")
                self.view.setText(err)
            return
        self._last_err = ""
        if pixmap is not None:
            self.view.setPixmap(pixmap.scaled(self.view.size(), Qt.KeepAspectRatio,
                                              Qt.SmoothTransformation))
        now = time.perf_counter(); self._times.append(now)
        while self._times and now - self._times[0] > 1.0:
            self._times.pop(0)
        self.lbl_fps.setText(str(len(self._times)))
        self.lbl_det.setText(str(count))
        if self.pill.objectName() != "pillOk":
            self.pill.setText("LIVE"); self.pill.setObjectName("pillOk")

    def closeEvent(self, event):
        try:
            self.cap.stop()
        except Exception:
            pass
        try:
            self.worker_thread.quit(); self.worker_thread.wait(2000)
        except Exception:
            pass
        self.window_closed.emit()
        super().closeEvent(event)