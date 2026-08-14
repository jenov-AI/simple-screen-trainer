"""Annotation canvas: bbox + polygon editing, freeze banner, clip + zoom-to-clip."""

import time
import math
import numpy as np
from PyQt5.QtWidgets import (QWidget, QInputDialog, QMenu, QDialog, QComboBox,
                             QCompleter, QVBoxLayout, QDialogButtonBox, QLabel)
from PyQt5.QtCore import Qt, QPointF, QRectF, QRect, pyqtSignal, QTimer
from PyQt5.QtGui import (
    QPainter, QImage, QPixmap, QColor, QPen, QBrush,
    QPolygonF, QFont, QCursor
)

try:
    from PyQt5.sip import voidptr as _voidptr
    _HAVE_SIP = True
except Exception:
    try:
        from PyQt5 import sip as _sipmod
        _voidptr = _sipmod.voidptr
        _HAVE_SIP = True
    except Exception:
        _HAVE_SIP = False

HANDLE_SIZE = 8
EDGE_HANDLE_SIZE = 6
MIN_POLY_POINTS = 3
CLIP_COLOR = QColor(45, 212, 191)


class Annotation:
    def __init__(self, points, class_id, class_name):
        self.points = points
        self.class_id = class_id
        self.class_name = class_name

    def bbox(self):
        xs = [p.x() for p in self.points]
        ys = [p.y() for p in self.points]
        return QRectF(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))

    def is_rect(self):
        return len(self.points) == 4


class ClassifyDialog(QDialog):
    """Type a new class, or pick / autocomplete an existing one."""
    def __init__(self, existing, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Classify region")
        self.setFixedWidth(340)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Class name (type a new one, or pick existing):"))
        self.combo = QComboBox()
        self.combo.setEditable(True)
        self.combo.setInsertPolicy(QComboBox.NoInsert)
        self.combo.addItems(sorted(existing))
        comp = QCompleter(sorted(existing), self)
        comp.setCaseSensitivity(Qt.CaseInsensitive)
        comp.setCompletionMode(QCompleter.PopupCompletion)
        comp.setFilterMode(Qt.MatchContains)
        self.combo.setCompleter(comp)
        lay.addWidget(self.combo)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    @property
    def value(self):
        return self.combo.currentText().strip()


class AnnotationCanvas(QWidget):
    annotation_added = pyqtSignal(Annotation)
    status_message = pyqtSignal(str)
    freeze_requested = pyqtSignal()
    resume_requested = pyqtSignal()
    clip_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)

        self._frame = None
        self._pixmap = None

        self.annotations = []
        self.classes = {}
        self._next_class_id = 0

        self._selected_idx = -1
        self._dragging_handle = -1
        self._dragging_edge = -1
        self._drawing = False
        self._draw_start = QPointF()
        self._draw_end = QPointF()
        self._pending_points = None

        self._hover_handle = -1
        self._hover_edge = -1

        # Freeze + mode + clip state
        self.paused = False
        self.mode = "annotate"
        self._clip = None
        self._clip_drag = None
        self._clip_corner = -1
        self._clip_move_off = QPointF()
        self._clip_draw_start = QPointF()
        self._clip_draw_end = QPointF()

        self._blink_timer = QTimer(self)
        self._blink_timer.setInterval(220)
        self._blink_timer.timeout.connect(self._on_blink)
        self._blink_timer.start()

    # ---- mode / clip API ----
    def set_mode(self, m):
        self.mode = m
        self._selected_idx = -1
        if m == "clip":
            self.freeze_requested.emit()
        self.update()

    def clear_clip(self):
        self._clip = None
        self._clip_drag = None
        self._clip_corner = -1
        self.clip_changed.emit()
        self.update()

    def set_paused(self, b):
        self.paused = bool(b)
        self.update()

    def _on_blink(self):
        if self.paused:
            self.update()

    # ---- frame -> pixmap (zero copy when possible) ----
    def _frame_to_pixmap(self, frame):
        h, w, ch = frame.shape
        bpl = ch * w
        if (_HAVE_SIP and frame.dtype == np.uint8 and ch == 3
                and frame.flags["C_CONTIGUOUS"]):
            qimg = QImage(_voidptr(frame.data), w, h, bpl, QImage.Format_BGR888)
            return QPixmap.fromImage(qimg)
        qimg = QImage(frame.tobytes(), w, h, bpl, QImage.Format_BGR888)
        return QPixmap.fromImage(qimg)

    def update_frame(self, frame):
        self._frame = frame
        self._pixmap = self._frame_to_pixmap(frame)
        self.update()

    # ---- save payload: crop by clip + remap annotations ----
    def get_save_data(self, full_frame=None):
        base = full_frame if full_frame is not None else self._frame
        if base is None:
            return (None, [])
        if self._clip is None or self._clip.width() <= 0 or self._clip.height() <= 0:
            return (base, list(self.annotations))

        H, W = base.shape[:2]
        x0 = int(round(self._clip.x() * W));  y0 = int(round(self._clip.y() * H))
        x1 = int(round((self._clip.x() + self._clip.width()) * W))
        y1 = int(round((self._clip.y() + self._clip.height()) * H))
        x0 = max(0, min(W, x0)); x1 = max(0, min(W, x1))
        y0 = max(0, min(H, y0)); y1 = max(0, min(H, y1))
        if x1 <= x0 or y1 <= y0:
            return (base, list(self.annotations))
        cropped = base[y0:y1, x0:x1].copy()

        cx, cy = self._clip.x(), self._clip.y()
        cw, ch = self._clip.width(), self._clip.height()
        out = []
        for ann in self.annotations:
            newpts, xs, ys = [], [], []
            for p in ann.points:
                nx = max(0.0, min(1.0, (p.x() - cx) / cw))
                ny = max(0.0, min(1.0, (p.y() - cy) / ch))
                newpts.append(QPointF(nx, ny)); xs.append(nx); ys.append(ny)
            if (max(xs) - min(xs)) <= 1e-4 or (max(ys) - min(ys)) <= 1e-4:
                continue
            out.append(Annotation(newpts, ann.class_id, ann.class_name))
        return (cropped, out)

    # ---- coordinate helpers (zooms to the clip when one is set) ----
    def _active_clip(self):
        c = self._clip
        if c is not None and c.width() > 0.001 and c.height() > 0.001:
            return c
        return None

    def _display_rect(self):
        if self._pixmap is None:
            return QRectF()
        ww, wh = self.width(), self.height()
        clip = self._active_clip()
        if clip is None:
            pw, ph = float(self._pixmap.width()), float(self._pixmap.height())
        else:
            pw = max(1.0, clip.width() * self._pixmap.width())
            ph = max(1.0, clip.height() * self._pixmap.height())
        s = min(ww / pw, wh / ph)
        iw, ih = pw * s, ph * s
        return QRectF((ww - iw) / 2, (wh - ih) / 2, iw, ih)

    def _widget_to_norm(self, pt):
        dr = self._display_rect()
        if dr.width() == 0 or dr.height() == 0:
            return QPointF(0, 0)
        lx = (pt.x() - dr.x()) / dr.width()
        ly = (pt.y() - dr.y()) / dr.height()
        clip = self._active_clip()
        if clip is not None:
            fx = clip.x() + lx * clip.width()
            fy = clip.y() + ly * clip.height()
        else:
            fx, fy = lx, ly
        return QPointF(max(0.0, min(1.0, fx)), max(0.0, min(1.0, fy)))

    def _norm_to_widget(self, pt):
        dr = self._display_rect()
        clip = self._active_clip()
        if clip is not None:
            lx = (pt.x() - clip.x()) / clip.width()
            ly = (pt.y() - clip.y()) / clip.height()
        else:
            lx, ly = pt.x(), pt.y()
        return QPointF(dr.x() + lx * dr.width(), dr.y() + ly * dr.height())

    # ---- annotation handles ----
    def _handles(self, ann):
        return [self._norm_to_widget(p) for p in ann.points]

    def _edge_midpoints(self, ann):
        pts = self._handles(ann); mids = []; n = len(pts)
        for i in range(n):
            j = (i + 1) % n
            mids.append(QPointF((pts[i].x() + pts[j].x()) / 2,
                                (pts[i].y() + pts[j].y()) / 2))
        return mids

    def _hit_handle(self, pos, ann):
        for i, h in enumerate(self._handles(ann)):
            if (pos - h).manhattanLength() < HANDLE_SIZE + 4:
                return i
        return -1

    def _hit_edge(self, pos, ann):
        for i, m in enumerate(self._edge_midpoints(ann)):
            if (pos - m).manhattanLength() < EDGE_HANDLE_SIZE + 4:
                return i
        return -1

    # ---- clip geometry ----
    def _clip_widget_rect(self):
        if self._clip is None:
            return QRectF()
        tl = self._norm_to_widget(QPointF(self._clip.x(), self._clip.y()))
        br = self._norm_to_widget(QPointF(self._clip.x() + self._clip.width(),
                                          self._clip.y() + self._clip.height()))
        return QRectF(tl, br).normalized()

    def _clip_corners(self):
        cr = self._clip_widget_rect()
        if cr.isNull():
            return []
        return [cr.topLeft(), QPointF(cr.right(), cr.top()),
                cr.bottomRight(), QPointF(cr.left(), cr.bottom())]

    def _hit_clip_corner(self, pos):
        for i, h in enumerate(self._clip_corners()):
            if (pos - h).manhattanLength() < HANDLE_SIZE + 4:
                return i
        return -1

    def _resize_clip(self, corner, n):
        c = self._clip
        l, t = c.x(), c.y()
        r, b = c.x() + c.width(), c.y() + c.height()
        nx = max(0.0, min(1.0, n.x())); ny = max(0.0, min(1.0, n.y()))
        if corner == 0:   l, t = nx, ny
        elif corner == 1: r, t = nx, ny
        elif corner == 2: r, b = nx, ny
        else:             l, b = nx, ny
        nl, nt = min(l, r), min(t, b)
        nw, nh = abs(r - l), abs(b - t)
        nw = min(nw, 1.0 - nl); nh = min(nh, 1.0 - nt)
        self._clip = QRectF(nl, nt, nw, nh)

    # ---- painting ----
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(30, 30, 30))

        if self._pixmap:
            dr = self._display_rect()
            clip = self._active_clip()
            if clip is None:
                p.drawPixmap(dr.toRect(), self._pixmap)
            else:
                Pw, Ph = self._pixmap.width(), self._pixmap.height()
                sx = max(0, min(Pw - 1, int(clip.x() * Pw)))
                sy = max(0, min(Ph - 1, int(clip.y() * Ph)))
                sw = max(1, min(Pw - sx, int(clip.width() * Pw)))
                sh = max(1, min(Ph - sy, int(clip.height() * Ph)))
                p.drawPixmap(dr.toRect(), self._pixmap, QRect(sx, sy, sw, sh))

        colours = [QColor(0, 200, 0), QColor(0, 150, 255), QColor(255, 100, 0),
                   QColor(255, 0, 200), QColor(255, 255, 0), QColor(0, 255, 255)]
        for idx, ann in enumerate(self.annotations):
            col = colours[ann.class_id % len(colours)]
            selected = (idx == self._selected_idx)
            p.setPen(QPen(col, 2 if not selected else 3))
            p.setBrush(QBrush(QColor(col.red(), col.green(), col.blue(), 40)))
            wpts = self._handles(ann)
            p.drawPolygon(QPolygonF(wpts))
            if wpts:
                p.setPen(QPen(Qt.white)); p.setFont(QFont("Arial", 10, QFont.Bold))
                p.drawText(wpts[0] + QPointF(4, -6), ann.class_name)
            if selected:
                for h in wpts:
                    p.setPen(QPen(Qt.white, 1)); p.setBrush(QBrush(col))
                    p.drawRect(QRectF(h.x() - HANDLE_SIZE / 2, h.y() - HANDLE_SIZE / 2,
                                      HANDLE_SIZE, HANDLE_SIZE))
                for m in self._edge_midpoints(ann):
                    p.setPen(QPen(Qt.white, 1)); p.setBrush(QBrush(Qt.gray))
                    p.drawEllipse(m, EDGE_HANDLE_SIZE / 2, EDGE_HANDLE_SIZE / 2)

        if self._drawing:
            p.setPen(QPen(QColor(255, 255, 0), 2, Qt.DashLine))
            p.setBrush(QBrush(QColor(255, 255, 0, 30)))
            p.drawRect(QRectF(self._draw_start, self._draw_end).normalized())

        # clip indicator (the view is zoomed to it)
        if self._clip is not None and self._clip.width() > 0 and self._clip.height() > 0:
            cw = self._clip_widget_rect()
            p.setBrush(Qt.NoBrush); p.setPen(QPen(CLIP_COLOR, 2)); p.drawRect(cw)
            for h in self._clip_corners():
                p.setPen(QPen(Qt.white, 1)); p.setBrush(QBrush(CLIP_COLOR))
                p.drawRect(QRectF(h.x() - HANDLE_SIZE / 2, h.y() - HANDLE_SIZE / 2,
                                  HANDLE_SIZE, HANDLE_SIZE))
            p.setPen(CLIP_COLOR); p.setFont(QFont("Segoe UI", 9, QFont.Bold))
            p.drawText(QPointF(cw.x() + 6, cw.y() + 16), "SAVE REGION (zoomed)")

        if self._clip_drag == "new":
            p.setPen(QPen(CLIP_COLOR, 2, Qt.DashLine))
            p.setBrush(QBrush(QColor(45, 212, 191, 25)))
            p.drawRect(QRectF(self._clip_draw_start, self._clip_draw_end).normalized())

        # paused banner (mode aware)
        if self.paused:
            bar_h = 30
            p.setPen(Qt.NoPen)
            p.fillRect(0, 0, self.width(), bar_h, QColor(18, 18, 22, 212))
            p.fillRect(0, bar_h, self.width(), 2, QColor(245, 176, 40))
            a = int((math.sin(time.time() * 5.0) * 0.5 + 0.5) * 170) + 50
            p.setBrush(QColor(245, 176, 40, a))
            p.drawEllipse(14, bar_h // 2 - 5, 10, 10)
            p.setPen(QColor(236, 238, 244)); p.setFont(QFont("Segoe UI", 10, QFont.DemiBold))
            if self.mode == "clip":
                if self._clip is None:
                    msg = "CLIP MODE - drag a box to set the save region (outside is excluded)"
                else:
                    msg = "CLIP SET - move it / drag corners, then switch to Annotate to label inside"
            else:
                msg = "CAPTURE PAUSED - draw / reshape, then Save, then Continue"
            p.drawText(32, 0, self.width() - 44, bar_h,
                       Qt.AlignVCenter | Qt.AlignLeft, msg)

        p.end()

    # ---- mouse ----
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = event.pos()
            if self.mode == "clip":
                if self._clip is not None:
                    ci = self._hit_clip_corner(pos)
                    if ci >= 0:
                        self._clip_drag = "resize"; self._clip_corner = ci; return
                    if self._clip_widget_rect().contains(pos):
                        self._clip_drag = "move"
                        n = self._widget_to_norm(pos)
                        self._clip_move_off = QPointF(n.x() - self._clip.x(),
                                                      n.y() - self._clip.y())
                        return
                self._clip_drag = "new"
                self._clip_draw_start = pos; self._clip_draw_end = pos
                self.freeze_requested.emit(); self.update(); return

            if self._selected_idx >= 0:
                ann = self.annotations[self._selected_idx]
                hi = self._hit_handle(pos, ann)
                if hi >= 0:
                    self._dragging_handle = hi; return
                ei = self._hit_edge(pos, ann)
                if ei >= 0:
                    npt = self._widget_to_norm(self._edge_midpoints(ann)[ei])
                    ann.points.insert(ei + 1, npt)
                    self._dragging_handle = ei + 1; self.update(); return

            for idx in reversed(range(len(self.annotations))):
                ann = self.annotations[idx]
                if QPolygonF(self._handles(ann)).containsPoint(pos, Qt.OddEvenFill):
                    self._selected_idx = idx; self.update(); return

            self._selected_idx = -1
            self.freeze_requested.emit()
            self._drawing = True; self._draw_start = pos; self._draw_end = pos
            self.update()

        elif event.button() == Qt.RightButton:
            pos = event.pos()
            if self._selected_idx >= 0:
                ann = self.annotations[self._selected_idx]
                hi = self._hit_handle(pos, ann)
                if hi >= 0 and len(ann.points) > MIN_POLY_POINTS:
                    ann.points.pop(hi); self._dragging_handle = -1
                    self.update(); return
            self._show_classify_menu(pos)

    def mouseMoveEvent(self, event):
        pos = event.pos()
        if self._clip_drag == "new":
            self._clip_draw_end = pos; self.update(); return
        if self._clip_drag == "move":
            n = self._widget_to_norm(pos)
            nx = max(0.0, min(1.0 - self._clip.width(), n.x() - self._clip_move_off.x()))
            ny = max(0.0, min(1.0 - self._clip.height(), n.y() - self._clip_move_off.y()))
            self._clip = QRectF(nx, ny, self._clip.width(), self._clip.height())
            self.clip_changed.emit(); self.update(); return
        if self._clip_drag == "resize":
            self._resize_clip(self._clip_corner, self._widget_to_norm(pos))
            self.clip_changed.emit(); self.update(); return
        if self._drawing:
            self._draw_end = pos; self.update(); return
        if self._dragging_handle >= 0 and self._selected_idx >= 0:
            ann = self.annotations[self._selected_idx]
            ann.points[self._dragging_handle] = self._widget_to_norm(pos)
            self.update(); return

        if self.mode == "clip":
            if self._clip is not None:
                ci = self._hit_clip_corner(pos)
                if ci in (0, 2): self.setCursor(QCursor(Qt.SizeFDiagCursor)); return
                if ci in (1, 3): self.setCursor(QCursor(Qt.SizeBDiagCursor)); return
                if self._clip_widget_rect().contains(pos):
                    self.setCursor(QCursor(Qt.SizeAllCursor)); return
            self.setCursor(QCursor(Qt.CrossCursor)); return

        if self._selected_idx >= 0:
            ann = self.annotations[self._selected_idx]
            if self._hit_handle(pos, ann) >= 0:
                self.setCursor(QCursor(Qt.ClosedHandCursor)); return
            if self._hit_edge(pos, ann) >= 0:
                self.setCursor(QCursor(Qt.CrossCursor)); return
        self.setCursor(QCursor(Qt.ArrowCursor))

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self._clip_drag == "new":
                self._clip_drag = None
                r = QRectF(self._clip_draw_start, self._clip_draw_end).normalized()
                if r.width() > 10 and r.height() > 10:
                    tl = self._widget_to_norm(r.topLeft())
                    br = self._widget_to_norm(r.bottomRight())
                    x = max(0.0, min(1.0, tl.x())); y = max(0.0, min(1.0, tl.y()))
                    w = max(0.0, min(1.0 - x, br.x() - tl.x()))
                    h = max(0.0, min(1.0 - y, br.y() - tl.y()))
                    self._clip = QRectF(x, y, w, h)
                else:
                    self._clip = None
                self.clip_changed.emit(); self.update(); return
            if self._clip_drag in ("move", "resize"):
                self._clip_drag = None; self.clip_changed.emit(); self.update(); return
            if self._drawing:
                self._drawing = False
                r = QRectF(self._draw_start, self._draw_end).normalized()
                if r.width() > 10 and r.height() > 10:
                    tl = self._widget_to_norm(r.topLeft())
                    br = self._widget_to_norm(r.bottomRight())
                    pts = [tl, QPointF(br.x(), tl.y()), br, QPointF(tl.x(), br.y())]
                    self._pending_points = pts
                    self._prompt_classify(pts)
                else:
                    if not self.annotations:
                        self.resume_requested.emit()
                self.update(); return
            if self._dragging_handle >= 0:
                self._dragging_handle = -1; self.update()

    def mouseDoubleClickEvent(self, event):
        pos = event.pos()
        for idx in reversed(range(len(self.annotations))):
            ann = self.annotations[idx]
            if QPolygonF(self._handles(ann)).containsPoint(pos, Qt.OddEvenFill):
                self._selected_idx = idx
                self._prompt_classify(ann.points, existing_idx=idx); return

    # ---- classify ----
    def _prompt_classify(self, points, existing_idx=-1):
        dlg = ClassifyDialog(list(self.classes.values()), self)
        if dlg.exec_() != QDialog.Accepted:
            return
        name = dlg.value
        if not name:
            return
        if name not in self.classes.values():
            cid = self._next_class_id; self.classes[cid] = name
            self._next_class_id += 1
        else:
            cid = [k for k, v in self.classes.items() if v == name][0]
        if existing_idx >= 0:
            self.annotations[existing_idx].class_id = cid
            self.annotations[existing_idx].class_name = name
        else:
            self.annotations.append(Annotation(points, cid, name))
            self._selected_idx = len(self.annotations) - 1
            self.annotation_added.emit(self.annotations[self._selected_idx])
        self.update()
        self.status_message.emit("Annotated '%s' (total: %d)" % (name, len(self.annotations)))

    def _show_classify_menu(self, pos):
        if self._selected_idx < 0:
            return
        menu = QMenu(self)
        act = menu.addAction("Re-classify selected")
        act_del = menu.addAction("Delete selected")
        chosen = menu.exec_(self.mapToGlobal(pos))
        if chosen == act:
            ann = self.annotations[self._selected_idx]
            self._prompt_classify(ann.points, existing_idx=self._selected_idx)
        elif chosen == act_del:
            self.annotations.pop(self._selected_idx)
            self._selected_idx = -1; self.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete and self._selected_idx >= 0:
            self.annotations.pop(self._selected_idx)
            self._selected_idx = -1; self.update()
        elif event.key() == Qt.Key_Escape:
            self._selected_idx = -1; self._drawing = False; self.update()