"""Main application window - ties everything together."""

import sys
import os

# Load PyTorch BEFORE PyQt5 to avoid the c10.dll WinError 1114
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch  # noqa: E402,F401
from ultralytics import YOLO  # noqa: E402,F401

from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QDialog, QFileDialog,
    QMessageBox, QGroupBox, QSpinBox, QCheckBox, QSlider, QSplitter,
)
from PyQt5.QtCore import Qt

from screen_capture import list_screens, ScreenCaptureThread
from annotation_canvas import AnnotationCanvas
from training_manager import DatasetManager, TrainingThread, export_pt
from test_window import TestWindow


class ScreenSelectDialog(QDialog):
    def __init__(self, screens, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Screen")
        self.setFixedSize(350, 120)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Choose the monitor to capture:"))
        self.combo = QComboBox()
        for s in screens:
            self.combo.addItem("[%d] %s  (%dx%d)" %
                               (s["index"], s["name"], s["width"], s["height"]), s["index"])
        layout.addWidget(self.combo)
        btn = QPushButton("OK"); btn.clicked.connect(self.accept); layout.addWidget(btn)

    @property
    def selected_index(self):
        return self.combo.currentData()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO11 Screen Trainer")
        self.resize(1280, 800)

        self.monitor_index = -1
        self.capture_thread = None
        self.dataset = None
        self._last_frame = None
        self._best_pt = None
        self._training_thread = None
        self._test_window = None
        self._project_dir = ""
        self._capture_paused = False

        self._build_ui()
        self._build_menu()
        self.statusBar().showMessage("Select a screen to begin.")

    # --------------------------------------------------------------- UI
    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        self.canvas = AnnotationCanvas()
        self.canvas.annotation_added.connect(self._on_annotation)
        self.canvas.status_message.connect(lambda m: self.statusBar().showMessage(m, 4000))
        self.canvas.freeze_requested.connect(self._pause_capture)
        self.canvas.resume_requested.connect(self._resume_capture)
        self.canvas.clip_changed.connect(self._on_clip_changed)
        splitter.addWidget(self.canvas)

        panel = QGroupBox("Controls"); pl = QVBoxLayout(panel)

        pl.addWidget(QLabel("Annotations on screen:"))
        self.lbl_count = QLabel("0")
        self.lbl_count.setStyleSheet("font-size:20px; font-weight:bold;")
        pl.addWidget(self.lbl_count)

        self.btn_continue = QPushButton(">> Continue capture")
        self.btn_continue.setStyleSheet(
            "QPushButton{background:#1f7a4d;color:#eafff2;border:none;border-radius:8px;"
            "padding:9px 14px;font-weight:700;font-size:13px;}"
            "QPushButton:hover{background:#27a065;}QPushButton:pressed{background:#155c39;}")
        self.btn_continue.clicked.connect(self._resume_capture)
        self.btn_continue.setVisible(False); pl.addWidget(self.btn_continue)

        self.lbl_pause_hint = QLabel(
            "Stream is frozen. Clip a region (view zooms to it), label inside, Save, "
            "then Continue. Tick 'Keep clip + annotations' to reuse them.")
        self.lbl_pause_hint.setWordWrap(True)
        self.lbl_pause_hint.setStyleSheet("color:#9aa3b2; font-size:11px;")
        self.lbl_pause_hint.setVisible(False); pl.addWidget(self.lbl_pause_hint)

        self.chk_keep = QCheckBox("Keep clip + annotations on Continue")
        self.chk_keep.setChecked(False)
        self.chk_keep.setToolTip("When enabled, Continue leaves your clip region and "
                                 "annotations on screen so you can reuse them on new frames.")
        pl.addWidget(self.chk_keep)

        mode_row = QHBoxLayout(); mode_row.setSpacing(0)
        mlbl = QLabel("Mode:"); mlbl.setStyleSheet("font-weight:700;")
        mode_row.addWidget(mlbl); mode_row.addSpacing(6)
        self.btn_mode_annotate = QPushButton("Annotate")
        self.btn_mode_clip = QPushButton("Clip region")
        self.btn_mode_annotate.clicked.connect(lambda: self._set_canvas_mode("annotate"))
        self.btn_mode_clip.clicked.connect(lambda: self._set_canvas_mode("clip"))
        mode_row.addWidget(self.btn_mode_annotate); mode_row.addWidget(self.btn_mode_clip)
        pl.addLayout(mode_row)

        self.btn_clear_clip = QPushButton("Clear clip")
        self.btn_clear_clip.clicked.connect(self.canvas.clear_clip)
        self.btn_clear_clip.setEnabled(False); pl.addWidget(self.btn_clear_clip)

        self.lbl_clip_status = QLabel("No clip - full frame saved")
        self.lbl_clip_status.setWordWrap(True)
        self.lbl_clip_status.setStyleSheet("color:#8b93a7; font-size:11px;")
        pl.addWidget(self.lbl_clip_status)

        pl.addSpacing(10)

        copies_row = QHBoxLayout()
        copies_row.addWidget(QLabel("Copies:"))
        self.spin_copies = QSpinBox(); self.spin_copies.setRange(1, 50); self.spin_copies.setValue(1)
        self.sld_copies = QSlider(Qt.Horizontal); self.sld_copies.setRange(1, 50); self.sld_copies.setValue(1)
        self.spin_copies.valueChanged.connect(self.sld_copies.setValue)
        self.sld_copies.valueChanged.connect(self.spin_copies.setValue)
        copies_row.addWidget(self.spin_copies); copies_row.addWidget(self.sld_copies, 1)
        pl.addLayout(copies_row)
        self.chk_vary = QCheckBox("Vary duplicates (recommended)")
        self.chk_vary.setChecked(True); pl.addWidget(self.chk_vary)

        self.btn_save = QPushButton("Save Frame + Annotations")
        self.btn_save.clicked.connect(self._save_frame)
        self.btn_save.setEnabled(False); pl.addWidget(self.btn_save)

        self.btn_clear = QPushButton("Clear Annotations")
        self.btn_clear.clicked.connect(self._clear_annotations); pl.addWidget(self.btn_clear)

        pl.addSpacing(20); pl.addWidget(QLabel("Training:"))
        row = QHBoxLayout(); row.addWidget(QLabel("Epochs:"))
        self.spin_epochs = QSpinBox(); self.spin_epochs.setRange(5, 500)
        self.spin_epochs.setValue(100); row.addWidget(self.spin_epochs); pl.addLayout(row)

        row_m = QHBoxLayout(); row_m.addWidget(QLabel("Model:"))
        self.cmb_model = QComboBox()
        self.cmb_model.addItems(["n (nano - best for <500 imgs)", "s (small)",
                                 "m (medium)", "l (large - needs 1000+ imgs)", "x (xlarge)"])
        self.cmb_model.setCurrentIndex(0); row_m.addWidget(self.cmb_model); pl.addLayout(row_m)

        row_i = QHBoxLayout(); row_i.addWidget(QLabel("Image size:"))
        self.cmb_imgsz = QComboBox(); self.cmb_imgsz.addItems(["640", "1024", "1280"])
        self.cmb_imgsz.setCurrentIndex(2); row_i.addWidget(self.cmb_imgsz); pl.addLayout(row_i)

        self.chk_seg = QCheckBox("Use segmentation model (yolo11-seg)"); pl.addWidget(self.chk_seg)

        self.btn_train = QPushButton("Train YOLO11")
        self.btn_train.clicked.connect(self._start_training)
        self.btn_train.setEnabled(False); pl.addWidget(self.btn_train)

        self.btn_export = QPushButton("Export best.pt")
        self.btn_export.clicked.connect(self._export_pt)
        self.btn_export.setEnabled(False); pl.addWidget(self.btn_export)

        self.btn_load_model = QPushButton("Load External .pt Model")
        self.btn_load_model.clicked.connect(self._load_external_model); pl.addWidget(self.btn_load_model)

        self.btn_test = QPushButton("Open Live Test Window")
        self.btn_test.clicked.connect(self._open_test)
        self.btn_test.setEnabled(False); pl.addWidget(self.btn_test)

        pl.addStretch()
        self.lbl_status = QLabel(""); self.lbl_status.setWordWrap(True); pl.addWidget(self.lbl_status)

        panel.setMinimumWidth(260)
        splitter.addWidget(panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([1000, 320])

    def _build_menu(self):
        mb = self.menuBar()
        fm = mb.addMenu("File")
        fm.addAction("Open / New Project...").triggered.connect(self._open_project)
        fm.addAction("Load .pt Model for Testing...").triggered.connect(self._load_external_model)
        fm.addAction("Quit").triggered.connect(self.close)
        cm = mb.addMenu("Capture")
        cm.addAction("Select Screen...").triggered.connect(self._select_screen)
        cm.addAction("Stop Capture").triggered.connect(self._stop_capture)
        mb.addMenu("Help").addAction("Usage").triggered.connect(self._show_help)

    # --------------------------------------------------------------- mode buttons
    def _set_canvas_mode(self, m):
        self.canvas.set_mode(m)
        a = (m == "annotate")
        self.btn_mode_annotate.setStyleSheet(
            "QPushButton{background:#2b6cb0;color:#fff;border:1px solid #2b6cb0;font-weight:700;"
            "padding:6px 12px;border-top-left-radius:6px;border-bottom-left-radius:6px;}"
            if a else
            "QPushButton{background:#23272f;color:#c7cdd9;border:1px solid #333a45;padding:6px 12px;"
            "border-top-left-radius:6px;border-bottom-left-radius:6px;}QPushButton:hover{background:#2c323c;}")
        self.btn_mode_clip.setStyleSheet(
            "QPushButton{background:#2b6cb0;color:#fff;border:1px solid #2b6cb0;font-weight:700;"
            "padding:6px 12px;border-top-right-radius:6px;border-bottom-right-radius:6px;}"
            if not a else
            "QPushButton{background:#23272f;color:#c7cdd9;border:1px solid #333a45;padding:6px 12px;"
            "border-top-right-radius:6px;border-bottom-right-radius:6px;}QPushButton:hover{background:#2c323c;}")

    def _source_shape(self):
        full = getattr(self.capture_thread, "last_full", None) if self.capture_thread else None
        if full is not None:
            return full.shape[:2]
        if self._last_frame is not None:
            return self._last_frame.shape[:2]
        return (1, 1)

    def _on_clip_changed(self):
        clip = self.canvas._clip
        if clip is None:
            self.lbl_clip_status.setText("No clip - full frame saved")
            self.lbl_clip_status.setStyleSheet("color:#8b93a7; font-size:11px;")
            self.btn_clear_clip.setEnabled(False); return
        H, W = self._source_shape()
        pw = max(1, int(round(clip.width() * W)))
        ph = max(1, int(round(clip.height() * H)))
        self.lbl_clip_status.setText("Clip active - saved image %d x %d px" % (pw, ph))
        self.lbl_clip_status.setStyleSheet("color:#34d399; font-weight:700; font-size:11px;")
        self.btn_clear_clip.setEnabled(True)

    # --------------------------------------------------------------- project
    def _open_project(self):
        d = QFileDialog.getExistingDirectory(self, "Select Project Folder")
        if not d:
            return
        self._project_dir = d; self.dataset = DatasetManager(d)
        self.btn_save.setEnabled(True); self.btn_train.setEnabled(True)
        self.statusBar().showMessage("Project: %s  (%d images)" % (d, self.dataset.num_images))
        yaml_path = Path(d) / "data.yaml"
        if yaml_path.exists():
            import yaml
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
            if data and data.get("names"):
                self.canvas.classes = data["names"]
                self.canvas._next_class_id = max(data["names"].keys()) + 1

    # --------------------------------------------------------------- capture
    def _on_frame(self, frame):
        if self._capture_paused:
            return
        self._last_frame = frame.copy()
        self.canvas.update_frame(frame)

    def _stop_capture(self):
        if self.capture_thread:
            try:
                self.capture_thread.set_paused(False)
            except Exception:
                pass
            self.capture_thread.stop(); self.capture_thread = None
        self.btn_continue.setVisible(False); self.lbl_pause_hint.setVisible(False)
        self.canvas.set_paused(False)
        self.statusBar().showMessage("Capture stopped.")

    def _begin_capture(self, index, clear=True):
        self._stop_capture()
        self.monitor_index = index; self._capture_paused = False
        if clear:
            self._clear_annotations()
            self.canvas.clear_clip()
        self._set_canvas_mode("annotate")
        self.capture_thread = ScreenCaptureThread(index, fps=30)
        self.capture_thread.frame_ready.connect(self._on_frame)
        self.capture_thread.start()
        self.statusBar().showMessage("Capturing screen %d..." % index)

    def _select_screen(self):
        screens = list_screens()
        if not screens:
            QMessageBox.warning(self, "Error", "No screens detected."); return
        dlg = ScreenSelectDialog(screens, self)
        if dlg.exec_() != QDialog.Accepted:
            return
        self._begin_capture(dlg.selected_index)

    # --------------------------------------------------------------- annotations
    def _on_annotation(self, ann):
        self.lbl_count.setText(str(len(self.canvas.annotations)))

    def _clear_annotations(self):
        self.canvas.annotations.clear(); self.canvas._selected_idx = -1
        self.canvas.update(); self.lbl_count.setText("0")

    def _save_frame(self):
        full = getattr(self.capture_thread, "last_full", None) if self.capture_thread else None
        frame_to_save, anns = self.canvas.get_save_data(full)
        if frame_to_save is None or not anns:
            QMessageBox.information(self, "Info",
                                    "Nothing to save (draw annotations inside the clip).")
            return
        if self.dataset is None:
            self._open_project()
            if self.dataset is None:
                return
        n = self.spin_copies.value(); vary = self.chk_vary.isChecked()
        if hasattr(self.dataset, "save_duplicates"):
            saved = self.dataset.save_duplicates(frame_to_save, anns, self.canvas.classes,
                                                 count=n, vary=vary)
        else:
            for _ in range(n):
                self.dataset.save_annotation(frame_to_save, anns, self.canvas.classes)
            saved = n
        self.statusBar().showMessage("Saved %d copy(ies). Dataset now has %d images."
                                     % (saved, self.dataset.num_images))

    # --------------------------------------------------------------- freeze / continue
    def _set_paused_ui(self, paused):
        self.btn_continue.setVisible(paused); self.lbl_pause_hint.setVisible(paused)
        self.canvas.set_paused(paused)
        if paused:
            self.statusBar().showMessage("Capture paused - annotate / clip, then Save, then Continue.")
        else:
            self.statusBar().showMessage("Capture running.")

    def _pause_capture(self):
        if self._capture_paused:
            return
        self._capture_paused = True
        if self.capture_thread and self.capture_thread.isRunning():
            self.capture_thread.set_paused(True)
        self._set_paused_ui(True)

    def _resume_capture(self):
        self._capture_paused = False
        keep = self.chk_keep.isChecked()
        if not keep:
            self.canvas.clear_clip()
            self._clear_annotations()
        self._set_canvas_mode("annotate")
        self._set_paused_ui(False)
        if self.capture_thread and self.capture_thread.isRunning():
            self.capture_thread.set_paused(False)
        elif self.monitor_index != -1:
            self._begin_capture(self.monitor_index, clear=not keep)

    # --------------------------------------------------------------- training
    def _start_training(self):
        if self.dataset is None or self.dataset.num_images < 3:
            QMessageBox.warning(self, "Not enough data",
                                "Save at least 3 annotated frames before training."); return
        if self._training_thread and self._training_thread.isRunning():
            QMessageBox.information(self, "Info", "Training already running."); return
        self.btn_train.setEnabled(False); self.lbl_status.setText("Training started...")
        self._training_thread = TrainingThread(
            self.dataset, epochs=self.spin_epochs.value(),
            imgsz=int(self.cmb_imgsz.currentText()),
            model_size=self.cmb_model.currentText()[0],
            use_segmentation=self.chk_seg.isChecked())
        self._training_thread.progress.connect(lambda m: self.lbl_status.setText(m))
        self._training_thread.finished_ok.connect(self._on_train_done)
        self._training_thread.finished_err.connect(self._on_train_err)
        self._training_thread.start()

    def _on_train_done(self, best_path):
        self._best_pt = best_path
        self.btn_train.setEnabled(True); self.btn_export.setEnabled(True); self.btn_test.setEnabled(True)
        self.lbl_status.setText("Training complete!\n%s" % best_path)
        self.statusBar().showMessage("Training finished successfully.")

    def _on_train_err(self, err):
        self.btn_train.setEnabled(True); self.lbl_status.setText("Error: %s" % err)
        QMessageBox.critical(self, "Training Error", err)

    # --------------------------------------------------------------- export / load
    def _export_pt(self):
        if not self._best_pt:
            return
        dest, _ = QFileDialog.getSaveFileName(self, "Export best.pt", "best.pt", "PyTorch Model (*.pt)")
        if dest:
            export_pt(self._best_pt, dest); self.statusBar().showMessage("Exported to %s" % dest)

    def _load_external_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load YOLO Model", "", "PyTorch Model (*.pt)")
        if not path:
            return
        self._best_pt = path; self.btn_test.setEnabled(True)
        self.lbl_status.setText("Loaded external model:\n%s" % path)
        self.statusBar().showMessage("Loaded model: %s" % path)
        if self.monitor_index == -1:
            QMessageBox.information(self, "Select Screen",
                                    "A screen must be selected to test the model. Please select one now.")
            self._select_screen()
        else:
            self._open_test()

    # --------------------------------------------------------------- test window
    def _open_test(self):
        if not self._best_pt:
            QMessageBox.information(self, "Info", "Train a model first or load an external .pt file."); return
        if self.monitor_index == -1:
            QMessageBox.warning(self, "No Screen Selected", "Go to Capture -> Select Screen first."); return
        if self._test_window and self._test_window.isVisible():
            self._test_window.raise_(); self._test_window.activateWindow(); return
        self._stop_capture()
        self._test_window = TestWindow(self._best_pt, self.monitor_index, self,
                                       initial_roi=self.canvas._clip)
        self._test_window.window_closed.connect(self._on_test_closed)
        self._test_window.show()

    def _on_test_closed(self):
        self._test_window = None
        if self.monitor_index != -1:
            self._resume_capture()

    # --------------------------------------------------------------- help
    def _show_help(self):
        QMessageBox.information(self, "Usage", (
            "<h3>YOLO11 Screen Trainer</h3><ol>"
            "<li><b>File - Open Project</b> then <b>Capture - Select Screen</b>.</li>"
            "<li>Drawing pauses the stream; drag the divider to resize the preview.</li>"
            "<li><b>Clip region</b>: drag a box; the view zooms to it and hides the rest.</li>"
            "<li><b>Annotate</b>: label inside the clip; Save writes the crop.</li>"
            "<li><b>Copies</b>: save the same frame N times (Vary adds light augmentation).</li>"
            "<li><b>Keep clip + annotations</b>: Continue reuses them on new frames.</li>"
            "<li><b>Open Live Test Window</b> is a normal resizable window.</li>"
            "</ol><p>Keys: <b>Delete</b> removes selected; <b>Esc</b> deselects.</p>"))

    def closeEvent(self, event):
        self._stop_capture()
        if self._test_window:
            self._test_window.close()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()