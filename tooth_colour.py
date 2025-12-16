import sys
import os
import cv2
import torch
import numpy as np
import time
from ultralytics import YOLO
from PyQt5 import QtCore, QtGui, QtWidgets

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 640
MAX_CAMERAS = 4

class CameraThread(QtCore.QThread):
    frame_signal = QtCore.pyqtSignal(np.ndarray, int)

    def __init__(self, cam_id, process_callback, portrait_mode=False, frame_width=1920, frame_height=1080):
        super().__init__()
        self.cam_id = cam_id
        self.process_callback = process_callback
        self.running = True
        self.portrait_mode = portrait_mode
        self.frame_width = frame_width
        self.frame_height = frame_height

    def run(self):
        cap = cv2.VideoCapture(self.cam_id, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        if not cap.isOpened():
            print(f"[ERROR] Camera {self.cam_id} could not open!")
            return
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            if self.portrait_mode:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            try:
                frame, state = self.process_callback(frame)
            except Exception as e:
                print(f"[EXCEPTION] In process_callback: {e}")
                state = 0
            self.frame_signal.emit(frame, state)
            self.msleep(30)
        cap.release()

    def stop(self):
        self.running = False
        self.wait()

class VideoWindow(QtWidgets.QWidget):
    def __init__(self, overlay_frac=0.2):
        super().__init__()
        self.setWindowTitle("Webcam Output")
        self.resize(1920, 1080)
        self.label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.label.setScaledContents(False)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.label)

        self.overlay_sources = {}
        self.overlay_frac = overlay_frac
        self.fullscreen = False
        self.current_state = 0
        self.mouth_detect_start = None
        self.tooth_detect_start = None
        self.last_detection_time = time.time()

    def update_overlays(self, idle_path, smile_path, promo_path):
        for src in self.overlay_sources.values():
            if src['type'] == 'video' and src.get('cap'):
                src['cap'].release()
        self.overlay_sources.clear()

        for state, path in zip((0, 1, 2), (idle_path, smile_path, promo_path)):
            ext = os.path.splitext(path)[1].lower()
            if ext == '.gif':
                movie = QtGui.QMovie(path)
                movie.start()
                self.overlay_sources[state] = {'type': 'gif', 'movie': movie}
            elif ext in ('.mp4', '.avi', '.mov', '.mkv'):
                cap = cv2.VideoCapture(path)
                self.overlay_sources[state] = {'type': 'video', 'cap': cap}
            else:
                pix = QtGui.QPixmap(path)
                self.overlay_sources[state] = {'type': 'image', 'pixmap': pix}

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_F11:
            self.fullscreen = not self.fullscreen
            self.showFullScreen() if self.fullscreen else self.showNormal()

    def display_frame(self, frame, state):
        now = time.time()
        if state == 1:
            if self.mouth_detect_start is None:
                self.mouth_detect_start = now
            if now - self.mouth_detect_start >= 0.5:
                self.current_state = 1
            self.tooth_detect_start = None
            self.last_detection_time = now
        elif state == 2:
            if self.tooth_detect_start is None:
                self.tooth_detect_start = now
            if now - self.tooth_detect_start >= 1.0:
                self.current_state = 2
            self.mouth_detect_start = None
            self.last_detection_time = now
        else:
            self.mouth_detect_start = None
            self.tooth_detect_start = None

        if now - self.last_detection_time > 2.0:
            self.current_state = 0

        h, w = frame.shape[:2]
        img = QtGui.QImage(frame.data, w, h, 3 * w, QtGui.QImage.Format_BGR888)
        pix = QtGui.QPixmap.fromImage(img)
        pix = pix.scaled(self.label.width(), self.label.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

        src = self.overlay_sources.get(self.current_state)
        if src:
            if src['type'] == 'gif':
                overlay_pix = src['movie'].currentPixmap()
            elif src['type'] == 'video':
                cap = src['cap']
                ret, vframe = cap.read()
                if ret:
                    vh, vw = vframe.shape[:2]
                    qim = QtGui.QImage(vframe.data, vw, vh, 3 * vw, QtGui.QImage.Format_BGR888)
                    overlay_pix = QtGui.QPixmap.fromImage(qim)
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    overlay_pix = None
            else:
                overlay_pix = src['pixmap']

            if overlay_pix and not overlay_pix.isNull():
                ow = int(pix.width() * self.overlay_frac)
                oh = int(ow * overlay_pix.height() / overlay_pix.width())
                overlay_scaled = overlay_pix.scaled(ow, oh, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
                x = (pix.width() - overlay_scaled.width()) // 2
                y = 50
                painter = QtGui.QPainter(pix)
                painter.drawPixmap(x, y, overlay_scaled)
                painter.end()

        self.label.setPixmap(pix)

class InferenceApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Good Tooth")
        self.resize(400, 450)
        self._setup_models()
        self._build_ui()
        self.cam_thread = None
        self.video_window = None

    def _setup_models(self):
        base = getattr(sys, "_MEIPASS", os.getcwd())
        self.mouth_model = YOLO(os.path.join(base, "mouth_nano.pt"))
        self.seg_model = YOLO(os.path.join(base, "tooth_nano.pt"), task="segment")

    def _browse_file(self, line_edit):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Overlay File", "", "Media Files (*.png *.jpg *.jpeg *.bmp *.gif *.mp4 *.avi *.mov)")
        if path:
            line_edit.setText(path)

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        ctrl = QtWidgets.QFormLayout()

        self.mouth_thresh = QtWidgets.QDoubleSpinBox()
        self.mouth_thresh.setRange(0, 1)
        self.mouth_thresh.setValue(0.4)
        self.tooth_thresh = QtWidgets.QDoubleSpinBox()
        self.tooth_thresh.setRange(0, 1)
        self.tooth_thresh.setValue(0.3)
        self.hue_shift = QtWidgets.QSpinBox()
        self.hue_shift.setRange(0, 179)
        self.hue_shift.setValue(0)
        self.white_check = QtWidgets.QCheckBox("Whiten teeth")
        self.white_check.setChecked(True)
        self.orientation = QtWidgets.QComboBox()
        self.orientation.addItems(["Landscape", "Portrait"])

        self.frame_width = QtWidgets.QSpinBox()
        self.frame_width.setRange(320, 3840)
        self.frame_width.setValue(1920)
        self.frame_height = QtWidgets.QSpinBox()
        self.frame_height.setRange(240, 2160)
        self.frame_height.setValue(1080)

        self.idle_overlay = QtWidgets.QLineEdit()
        idle_browse = QtWidgets.QPushButton("Browse")
        idle_browse.clicked.connect(lambda: self._browse_file(self.idle_overlay))
        idle_row = QtWidgets.QHBoxLayout()
        idle_row.addWidget(self.idle_overlay)
        idle_row.addWidget(idle_browse)

        self.smile_overlay = QtWidgets.QLineEdit()
        smile_browse = QtWidgets.QPushButton("Browse")
        smile_browse.clicked.connect(lambda: self._browse_file(self.smile_overlay))
        smile_row = QtWidgets.QHBoxLayout()
        smile_row.addWidget(self.smile_overlay)
        smile_row.addWidget(smile_browse)

        self.promo_overlay = QtWidgets.QLineEdit()
        promo_browse = QtWidgets.QPushButton("Browse")
        promo_browse.clicked.connect(lambda: self._browse_file(self.promo_overlay))
        promo_row = QtWidgets.QHBoxLayout()
        promo_row.addWidget(self.promo_overlay)
        promo_row.addWidget(promo_browse)

        self.overlay_frac = QtWidgets.QDoubleSpinBox()
        self.overlay_frac.setRange(0.1, 1.0)
        self.overlay_frac.setSingleStep(0.1)
        self.overlay_frac.setValue(0.5)

        ctrl.addRow("Mouth Threshold:", self.mouth_thresh)
        ctrl.addRow("Tooth Threshold:", self.tooth_thresh)
        ctrl.addRow("Hue Shift:", self.hue_shift)
        ctrl.addRow(self.white_check)
        ctrl.addRow("Orientation:", self.orientation)
        ctrl.addRow("Frame Width:", self.frame_width)
        ctrl.addRow("Frame Height:", self.frame_height)

        # Camera selection
        self.cam_select = QtWidgets.QComboBox()
        for i in range(MAX_CAMERAS):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                self.cam_select.addItem(f"Camera {i} (ID:{i})", i)
                cap.release()
        if self.cam_select.count() == 0:
            self.cam_select.addItem("Default Camera (ID:0)", 0)
        ctrl.addRow("Select Webcam:", self.cam_select)

        # Overlay rows
        ctrl.addRow("Idle Overlay:", idle_row)
        ctrl.addRow("Smile Overlay:", smile_row)
        ctrl.addRow("Promo Overlay:", promo_row)
        ctrl.addRow("Overlay Size:", self.overlay_frac)

        layout.addLayout(ctrl)
        self.btn_webcam = QtWidgets.QPushButton("Start Webcam")
        self.btn_webcam.clicked.connect(self.run_webcam)
        layout.addWidget(self.btn_webcam)

    def run_webcam(self):
        if self.cam_thread and self.cam_thread.isRunning():
            self.cam_thread.stop()
            self.cam_thread = None
        if self.video_window:
            self.video_window.close()
            self.video_window = None

        cam_id = self.cam_select.currentData()
        portrait_mode = (self.orientation.currentText() == "Portrait")
        fw = self.frame_width.value()
        fh = self.frame_height.value()

        self.video_window = VideoWindow(overlay_frac=self.overlay_frac.value())
        self.video_window.update_overlays(
            self.idle_overlay.text(),
            self.smile_overlay.text(),
            self.promo_overlay.text()
        )
        self.video_window.show()

        self.cam_thread = CameraThread(cam_id, self.process_frame, portrait_mode, fw, fh)
        self.cam_thread.frame_signal.connect(self.video_window.display_frame)
        self.cam_thread.start()

    def closeEvent(self, event):
        if self.cam_thread and self.cam_thread.isRunning():
            self.cam_thread.stop()
        if self.video_window:
            self.video_window.close()
        event.accept()

    def process_frame(self, frame):
        orig_h, orig_w = frame.shape[:2]
        state = 0
        frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        res = self.mouth_model.predict(source=frame_resized, imgsz=IMG_SIZE, device=DEVICE, verbose=False)[0]
        mouth_box = None
        scale_x, scale_y = orig_w / IMG_SIZE, orig_h / IMG_SIZE
        for bbox, conf, cls_ in zip(res.boxes.xyxy, res.boxes.conf, res.boxes.cls):
            if conf >= self.mouth_thresh.value() and res.names[int(cls_)].lower() == "mouth":
                x1, y1, x2, y2 = bbox.tolist()
                x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                mouth_box = (x1, y1, x2, y2)
                state = 1
                break
        if not mouth_box:
            return frame, state

        x1, y1, x2, y2 = mouth_box
        x1, x2 = np.clip([x1, x2], 0, orig_w)
        y1, y2 = np.clip([y1, y2], 0, orig_h)
        if x2 <= x1 or y2 <= y1:
            return frame, state

        crop = frame[y1:y2, x1:x2]
        crop_resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
        seg = self.seg_model.predict(source=crop_resized, imgsz=IMG_SIZE, device=DEVICE, verbose=False)[0]
        masks = []
        mouth_area = (x2 - x1) * (y2 - y1)
        for mask, conf in zip(seg.masks.data.cpu().numpy(), seg.boxes.conf.cpu().numpy()):
            if conf >= self.tooth_thresh.value():
                mask_resized = cv2.resize(mask, (x2 - x1, y2 - y1))
                if np.sum(mask_resized > 0.05) < 0.5 * mouth_area:
                    masks.append(mask_resized)
        if masks:
            state = 2
            frame = self._apply_hue(frame, masks, (y1, x1))
        return frame, state

    def _apply_hue(self, frame, masks, offset):
        y_off, x_off = offset
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        mask_map = np.zeros_like(h, bool)
        for m in masks:
            mask_map[y_off:y_off + m.shape[0], x_off:x_off + m.shape[1]] |= (m > 0.5)
        if self.white_check.isChecked():
            s[mask_map] = 0
        else:
            h[mask_map] = (h[mask_map] + self.hue_shift.value()) % 180
        return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = InferenceApp()
    win.show()
    sys.exit(app.exec_())
