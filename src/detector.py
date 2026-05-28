import numpy as np
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path="yolov8m.pt", conf_threshold=0.5):
        self.model = YOLO(model_path)
        self.model.to("cuda")
        self.conf_threshold = conf_threshold
        self.ball_conf_threshold = 0.3  # lower threshold for ball

    def detect(self, frame):
        """
        Returns:
            detections: np.ndarray (N, 5) -> [x1, y1, x2, y2, conf]
            crops:      list of N BGR arrays
            ball:       [cx, cy] or None
        """
        results = self.model(frame, verbose=False)
        detections = []
        crops      = []
        ball       = None

        h, w       = frame.shape[:2]
        frame_area = h * w

        for r in results:
            for box in r.boxes:
                cls  = int(box.cls[0])
                conf = float(box.conf[0])

                # ── Ball detection ────────────────────────────────
                if cls == 32 and conf >= self.ball_conf_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    ball = [int((x1+x2)/2), int((y1+y2)/2)]
                    continue

                # ── Player detection ──────────────────────────────
                if cls != 0:
                    continue
                if conf < self.conf_threshold:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                box_w = x2 - x1
                box_h = y2 - y1

                if box_h < 0.05 * h:
                    continue
                if (box_w * box_h) > 0.15 * frame_area:
                    continue
                if (box_w / box_h) > 1.2:
                    continue

                detections.append([x1, y1, x2, y2, conf])
                cx1, cy1 = max(0, int(x1)), max(0, int(y1))
                cx2, cy2 = min(w, int(x2)), min(h, int(y2))
                crops.append(frame[cy1:cy2, cx1:cx2])

        if detections:
            return np.array(detections), crops, ball
        return np.empty((0, 5)), [], ball