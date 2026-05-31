from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List

RELEVANT_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: np.ndarray
    center: np.ndarray
    track_id: int = -1

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> float:
        return self.width * self.height


class Detector:
    def __init__(self, model_name: str = "yolo11n.pt"):
        try:
            from ultralytics import YOLO
            self._model = YOLO(model_name)
            self._available = True
            print("[Detector] YOLOv11n loaded OK")
        except Exception as e:
            print(f"[Detector] YOLO not available ({e}) - using dummy detections")
            self._available = False

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if not self._available:
            return self._dummy_detections(frame)
        try:
            results = self._model.track(frame, persist=True, verbose=False)
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    cls_id     = int(box.cls[0])
                    class_name = self._model.names[cls_id]
                    if class_name not in RELEVANT_CLASSES:
                        continue
                    conf       = float(box.conf[0])
                    x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
                    cx         = (x1 + x2) / 2
                    cy         = (y1 + y2) / 2
                    track_id   = int(box.id[0]) if box.id is not None else -1
                    detections.append(Detection(
                        class_name=class_name,
                        confidence=conf,
                        bbox=np.array([x1, y1, x2, y2]),
                        center=np.array([cx, cy]),
                        track_id=track_id,
                    ))
            return detections
        except Exception as e:
            print(f"[Detector] Detection error: {e}")
            return self._dummy_detections(frame)

    def _dummy_detections(self, frame: np.ndarray) -> List[Detection]:
        h, w = frame.shape[:2]
        return [Detection(
            class_name="person",
            confidence=float(np.clip(0.72 + 0.1 * np.random.randn(), 0, 1)),
            bbox=np.array([w*0.4, h*0.2, w*0.5, h*0.8]),
            center=np.array([w*0.45, h*0.5]),
            track_id=1,
        )]