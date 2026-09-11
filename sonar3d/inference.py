from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class Detection:
    id: str
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_yolo(model_path: str | Path):
    """Load the user's Ultralytics checkpoint lazily."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Run: pip install -r requirements.txt"
        ) from exc
    return YOLO(str(model_path))


def predict(
    model,
    image: Image.Image,
    confidence: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 768,
    device: str | int | None = None,
) -> list[Detection]:
    rgb = np.asarray(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    kwargs = dict(
        source=bgr,
        conf=float(confidence),
        iou=float(iou),
        imgsz=int(imgsz),
        verbose=False,
    )
    if device not in (None, "auto"):
        kwargs["device"] = device

    result = model.predict(**kwargs)[0]
    names = result.names
    detections: list[Detection] = []

    if result.boxes is None:
        return detections

    for idx, box in enumerate(result.boxes):
        xyxy = box.xyxy[0].detach().cpu().numpy().astype(float)
        conf = float(box.conf[0].detach().cpu())
        cls_id = int(box.cls[0].detach().cpu())
        detections.append(
            Detection(
                id=f"ANOM-{idx + 1:04d}",
                class_id=cls_id,
                class_name=str(names.get(cls_id, cls_id)),
                confidence=conf,
                x1=float(xyxy[0]),
                y1=float(xyxy[1]),
                x2=float(xyxy[2]),
                y2=float(xyxy[3]),
            )
        )
    return detections


CLASS_COLORS = {
    "shipwreck": "#f59e0b",
    "pipeline_cable": "#22c55e",
    "ghost_net": "#38bdf8",
    "submarine_pipeline": "#a78bfa",
}


def draw_detections(image: Image.Image, detections: list[Detection]) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    for det in detections:
        color = CLASS_COLORS.get(det.class_name, "#ffffff")
        x1, y1, x2, y2 = det.bbox
        width = max(2, round(out.width / 350))
        draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
        label = f"{det.class_name}  {det.confidence * 100:.1f}%"
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = 180, 20
        ty = max(0, y1 - th - 8)
        draw.rounded_rectangle((x1, ty, x1 + tw + 10, ty + th + 6), radius=4, fill=color)
        draw.text((x1 + 5, ty + 2), label, fill="#071018", font=font)
    return out
