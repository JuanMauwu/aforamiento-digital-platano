from ultralytics import YOLO
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

model = YOLO(str(ROOT / "models" / "model-full-data.pt"))

metrics = model.val(
    data=str(ROOT / "data.yaml"), 
    split="test",
    imgsz=864,
    batch=8,       
    device="cpu", 
    max_det=600
)