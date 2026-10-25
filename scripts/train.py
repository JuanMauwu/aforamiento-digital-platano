from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

#Modelo preentrenado pequeño: se descarga solo la primera vez (el .pt está en .gitignore)
model = YOLO("yolov5n.pt")

model.train(
    data=str(ROOT / "data.yaml"),  #ruta absoluta
    epochs=80,
    patience=15,
    imgsz=864,
    batch=8,       
    device="cpu", 
    project=str(ROOT / "runs"),
    name="first-run-all",
    fraction=1.00,
    max_det=600
)