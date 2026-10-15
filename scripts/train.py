from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

#Modelo preentrenado pequeño: se descarga solo la primera vez (el .pt está en .gitignore)
model = YOLO("yolov8n.pt")

model.train(
    data=str(ROOT / "data.yaml"),  #ruta absoluta
    imgsz=640,
    batch=8,       
    device="cpu", 
    project=str(ROOT / "runs"),
    name="macolla_v1",
)