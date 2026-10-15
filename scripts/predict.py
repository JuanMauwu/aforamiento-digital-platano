from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

model = YOLO(ROOT / "runs/macolla_v1/weights/best.pt")

results = model.predict(
    source=ROOT / "dataset/images/test",
    conf=0.25,  
    save=True,
)

#Para el aforo: cantidad de macollas detectadas por imagen
for r in results:
    print(Path(r.path).name, "->", len(r.boxes), "macollas")