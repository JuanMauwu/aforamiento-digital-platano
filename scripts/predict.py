from __future__ import annotations

from pathlib import Path
from ultralytics import YOLO
import argparse
import json
import os

ROOT = Path(__file__).resolve().parent.parent
model = YOLO(ROOT / "models" / "model-full-data.pt")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit(
            "Falta ultralytics. Instala: pip install -e .[deep]"
        ) from e

    os.makedirs(args.out, exist_ok=True)
    results = model.predict(args.image, 
        conf=args.conf,
        imgsz=864,
        device="cpu",
        iou=0.30,
        max_det=600,
        save=True, 
        project=args.out, 
        name="pred")

    counts = {}
    for r in results:
        if r.boxes is None:
            continue
        for c in r.boxes.cls.tolist():
            name = model.names[int(c)]
            counts[name] = counts.get(name, 0) + 1
        r.show()
    print(json.dumps({"conteo": counts}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

'''
results = model.predict(
    source=ROOT / "dataset/images/test",
    conf=0.25,  
    save=True,
)

#Para el aforo: cantidad de macollas detectadas por imagen
for r in results:
    print(Path(r.path).name, "->", len(r.boxes), "macollas")
'''