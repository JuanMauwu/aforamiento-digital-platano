#ver que cada imagen tenga su label y viceversa.
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXTS = {".jpg", ".jpeg", ".png"}

for split in ("train", "val", "test"):
    imgs = {p.stem for p in (ROOT / "dataset/images" / split).glob("*") if p.suffix.lower() in EXTS}
    lbls = {p.stem for p in (ROOT / "dataset/labels" / split).glob("*.txt")}

    print(f"[{split}] imágenes: {len(imgs)} | labels: {len(lbls)}")
    #Imágenes sin label no es error (YOLO las toma como "fondo"), pero conviene saberlo
    print("  sin label :", sorted(imgs - lbls)[:5])
    #Labels sin imagen sí es problema: probablemente nombres distintos
    print("  sin imagen:", sorted(lbls - imgs)[:5])