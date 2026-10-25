from pathlib import Path
from collections import Counter

images_dir = Path("dataset/images/train")
labels_dir = Path("dataset/labels/train")
fraction = 0.30

images = sorted(
    p for p in images_dir.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
)

n = round(len(images) * fraction)
selected = images[:n]

counts = Counter()

for image in selected:
    label = labels_dir / f"{image.stem}.txt"

    if not label.exists():
        counts["sin etiqueta"] += 1
        continue

    classes = set()

    for line in label.read_text().splitlines():
        fields = line.split()
        if fields:
            classes.add(int(float(fields[0])))

    if classes == {0}:
        counts["platano"] += 1
    elif classes == {1}:
        counts["banano"] += 1
    elif classes:
        counts["mixta"] += 1
    else:
        counts["sin objetos"] += 1

print(f"Total de imágenes: {len(images)}")
print(f"30 % seleccionado: {n}")
print("Distribución hipotética del primer 30 %:", counts)
print("Primera imagen:", selected[0].name if selected else "Ninguna")
print("Última imagen:", selected[-1].name if selected else "Ninguna")