
from pathlib import Path
import cv2
from ultralytics import YOLO

# Ajustar a las rutas reales de tu proyecto
MODEL_PATH = "models/model-full-data.pt"
IMAGES_DIR = Path("dataset/images/test")
LABELS_DIR = Path("dataset/labels/test")
OUTPUT_DIR = Path("auditoria_falsos_positivos")

IMG_SIZE = 864
CONF = 0.30
IOU_MATCH = 0.50
MAX_DET = 600

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model = YOLO(MODEL_PATH)


def load_labels(label_path, width, height):
    """Lee etiquetas YOLO: class x_center y_center width height."""
    boxes = []

    if not label_path.exists():
        return boxes

    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue

        cls, xc, yc, bw, bh = map(float, parts)

        x1 = (xc - bw / 2) * width
        y1 = (yc - bh / 2) * height
        x2 = (xc + bw / 2) * width
        y2 = (yc + bh / 2) * height

        boxes.append({
            "cls": int(cls),
            "xyxy": [x1, y1, x2, y2],
            "matched": False
        })

    return boxes


def box_iou(a, b):
    """Intersección sobre unión entre dos cajas xyxy."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


images = sorted(
    p for p in IMAGES_DIR.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
)

total_fp = 0
total_predictions = 0

for image_path in images:
    image = cv2.imread(str(image_path))
    if image is None:
        print("No se pudo leer:", image_path)
        continue

    height, width = image.shape[:2]
    label_path = LABELS_DIR / f"{image_path.stem}.txt"
    gt_boxes = load_labels(label_path, width, height)

    # No usar save=True: guardaremos nuestras propias anotaciones.
    result = model.predict(
        source=image,
        iou=0.30,
        imgsz=IMG_SIZE,
        conf=CONF,
        max_det=MAX_DET,
        device="cpu",
        verbose=False
    )[0]

    predictions = []

    if result.boxes is not None:
        for b in result.boxes:
            predictions.append({
                "cls": int(b.cls.item()),
                "conf": float(b.conf.item()),
                "xyxy": b.xyxy[0].cpu().tolist()
            })

    # Primero se asignan las predicciones de mayor confianza.
    predictions.sort(key=lambda p: p["conf"], reverse=True)

    false_positives = []

    for pred in predictions:
        total_predictions += 1
        candidates = [
            gt for gt in gt_boxes
            if not gt["matched"] and gt["cls"] == pred["cls"]
        ]

        best_gt = None
        best_iou = 0.0

        for gt in candidates:
            score = box_iou(pred["xyxy"], gt["xyxy"])
            if score > best_iou:
                best_iou = score
                best_gt = gt

        if best_gt is not None and best_iou >= IOU_MATCH:
            best_gt["matched"] = True
        else:
            false_positives.append(pred)

    # Verde = etiquetas reales; rojo = predicciones sin coincidencia.
    annotated = image.copy()

    for gt in gt_boxes:
        x1, y1, x2, y2 = map(int, gt["xyxy"])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated, f"GT {gt['cls']}",
            (x1, max(15, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )

    for pred in false_positives:
        x1, y1, x2, y2 = map(int, pred["xyxy"])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            annotated,
            f"FP {pred['cls']} {pred['conf']:.2f}",
            (x1, max(15, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
        )

    total_fp += len(false_positives)

    # Guardar solamente imágenes con predicciones sin coincidencia.
    if false_positives:
        output_path = OUTPUT_DIR / f"{image_path.stem}_audit.jpg"
        cv2.imwrite(str(output_path), annotated)
        print(
            f"{image_path.name}: {len(false_positives)} "
            f"predicciones sin coincidencia -> {output_path}"
        )

print("\n--- Resumen ---")
print("Imágenes procesadas:", len(images))
print("Predicciones examinadas:", total_predictions)
print("Predicciones sin coincidencia:", total_fp)
print("Imágenes guardadas en:", OUTPUT_DIR.resolve())
