from __future__ import annotations

import argparse
import os
from pathlib import Path
import cv2
import albumentations as A

# --- CONFIGURACIÓN ---
CLASS_OF_INTEREST = 1     # La clase que SÍ debe estar
OTHER_CLASS = 0    # La clase que NO debe estar
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# Definir las transformaciones (ejemplo: volteo horizontal y brillo)
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Blur(blur_limit=3, p=0.1),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def find_dirs(root):
    """Returns a list of the different sub folders"""
    if not os.path.isdir(root):
        raise SystemExit(
            f"No se pudo encontrar la ruta ingresada: {root}."
        )

    folders = []
    
    for path in root.glob("*"):
        if os.path.isdir(path):
            folders.append(path)
    
    return folders

def identify_yolo(src_dir):
    # Identifies the YOLO  folder structure of the src_dir dataset
    # and returns the train image and labels subdirectories
    sub_dirs = []
    # pattern: src_dir/images/train, src_dir/labels/train
    img = src_dir / "images" / "train"
    lbl = src_dir / "labels" / "train"
    if img.is_dir() and lbl.is_dir():
        sub_dirs = [img, lbl]
    if sub_dirs:
        return sub_dirs
    
    # pattern: src_dir/train/images, src_/train/labels
    img = src_dir / "train" / "images"
    lbl = src_dir / "train" / "labels"
    if img.is_dir() and lbl.is_dir():
        sub_dirs = [img, lbl]
    if sub_dirs:
        return sub_dirs

    # pattern: src_dir/train
    img_lbl = src_dir / "train"
    if img_lbl.is_dir():
        sub_dirs = [img_lbl, img_lbl]

    return sub_dirs

def process_dataset(dataset_dir):
    sub_dirs = identify_yolo(dataset_dir)
    img_dir = sub_dirs[0]
    labels_dir = sub_dirs[1]

    new_images = []
    new_labels = []
    added = 0

    for img_path in img_dir.glob("*"):
        if not img_path.is_file():
            continue
        
        if not img_path.suffix in IMG_EXT:
            continue
            
        base_name = img_path.stem
        label_name = f"{base_name}.txt"
        label_path = os.path.join(labels_dir, label_name)

        if not Path(label_path).is_file():
            continue
        
        classes, bboxes = read_labels(label_path)

        # CONDICIÓN: Tiene la clase deseada Y NO tiene la clase a evitar
        if (CLASS_OF_INTEREST in classes) and (OTHER_CLASS not in classes):
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 2. Aplicar Data Augmentation
            for i in range(3): 
                transformed = transform(image=image, bboxes=bboxes, class_labels=classes)
                transformed_image = transformed['image']
                transformed_bboxes = transformed['bboxes']
                transformed_classes = transformed['class_labels']
                
                # 3. Guardar nueva imagen y etiqueta con un prefijo
                new_base_name = f"aug_{i}_{base_name}"
                new_images.append(f"{new_base_name}.jpg")
                new_labels.append(f"{new_base_name}.txt")
                added += 1
                # Guardar Imagen
                new_img_path = os.path.join(img_dir, f"{new_base_name}.jpg")
                cv2.imwrite(new_img_path, cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR))
                
                # Guardar Etiqueta (.txt)
                new_label_path = os.path.join(labels_dir, f"{new_base_name}.txt")
                with open(new_label_path, 'w') as f:
                    for cls, bbox in zip(transformed_classes, transformed_bboxes):
                        f.write(f"{cls} {' '.join(map(str, bbox))}\n")
        
    return added, new_images, new_labels

def read_labels(label_path):
    bboxes = []
    classes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    cls = int(parts[0])
                    bbox = [float(x) for x in parts[1:]]
                    classes.append(cls)
                    bboxes.append(bbox)
    return classes, bboxes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Directorio raiz con datasets para aumentar")
    args = ap.parse_args()

    origin_folder = Path(args.dir)
    folders = find_dirs(origin_folder)

    if not folders:
        raise SystemExit(
            f"No hay directorios dentro del directorio: {origin_folder}. "
        )

    summary = {}
    # --- PROCESAMIENTO ---
    for folder in folders:
        summary[folder.name] = process_dataset(folder)

    # Print Summary
    for key, values in summary.items():
        print(f"\nCarpeta: {key}\n")
        print(f"---{values[0]} nuevas imagenes y etiquetas:\n")
        images = values[1]
        labels = values[2]
        for i in range(len(images)):
            print(f"{images[i]} - {labels[i]}")

if __name__ == "__main__":
    main()