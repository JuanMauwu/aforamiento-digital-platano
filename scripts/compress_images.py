from __future__ import annotations

import argparse
import os
from pathlib import Path
import cv2

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

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
    splits = {}
    
    # pattern: src_dir/images/train, src_dir/labels/train
    for split in ("train", "val", "valid", "test"):
        img = src_dir / "images" / split
        if img.is_dir():
            splits[split] = f"images/{split}"
    if splits:
        return splits
    
    # pattern: src_dir/train/images, src_/train/labels
    for split in ("train", "val", "valid", "test"):
        img = src_dir / split / "images"
        if img.is_dir():
            splits[split] = f"{split}/images"
    if splits:
        return splits

    # pattern: src_dir/train, src_dir/test, src_dir/valid
    for split in ("train", "val", "valid", "test"):
        img_lbl = src_dir / split
        if img_lbl.is_dir():
            splits[split] = split
    return splits

def process_dataset(dataset_dir, out_dir):

    # makedir if not exist
    os.makedirs(out_dir, exist_ok=True)

    splits = identify_yolo(dataset_dir)

    resized_images = {}
    skipped_images = {}
    resized = {}
    total = {}

    for partition, folder in splits.items():

        # makedir if not exist
        out_folder_dir = out_dir / Path(folder)
        os.makedirs(out_folder_dir, exist_ok=True)

        img_folder = dataset_dir / Path(folder)

        rsz_img = 0
        total_img = 0
        rsz_images = []
        skp_images = []
        for img_path in img_folder.glob("*"):
            if not img_path.is_file():
                continue
            
            if not img_path.suffix in IMG_EXT:
                continue

            total_img += 1    
            base_name = img_path.name
            new_img_path = out_folder_dir / base_name

            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, _ = image.shape

            if height > width:
                ratio = width/height
                height = 1024
                width = int(round(ratio*height, 0))
            else:
                ratio = height/width
                width = 1024
                height = int(round(ratio*width, 0))
            
            dsize = (width, height)

            try:
                resized_img = cv2.resize(image,dsize, interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(new_img_path, cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR))
                rsz_images.append(base_name)
                rsz_img += 1
            except IOError:
                print(f"No se pudo escribir el archivo {base_name}.jpg")
                skp_images.append(base_name)
        
        total[partition] = total_img
        resized_images[partition] = rsz_images
        skipped_images[partition] = skp_images
        resized[partition] = rsz_img
        
    return total, resized, resized_images, skipped_images

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Directorio raiz con imagenes a redimensionar")
    ap.add_argument("--out", required=True, help="Directorio de salida para guardar imagenes redimensionadas")
    args = ap.parse_args()

    origin_folder = Path(args.dir)
    output_folder = Path(args.out)
    folders = find_dirs(origin_folder)

    if not folders:
        raise SystemExit(
            f"No hay directorios dentro del directorio: {origin_folder}. "
        )

    summary = {}
    # --- PROCESAMIENTO ---
    for folder in folders:
        out_folder = output_folder / folder
        summary[folder.name] = process_dataset(folder, out_folder)

    # Print Summary
    for key, values in summary.items():
        print(f"\nCarpeta: {key}\n")
        for partition in values[0].keys():
            print(f"--Partición {partition}--")
            print(f"---{values[1][partition]}/{values[0][partition]} Imagenes redimensionadas:\n")
            resized = values[2][partition]
            skipped = values[3][partition]

            print("Resized images:")
            for i in range(len(resized)):
                print(f"img {i+1}:{resized[i]}")

            print("Skipped images:")
            for i in range(len(skipped)):
                print(f"img {i+1}:{skipped[i]}")

if __name__ == "__main__":
    main()