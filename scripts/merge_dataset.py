"""Combines the differents datasets in the YOLO structure
    L dataset
      L images 
         L test
         L train
         L val
      L labels
         L test
         L train
         L val
Usage: python scripts/merge_dataset.py --dir original / folder / with / datasets --sim
original_dir: Absolute path of the folder with the differents datasets
"""
from __future__ import annotations

import argparse
import os
import shutil
import math
from pathlib import Path
import time
import sys

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
LABEL_EXT = (".txt",)

def move_lines_up(n):
    sys.stdout.write(f"\033[F" * n)
    sys.stdout.flush()

def clean_screen_down():
    sys.stdout.write("\033[J")
    sys.stdout.flush()

def clean_screen(last_overflow):
    lines = 4 + last_overflow
    move_lines_up(lines)
    clean_screen_down()

def print_proccess(folder, partition, sub_dirs, info, is_first, last_overflow):
    if not is_first:
        clean_screen(last_overflow)

    print(f"Moviendo archivos carpeta: {folder}")
    print(f"--Particion: {partition}")

    console_width = shutil.get_terminal_size().columns
    overflow = math.ceil(len(f"---Subdiretorios encontrados: {sub_dirs[0]}, {sub_dirs[1]}")/console_width)-1
    overflow += math.ceil(len(info)/console_width)-1
    print(f"---Subdiretorios encontrados: {sub_dirs[0]}, {sub_dirs[1]}\n")
    print(info, end="\r")
    time.sleep(0.01)

    return overflow


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
        lbl = src_dir / "labels" / split
        if img.is_dir() and lbl.is_dir():
            splits[split] = (img, lbl)
    if splits:
        return splits
    
    # pattern: src_dir/train/images, src_/train/labels
    for split in ("train", "val", "valid", "test"):
        img = src_dir / split / "images"
        lbl = src_dir / split / "labels"
        if img.is_dir() and lbl.is_dir():
            splits[split] = (img, lbl)
    if splits:
        return splits

    # pattern: src_dir/train, src_dir/test, src_dir/valid
    for split in ("train", "val", "valid", "test"):
        img_lbl = src_dir / split
        if img_lbl.is_dir():
            splits[split] = (img_lbl, img_lbl)
    return splits

def move_images(img_dir, destination_dir, info, simulation):

    img_moved = 0
    last_overflow = 0
    is_first = True

    dataset = info[0]
    partition = info[1]
    sub_dirs = info[2]

    for path in img_dir.glob("*"):
        if not path.is_file():
            continue
        
        if not path.suffix in IMG_EXT:
            continue

        img_moved += 1
        file_destination = destination_dir / path.name
        
        if simulation:
            info_print = f" Simulación: Moviendo: {dataset}/.../{path.name} -> dataset/{partition}/{file_destination.name}"
        else:
            if file_destination.exists():
                file_destination.unlink()
                
            shutil.move(str(path), str(file_destination))
            info_print = f"[MOVIDO] {dataset}/.../{path.name} -> dataset/{partition}/{file_destination.name}"

        last_overflow = print_proccess(dataset, partition, sub_dirs, info_print, is_first, last_overflow)
        if is_first:
            is_first = False

    clean_screen(last_overflow)
    return img_moved

def move_labels(label_dir, destination_dir, info, simulation):

    labels_moved = 0
    last_overflow = 0
    is_first = True

    dataset = info[0]
    partition = info[1]
    sub_dirs = info[2]

    for path in label_dir.glob("*"):
        if not path.is_file():
            continue
        if not path.suffix in LABEL_EXT:
            continue
        
        labels_moved += 1
        file_destination = destination_dir / path.name
        
        if simulation:
            info_print = f" Simulación: Moviendo: {dataset}/.../{path.name} -> {dataset}/{partition}/{file_destination.name}"
        else:
            shutil.move(str(path), str(file_destination))
            info_print = f"[MOVIDO] {dataset}/.../{path.name} -> {dataset}/{partition}/{file_destination.name}"
        
        last_overflow = print_proccess(dataset, partition, sub_dirs, info_print, is_first, last_overflow)
        if is_first:
            is_first = False
    
    clean_screen(last_overflow)
    return labels_moved

def move_dataset(src_dir, simulation):

    splits_folders = identify_yolo(src_dir)
    img_moved = {}
    labels_moved = {}

    for key, folders in splits_folders.items():
        folder_name = key

        if folder_name =="valid":
            folder_name = "val" 

        img_destination = destination = DATASET_DIR / "images" / folder_name 
        labels_destination = DATASET_DIR / "labels" / folder_name 

        info = (src_dir.name, folder_name, folders)
        
        img_moved[folder_name] = move_images(folders[0], img_destination, info, simulation)
        labels_moved[folder_name] = move_labels(folders[1], labels_destination, info, simulation)
    
    return img_moved, labels_moved

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Directorio original de los datasets")
    ap.add_argument("--sim", required=False, action="store_true", help="Realiza simulación del movimiento")
    args = ap.parse_args()

    origin_folder = Path(args.dir)
    simulation = args.sim
    folders = find_dirs(origin_folder)

    if not folders:
        raise SystemExit(
            f"No hay directorios dentro del directorio: {args.root}. "
        )

    summary = {}

    for folder in folders:
        summary[folder.name] = move_dataset(folder, simulation)

    print("\n------------RESUMEN DE RESULTADOS------------------\n")
    for key, data in summary.items():
        print(f"Resultados para carpeta {key}")
        for sub_key in data[0].keys():
            print(f"--Partición: {sub_key}")
            print(f"----imagenes movidas: {data[0][sub_key]}, labels movidos: {data[1][sub_key]}\n")

if __name__ == "__main__":
    main()
