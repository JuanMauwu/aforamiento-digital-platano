
import argparse
import os
import shutil
from pathlib import Path

LABEL_EXT = (".txt",)

def process_file(file_path, new_label):
    out_lines = []

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            
            parts[0] = new_label
            out_lines.append(" ".join(parts))
    
    with open(file_path, "w", encoding="utf-8") as of:
        of.write("\n".join(out_lines) + ("\n" if out_lines else ""))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Directorio de labels a cambiar")
    args = ap.parse_args()

    folder = Path(args.dir)
    
    if not os.path.isdir(folder):
        raise SystemExit(
        f"No se pudo encontrar la ruta ingresada: {folder}."
    )

    for path in folder.rglob("*"):
        if not path.suffix in LABEL_EXT:
            continue

        process_file(path, "1")

if __name__ == "__main__":
    main()
