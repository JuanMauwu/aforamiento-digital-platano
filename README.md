# aforamiento-digital-platano

## Merge Dataset Script

Este script busca combinar todas las imagenes y labels en la carpeta centralizada del repositorio. El uso es mediante argumentos en el llamado del script en consola. Tiene como argumentos: dir: cadena con el directorio absoluto de la carpeta centralizada donde descargaron los datasets. --sim (simulación) (IMPORTANTE PRIMERA CORRIDA) añadir en la primera corrida para que evaluen que si encuentra todos los subdirectorios y se mueven todos los archivos correctamente. Eliminar cuando se este seguro de mover. Asegurarse de descargar los datasets con el formato YOLO. cualquiera de las estructuras YOLO funciona. Las etiquetas del dataset procesado combinando mosaico, recortando y etiquetando manualmente en Roboflow deben cambiar al valor de banano: 1. Usar script de Thomas primero sobre estas etiquetas antes de mover.

```bash
python scripts/merge_dataset.py --dir Ruta/absoluta/al/directorio --sim
python scripts/merge_dataset.py --dir Ruta/absoluta/al/directorio 
```
o

```bash
python3 scripts/merge_dataset.py --dir Ruta/absoluta/al/directorio --sim
python3 scripts/merge_dataset.py --dir Ruta/absoluta/al/directorio 
```