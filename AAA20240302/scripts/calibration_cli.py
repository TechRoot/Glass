"""
Interfaz de línea de comandos para tareas de calibración.

Este script agrupa los algoritmos de homografía y Kabsch en una herramienta
de línea de comandos.  Lee archivos CSV con correspondencias de puntos y
muestra los parámetros de la transformación estimada.

Autor: A. Alonso
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import numpy as np

from ..lib.homography import compute_homography, apply_homography
from ..lib.kabsch import kabsch
from ..lib.transformations import homogeneous_transform


def load_points(path: Path) -> np.ndarray:
    with path.open(newline='') as f:
        reader = csv.reader(f)
        return np.array([[float(x) for x in row] for row in reader])


def main() -> None:
    parser = argparse.ArgumentParser(description="CLI de calibración: homografía y alineamiento rígido")
    sub = parser.add_subparsers(dest='command', required=True)

    hom = sub.add_parser('homography', help='Estimar una homografía a partir de correspondencias de puntos')
    hom.add_argument('src', type=Path, help='CSV con puntos de origen (N×2)')
    hom.add_argument('dst', type=Path, help='CSV con puntos de destino (N×2)')

    kab = sub.add_parser('kabsch', help='Calcular alineamiento rígido usando el algoritmo de Kabsch')
    kab.add_argument('src', type=Path, help='CSV con puntos 3D de origen (N×3)')
    kab.add_argument('dst', type=Path, help='CSV con puntos 3D de destino (N×3)')

    args = parser.parse_args()
    if args.command == 'homography':
        src = load_points(args.src)
        dst = load_points(args.dst)
        H = compute_homography(src, dst)
        print("Homografía estimada:\n", H)
    elif args.command == 'kabsch':
        src = load_points(args.src)
        dst = load_points(args.dst)
        R, t = kabsch(src, dst)
        T = homogeneous_transform(R, t)
        print("Rotación estimada:\n", R)
        print("Traslación estimada:", t)
        print("Transformación homogénea:\n", T)


if __name__ == '__main__':
    main()
