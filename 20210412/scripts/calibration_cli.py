#!/usr/bin/env python3
"""
CLI para tareas de calibración geométrica y transformaciones lineales.

Este script proporciona subcomandos para estimar homografías entre planos,
alinear nubes de puntos en 3D mediante el algoritmo de Kabsch y aplicar
transformaciones homogéneas definidas por una rotación y una traslación. Los
    subcomandos son de solo lectura por defecto; la opción `--confirm` queda
    reservada para futuras extensiones que generen salidas en disco.

Opciones globales:
  --dry-run   Realiza una simulación sin efectos (por defecto True).
  --confirm   Confirma operaciones que escriben resultados (actualmente no se usa).
  --log-level Nivel de log (INFO, DEBUG, ...).
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

from ..lib import transformations as tf
from ..lib import homography as hg
from ..lib import kabsch as kb


def configure_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=numeric, format="%(levelname)s: %(message)s")


def read_homography_points(file_path: str) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Lee un CSV con columnas x_src,y_src,x_dst,y_dst y devuelve listas de puntos.

    Args:
        file_path: ruta al fichero CSV.

    Returns:
        Una tupla (src, dst) de listas de puntos (x, y).
    """
    src = []
    dst = []
    with open(file_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            src.append((float(row['x_src']), float(row['y_src'])))
            dst.append((float(row['x_dst']), float(row['y_dst'])))
    return src, dst


def read_kabsch_points(file_path: str) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
    """Lee un CSV con columnas x_ref,y_ref,z_ref,x_mov,y_mov,z_mov.

    Returns:
        Una tupla (P, Q) con listas de puntos 3D.
    """
    P: List[Tuple[float, float, float]] = []
    Q: List[Tuple[float, float, float]] = []
    with open(file_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            P.append((float(row['x_ref']), float(row['y_ref']), float(row['z_ref'])))
            Q.append((float(row['x_mov']), float(row['y_mov']), float(row['z_mov'])))
    return P, Q


def cmd_homography(args) -> None:
    src, dst = read_homography_points(args.input)
    H = hg.estimate_homography(src, dst)
    np.set_printoptions(precision=6, suppress=True)
    print("Matriz de homografía estimada:")
    print(H)
    # aplicar homografía a puntos opcionales
    if args.apply:
        apply_points = []
        for pair in args.apply.split(';'):
            coords = [float(c) for c in pair.split(',') if c.strip()]
            if len(coords) != 2:
                logging.error(f"Punto inválido: {pair}")
                continue
            apply_points.append((coords[0], coords[1]))
        if apply_points:
            result = hg.apply_homography(apply_points, H)
            print("Resultados de aplicar la homografía:")
            for p, r in zip(apply_points, result):
                print(f"{p} → ({r[0]:.6f}, {r[1]:.6f})")


def cmd_kabsch(args) -> None:
    P, Q = read_kabsch_points(args.input)
    R, t, rmsd = kb.kabsch(P, Q)
    np.set_printoptions(precision=6, suppress=True)
    print("Matriz de rotación:")
    print(R)
    print("Vector de traslación:")
    print(t)
    print(f"Error RMSD: {rmsd:.6e}")
    if args.incremental:
        inc = kb.KabschIncremental()
        for p, q in zip(P, Q):
            inc.add_points([p], [q])
        result = inc.get_transform()
        if result is not None:
            Rinc, tinc = result
            print("Matriz de rotación (incremental):")
            print(Rinc)
            print("Vector de traslación (incremental):")
            print(tinc)


def cmd_transform(args) -> None:
    # Construir la rotación a partir de cuaternión o matriz
    if args.quaternion:
        try:
            q_vals = [float(v) for v in args.quaternion.split(',')]
            if len(q_vals) != 4:
                raise ValueError
        except ValueError:
            logging.error("El cuaternión debe tener cuatro componentes w,x,y,z")
            return
        R = tf.quaternion_to_rotation(np.array(q_vals))
    elif args.rotation:
        try:
            r_vals = [float(v) for v in args.rotation.split(',')]
            if len(r_vals) != 9:
                raise ValueError
        except ValueError:
            logging.error("La matriz de rotación debe contener nueve valores separados por coma")
            return
        R = np.array(r_vals).reshape(3, 3)
    else:
        logging.error("Debe especificar --quaternion o --rotation para definir la rotación")
        return
    # vector de traslación
    try:
        t_vals = [float(v) for v in args.translation.split(',')]
        if len(t_vals) != 3:
            raise ValueError
    except ValueError:
        logging.error("La traslación debe contener tres valores x,y,z")
        return
    # punto a transformar
    try:
        p_vals = [float(v) for v in args.point.split(',')]
        if len(p_vals) != 3:
            raise ValueError
    except ValueError:
        logging.error("El punto debe tener tres coordenadas x,y,z")
        return
    T = tf.homogeneous_transform(R, np.array(t_vals))
    res = tf.apply_transform(np.array(p_vals), T)
    print(f"Punto transformado: ({res[0]:.6f}, {res[1]:.6f}, {res[2]:.6f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="CLI para calibración geométrica y transformaciones lineales")
    parser.add_argument('--dry-run', action='store_true', default=True, help='Simula la ejecución sin escribir archivos')
    parser.add_argument('--confirm', action='store_true', help='Confirma operaciones que generan salidas (no usado)')
    parser.add_argument('--log-level', default='INFO', help='Nivel de logging (INFO, DEBUG, ...)')
    subparsers = parser.add_subparsers(dest='command', required=True)
    # subcomando homography
    p_h = subparsers.add_parser('homography', help='Estima una homografía 3×3 a partir de correspondencias 2D')
    p_h.add_argument('--input', required=True, help='Fichero CSV con columnas x_src,y_src,x_dst,y_dst')
    p_h.add_argument('--apply', help='Puntos a transformar: lista x,y; separados por punto y coma p.ej. "0.5,0.3;1,1"')
    p_h.set_defaults(func=cmd_homography)
    # subcomando kabsch
    p_k = subparsers.add_parser('kabsch', help='Calcula rotación y traslación óptimas entre nubes 3D')
    p_k.add_argument('--input', required=True, help='Fichero CSV con columnas x_ref,y_ref,z_ref,x_mov,y_mov,z_mov')
    p_k.add_argument('--incremental', action='store_true', help='Muestra resultados de actualización incremental')
    p_k.set_defaults(func=cmd_kabsch)
    # subcomando transform
    p_t = subparsers.add_parser('transform', help='Aplica una transformación SE(3) a un punto')
    p_t.add_argument('--rotation', help='Rotación 3×3 como 9 valores separados por coma (fila mayor)')
    p_t.add_argument('--quaternion', help='Cuaternión w,x,y,z que define la rotación')
    p_t.add_argument('--translation', required=True, help='Vector de traslación x,y,z')
    p_t.add_argument('--point', required=True, help='Punto a transformar x,y,z')
    p_t.set_defaults(func=cmd_transform)

    args = parser.parse_args()
    configure_logging(args.log_level)
    args.func(args)


if __name__ == '__main__':
    main()