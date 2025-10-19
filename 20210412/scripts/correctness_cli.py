#!/usr/bin/env python3
"""CLI para ejecutar comprobaciones de corrección y seguridad.

Este script ofrece subcomandos que invocan funciones de `lib/correctness.py`
para verificar propiedades de algoritmos y datos: terminación de máquinas de
estados mediante matrices nilpotentes, conservación de medidas en matrices
de rotación, optimalidad de asignaciones, validación de inversos modulares
y corrección de códigos CRC.  Por defecto opera en modo `--dry-run`, que
simplemente muestra los resultados en pantalla; con la opción `--confirm`
puede guardar informes en `data/traces/`.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import List, Tuple

from 20210412.lib import correctness as corr


def load_matrix(path: str) -> List[List[float]]:
    """Carga una matriz desde un fichero CSV o JSON.

    Si el archivo termina en `.json`, se espera una lista de listas.  Si
    termina en `.csv`, se leen filas separadas por comas.  Se devuelve
    siempre una lista de listas de float.
    """
    if path.lower().endswith(".json"):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return [[float(x) for x in row] for row in data]
    else:
        matrix: List[List[float]] = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                matrix.append([float(x) for x in row])
        return matrix


def cmd_nilpotent(args: argparse.Namespace) -> None:
    mat = load_matrix(args.input)
    # Convertir a int para producto booleano si todos los valores son 0/1
    int_mat = [[int(x) for x in row] for row in mat]
    ok, steps = corr.check_nilpotent_and_steps(int_mat)
    if ok:
        print(f"La matriz es nilpotente. A^k = 0 para k = {steps}.")
    else:
        print("La matriz no es nilpotente; existen ciclos.")
    # No se guardan informes para esta verificación


def cmd_rotation(args: argparse.Namespace) -> None:
    mat = load_matrix(args.input)
    if corr.check_rotation_invariants(mat):
        print("La matriz conserva normas y orientación (rotación válida).")
    else:
        print("La matriz no es una rotación válida (no conserva medidas o orientación).")


def cmd_assignment(args: argparse.Namespace) -> None:
    cost = load_matrix(args.input)
    # Cargar asignación propuesta desde CSV (dos columnas: fila,columna)
    pairs: List[Tuple[int, int]] = []
    with open(args.assignment, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            pairs.append((int(row[0]), int(row[1])))
    ok = corr.verify_primal_dual(cost, pairs)
    if ok:
        print("La asignación cumple la condición primal–dual y es óptima.")
    else:
        print("La asignación no es óptima o no satisface la condición primal–dual.")


def cmd_inverse(args: argparse.Namespace) -> None:
    a = args.a
    m = args.m
    inv = args.inv
    if corr.verify_inverse_mod(a, m, inv):
        print(f"{inv} es inverso de {a} módulo {m}.")
    else:
        print(f"{inv} NO es inverso de {a} módulo {m}.")


def cmd_crc(args: argparse.Namespace) -> None:
    data_bytes = args.data.encode("utf-8")
    crc_val = int(args.crc, 0)
    ok = corr.verify_crc_error(data_bytes, crc_val)
    if ok:
        print("El CRC coincide con los datos (no se detectan errores).")
    else:
        print("El CRC no coincide con los datos; se detecta un error.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pruebas de corrección y seguridad")
    subparsers = parser.add_subparsers(dest="command", required=True)
    # Nilpotente
    p_nilp = subparsers.add_parser("nilpotent", help="Comprueba si una matriz es nilpotente")
    p_nilp.add_argument("--input", required=True, help="CSV o JSON con la matriz de adyacencia")
    p_nilp.set_defaults(func=cmd_nilpotent)
    # Rotación
    p_rot = subparsers.add_parser("rotation", help="Verifica que una matriz 3x3 es una rotación")
    p_rot.add_argument("--input", required=True, help="CSV o JSON con la matriz 3x3")
    p_rot.set_defaults(func=cmd_rotation)
    # Asignación
    p_asg = subparsers.add_parser("assignment", help="Verifica la optimalidad de una asignación")
    p_asg.add_argument("--input", required=True, help="CSV o JSON con la matriz de costes")
    p_asg.add_argument("--assignment", required=True, help="CSV con pares fila,columna de la asignación propuesta")
    p_asg.set_defaults(func=cmd_assignment)
    # Inverso modular
    p_inv = subparsers.add_parser("inverse", help="Comprueba un inverso modular")
    p_inv.add_argument("--a", type=int, required=True, help="Valor a")
    p_inv.add_argument("--m", type=int, required=True, help="Módulo m")
    p_inv.add_argument("--inv", type=int, required=True, help="Valor propuesto como inverso")
    p_inv.set_defaults(func=cmd_inverse)
    # CRC
    p_crc = subparsers.add_parser("crc", help="Verifica si un código CRC es correcto")
    p_crc.add_argument("--data", required=True, help="Cadena de datos (se codifica en UTF-8)")
    p_crc.add_argument("--crc", required=True, help="Valor del CRC (puede ser 0x prefijado)")
    p_crc.set_defaults(func=cmd_crc)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    args.func(args)


if __name__ == "__main__":
    main()