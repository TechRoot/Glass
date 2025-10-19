#!/usr/bin/env python3
"""CLI para interpolación y ajustes polinomiales.

Este script proporciona comandos de línea para realizar interpolación de
puntos mediante distintos métodos (Lagrange, Newton, baricéntrico) y
para ajustar datos a un polinomio de grado dado usando diversos
esquemas de mínimos cuadrados (ecuaciones normales, SVD, QR y ridge).

El programa soporta un modo `--dry-run` que muestra los resultados por
pantalla sin escribir archivos, y un modo `--confirm` que guarda
informes en el directorio `data/traces/`.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np

from 20210412.lib import interpolation as interp
from 20210412.lib import least_squares as ls


def read_xy_csv(path: str) -> Tuple[List[float], List[float]]:
    """Lee un CSV de dos columnas `x,y` y devuelve listas separadas."""
    xs: List[float] = []
    ys: List[float] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Acepta encabezados arbitrarios, toma las dos primeras columnas
        headers = reader.fieldnames
        if not headers or len(headers) < 2:
            raise ValueError("El CSV debe tener al menos dos columnas")
        for row in reader:
            x = float(row[headers[0]])
            y = float(row[headers[1]])
            xs.append(x)
            ys.append(y)
    return xs, ys


def design_matrix(xs: List[float], degree: int) -> List[List[float]]:
    """Construye una matriz de diseño para un polinomio de grado dado.

    Cada fila corresponde a [x^0, x^1, ..., x^degree]."""
    return [[(x ** k) for k in range(degree + 1)] for x in xs]


def cmd_interpolate(args: argparse.Namespace) -> None:
    xs, ys = read_xy_csv(args.input)
    logging.info("Se han leído %d puntos de interpolación", len(xs))
    if args.method == "lagrange":
        yval = interp.lagrange_interpolate(xs, ys, args.point)
        coeffs = interp.vandermonde_coeffs(xs, ys)
        print(f"Valor interpolado (Lagrange) en x={args.point}: {yval:.6f}")
        print("Coeficientes del polinomio (x^0, x^1, ...):")
        print(", ".join(f"{c:.6f}" for c in coeffs))
    elif args.method == "newton":
        coeffs = interp.newton_divided_differences(xs, ys)
        yval = interp.newton_eval(xs, coeffs, args.point)
        print(f"Valor interpolado (Newton) en x={args.point}: {yval:.6f}")
        print("Coeficientes de Newton (c0, c1, ...):")
        print(", ".join(f"{c:.6f}" for c in coeffs))
    elif args.method == "barycentric":
        weights = interp.barycentric_weights(xs)
        yval = interp.barycentric_eval(xs, ys, weights, args.point)
        print(f"Valor interpolado (baricéntrico) en x={args.point}: {yval:.6f}")
    else:
        raise ValueError(f"Método de interpolación desconocido: {args.method}")
    # Si se solicita confirmación, guardar resultados
    if args.confirm:
        out_dir = Path("data/traces")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "block6_interpolation.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y"])
            writer.writerow([args.point, yval])
        print(f"Resultado guardado en {out_path}")


def cmd_leastsq(args: argparse.Namespace) -> None:
    xs, ys = read_xy_csv(args.input)
    degree = args.degree
    if degree < 0:
        raise ValueError("El grado del polinomio debe ser no negativo")
    # Construir la matriz de diseño A y el vector b
    A = design_matrix(xs, degree)
    method = args.method
    if method == "normal":
        xvec, cond = ls.normal_equations(A, ys)
    elif method == "svd":
        xvec, cond = ls.svd_least_squares(A, ys)
    elif method == "qr":
        xvec, cond = ls.qr_least_squares(A, ys)
    elif method == "ridge":
        alpha = args.alpha if args.alpha is not None else 0.1
        xvec, cond = ls.ridge_regression(A, ys, alpha)
    else:
        raise ValueError(f"Método de ajuste desconocido: {method}")
    mse = ls.mean_squared_error(A, xvec, ys)
    print("Coeficientes del ajuste (x^0, x^1, ...):")
    print(", ".join(f"{c:.6f}" for c in xvec))
    print(f"Número de condición: {cond:.6f}")
    print(f"Error cuadrático medio: {mse:.6f}")
    # Guardar si se confirma
    if args.confirm:
        out_dir = Path("data/traces")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "block6_leastsquares.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["coef_index", "coef_value"])
            for idx, val in enumerate(xvec):
                writer.writerow([idx, val])
        print(f"Coeficientes guardados en {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interpolación y ajuste polinomial")
    parser.add_argument("--log-level", default="INFO", help="Nivel de log (DEBUG, INFO, WARNING, ERROR)")
    subparsers = parser.add_subparsers(dest="command", required=True)
    # Subcomando de interpolación
    parser_int = subparsers.add_parser("interpolate", help="Interpola puntos y evalúa el polinomio")
    parser_int.add_argument("--input", required=True, help="Ruta al CSV con columnas x,y")
    parser_int.add_argument("--method", choices=["lagrange", "newton", "barycentric"], default="lagrange")
    parser_int.add_argument("--point", type=float, required=True, help="Punto en el que evaluar el polinomio")
    parser_int.add_argument("--confirm", action="store_true", help="Guarda el resultado en data/traces")
    parser_int.set_defaults(func=cmd_interpolate)
    # Subcomando de mínimos cuadrados
    parser_ls = subparsers.add_parser("least-squares", help="Ajusta un polinomio por mínimos cuadrados")
    parser_ls.add_argument("--input", required=True, help="Ruta al CSV con columnas x,y")
    parser_ls.add_argument("--degree", type=int, default=1, help="Grado del polinomio a ajustar")
    parser_ls.add_argument(
        "--method",
        choices=["normal", "svd", "qr", "ridge"],
        default="normal",
        help="Método de resolución"
    )
    parser_ls.add_argument(
        "--alpha",
        type=float,
        help="Parámetro de regularización para ridge (λ)",
    )
    parser_ls.add_argument("--confirm", action="store_true", help="Guarda los coeficientes en data/traces")
    parser_ls.set_defaults(func=cmd_leastsq)
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    args.func(args)


if __name__ == "__main__":
    main()