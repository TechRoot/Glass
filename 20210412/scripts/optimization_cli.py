#!/usr/bin/env python3
"""CLI para tareas de optimización combinatoria.

Este script agrupa subcomandos para resolver problemas de asignación de
costes (método Húngaro), planificación de trabajos en dos máquinas (regla
de Johnson) y aplicación de una heurística simple de corte bidimensional.

Características comunes:
* `--dry-run` está activado por defecto para evitar escribir archivos.
* `--confirm` permite confirmar la operación y escribir resultados.
* `--input` indica la ruta del archivo CSV de entrada.
* `--output` permite especificar el archivo de salida (cuando aplique).
* `--log-level` controla la verbosidad de los mensajes.

Uso de cada subcomando:
  assign   → resuelve un problema de asignación a partir de una matriz de costes.
  schedule → aplica la regla de Johnson a un conjunto de tareas.
  cut      → invoca el script de heurística de corte bidimensional.
"""

from __future__ import annotations

import argparse
import logging
import csv
import subprocess
from pathlib import Path
from typing import List, Tuple

from ..lib import assignment
from ..lib import scheduling


def configure_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=numeric, format="%(levelname)s: %(message)s")


def cmd_assign(args) -> None:
    # Cargar matriz desde CSV y resolver asignación
    matrix: List[List[float]] = []
    with open(args.input, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            matrix.append([float(v) for v in row])
    assignment_result, total = assignment.hungarian(matrix)
    logging.info("Asignación óptima (fila->columna): %s", assignment_result)
    logging.info("Coste total: %.4f", total)
    # Si confirm, escribir a trazas
    if args.confirm:
        Path('data/traces').mkdir(parents=True, exist_ok=True)
        trace_path = args.output or 'data/traces/block5_assignment.csv'
        with open(trace_path, 'w', newline='') as f:
            writer = csv.writer(f)
            for r, c in assignment_result:
                writer.writerow([r, c, matrix[r][c]])
        logging.info("Trazas guardadas en %s", trace_path)
    else:
        logging.info("--dry-run: no se escribe fichero de resultados")


def cmd_schedule(args) -> None:
    # Cargar trabajos desde CSV (id,t1,t2) o (t1,t2)
    jobs: List[Tuple] = []
    with open(args.input, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if len(row) == 2:
                jobs.append((float(row[0]), float(row[1])))
            else:
                jobs.append((row[0], float(row[1]), float(row[2])))
    order = scheduling.johnson_schedule(jobs)
    makespan = scheduling.calculate_makespan(order, jobs)
    logging.info("Orden óptimo: %s", order)
    logging.info("Makespan total: %.4f", makespan)
    if args.confirm:
        Path('data/traces').mkdir(parents=True, exist_ok=True)
        trace_path = args.output or 'data/traces/block5_schedule.csv'
        with open(trace_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['job_index'])
            for idx in order:
                writer.writerow([idx])
        logging.info("Orden guardado en %s", trace_path)
    else:
        logging.info("--dry-run: no se escribe fichero de resultados")


def cmd_cut(args) -> None:
    # Construir comando para invocar el script de corte
    script_path = Path(__file__).resolve().parent.parent / 'lib' / 'cutting_heuristic.sh'
    cmd = [str(script_path), '--input', args.input]
    if args.board_size:
        cmd += ['--board-size', args.board_size]
    if args.output:
        cmd += ['--output', args.output]
    # Activar dry-run o confirm
    if args.confirm:
        cmd.append('--confirm')
    else:
        cmd.append('--dry-run')
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logging.error("Error ejecutando cutting_heuristic.sh: %s", e)


def main() -> None:
    parser = argparse.ArgumentParser(description="Herramientas de optimización combinatoria")
    parser.add_argument('--log-level', default='INFO', help='Nivel de logging')
    subparsers = parser.add_subparsers(dest='command', required=True)
    # Asignación
    p_assign = subparsers.add_parser('assign', help='Resuelve un problema de asignación (Hungarian)')
    p_assign.add_argument('--input', required=True, help='Archivo CSV con la matriz de costes')
    p_assign.add_argument('--output', help='Fichero donde guardar la asignación (opcional)')
    p_assign.add_argument('--dry-run', action='store_true', default=True, help='No escribe resultado (por defecto)')
    p_assign.add_argument('--confirm', action='store_true', help='Escribe resultado en un fichero')
    p_assign.set_defaults(func=cmd_assign)
    # Planificación
    p_sched = subparsers.add_parser('schedule', help='Aplica la regla de Johnson a trabajos en dos máquinas')
    p_sched.add_argument('--input', required=True, help='Archivo CSV con trabajos (id,t1,t2) o (t1,t2)')
    p_sched.add_argument('--output', help='Fichero donde guardar el orden')
    p_sched.add_argument('--dry-run', action='store_true', default=True, help='No escribe resultado (por defecto)')
    p_sched.add_argument('--confirm', action='store_true', help='Escribe resultado en un fichero')
    p_sched.set_defaults(func=cmd_schedule)
    # Corte
    p_cut = subparsers.add_parser('cut', help='Aplica heurística de corte bidimensional')
    p_cut.add_argument('--input', required=True, help='Archivo CSV con pedidos (ancho,alto)')
    p_cut.add_argument('--board-size', help='Dimensiones de la lámina, por defecto 100,100')
    p_cut.add_argument('--output', help='Archivo donde guardar el resultado de la heurística')
    p_cut.add_argument('--dry-run', action='store_true', default=True, help='No escribe resultado (por defecto)')
    p_cut.add_argument('--confirm', action='store_true', help='Escribe resultado en un fichero')
    p_cut.set_defaults(func=cmd_cut)
    args = parser.parse_args()
    configure_logging(args.log_level)
    args.func(args)


if __name__ == '__main__':
    main()