"""
Interfaz de línea de comandos para operaciones de lógica booleana y máquinas de estados finitos (FSM).

Este script expone funcionalidad de minimización booleana, conversión SOP/POS,
cálculo de términos de consenso, derivadas booleanas y márgenes de temporización,
así como operaciones de alcanzabilidad y equivalencia en autómatas.

Nota: esta CLI está intencionadamente simplificada; acepta minterminos
como enteros separados por comas y nombres de variables como cadenas separadas
por comas.  Los FSM se proporcionan como archivos JSON que mapean estados a
listas de sucesores; los estados finales pueden indicarse con el sufijo "_F" en
el nombre del estado.

Autor: Alex Alonso
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

from ..lib.boolean_logic import minimise_sop, boolean_derivative, consensus, timing_margin, parity_bit, double_pulse
from ..lib.fsm_utils import bfs_reachable, has_cycle, equivalent


def parse_int_list(arg: str) -> List[int]:
    return [int(x.strip()) for x in arg.split(',') if x.strip()]


def parse_str_list(arg: str) -> List[str]:
    return [x.strip() for x in arg.split(',') if x.strip()]


def load_fsm(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="CLI de lógica: utilidades booleanas y FSM")
    sub = parser.add_subparsers(dest='command', required=True)

    sop = sub.add_parser('sop', help='Minimizar una expresión suma de productos')
    sop.add_argument('minterms', type=str, help='Índices de minterminos separados por comas')
    sop.add_argument('vars', type=str, help='Nombres de variables separados por comas')

    deriv = sub.add_parser('deriv', help='Derivada booleana de una función')
    deriv.add_argument('minterms', type=str, help='Minterminos de la función')
    deriv.add_argument('var', type=int, help='Índice de la variable (comenzando en 0)')
    deriv.add_argument('n_vars', type=int, help='Número total de variables')

    cons = sub.add_parser('consensus', help='Términos de consenso de dos términos')
    cons.add_argument('a', type=str, help='Índices de minterminos del término A')
    cons.add_argument('b', type=str, help='Índices de minterminos del término B')

    tm = sub.add_parser('timing', help='Calcular margen temporal a partir de retardos')
    tm.add_argument('delays', type=str, help='Lista de retardos de puertas separada por comas (números)')

    par = sub.add_parser('parity', help='Calcular el bit de paridad de datos')
    par.add_argument('bits', type=str, help='Bits separados por comas (0 o 1)')

    dp = sub.add_parser('double', help='Comprobar doble impulso en la señal')
    dp.add_argument('bits', type=str, help='Bits separados por comas (0 o 1)')

    reach = sub.add_parser('reach', help='Alcanzabilidad BFS en un FSM')
    reach.add_argument('fsm', type=Path, help='JSON que mapea estados a listas de sucesores')
    reach.add_argument('start', type=str, help='Estado inicial')

    cyc = sub.add_parser('cycle', help='Detectar ciclo en un FSM')
    cyc.add_argument('fsm', type=Path, help='JSON que mapea estados a listas de sucesores')

    equiv = sub.add_parser('equiv', help='Comprobar equivalencia de dos FSM')
    equiv.add_argument('fsm1', type=Path)
    equiv.add_argument('fsm2', type=Path)
    equiv.add_argument('start1', type=str)
    equiv.add_argument('start2', type=str)

    args = parser.parse_args()
    if args.command == 'sop':
        mins = parse_int_list(args.minterms)
        vars_ = parse_str_list(args.vars)
        exprs = minimise_sop(mins, vars_)
        print(exprs)
    elif args.command == 'deriv':
        mins = parse_int_list(args.minterms)
        result = boolean_derivative(mins, args.var, args.n_vars)
        print(result)
    elif args.command == 'consensus':
        a = parse_int_list(args.a)
        b = parse_int_list(args.b)
        print(consensus(a, b))
    elif args.command == 'timing':
        delays = [float(x.strip()) for x in args.delays.split(',')]
        print(timing_margin(delays))
    elif args.command == 'parity':
        bits = [int(x.strip()) for x in args.bits.split(',')]
        print(parity_bit(bits))
    elif args.command == 'double':
        bits = [int(x.strip()) for x in args.bits.split(',')]
        print(double_pulse(bits))
    elif args.command == 'reach':
        fsm = load_fsm(args.fsm)
        reachable = bfs_reachable(fsm, args.start)
        print(sorted(reachable))
    elif args.command == 'cycle':
        fsm = load_fsm(args.fsm)
        print(has_cycle(fsm))
    elif args.command == 'equiv':
        fsm1 = load_fsm(args.fsm1)
        fsm2 = load_fsm(args.fsm2)
        print(equivalent(fsm1, fsm2, args.start1, args.start2))


if __name__ == '__main__':
    main()
