#!/usr/bin/env python3
"""
CLI para las utilidades de lógica y máquinas de estados.

Proporciona comandos para minimizar funciones, obtener formas normales, calcular
derivadas booleanas, añadir términos de consenso, evaluar márgenes temporales,
verificar FSM y comparar implementaciones con un miter. Todas las operaciones
son de sólo lectura salvo que se indique `--confirm`.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

from ..lib import boolean_logic as bl
from ..lib import fsm_utils as fu


def configure_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=numeric, format="%(levelname)s: %(message)s")


def cmd_minimize(args) -> None:
    minterms = [int(x) for x in args.minterms.split(',')] if args.minterms else []
    dont_cares = [int(x) for x in args.dont_cares.split(',')] if args.dont_cares else []
    patterns = bl.minimize_function(minterms, args.num_vars, dont_cares, method=args.method)
    vars_list = args.variables.split(',') if args.variables else [f'x{i}' for i in range(args.num_vars)]
    expressions = [bl.pattern_to_expression(p, vars_list) for p in patterns]
    print("Implicantes primos:", patterns)
    print("Forma simplificada:", ' | '.join(f"({e})" for e in expressions))


def cmd_sop(args) -> None:
    minterms = [int(x) for x in args.minterms.split(',')] if args.minterms else []
    vars_list = args.variables.split(',') if args.variables else [f'x{i}' for i in range(args.num_vars)]
    expr = bl.sum_of_products(minterms, args.num_vars, vars_list)
    print("Suma de productos:", expr)


def cmd_pos(args) -> None:
    zeros = [int(x) for x in args.zeros.split(',')] if args.zeros else []
    vars_list = args.variables.split(',') if args.variables else [f'x{i}' for i in range(args.num_vars)]
    expr = bl.product_of_sums(zeros, args.num_vars, vars_list)
    print("Producto de sumas:", expr)


def cmd_consensus(args) -> None:
    term1 = tuple(args.term1.split(','))
    term2 = tuple(args.term2.split(','))
    res = bl.add_consensus(term1, term2)
    print("Término de consenso:", res)


def cmd_derivative(args) -> None:
    # definimos la función original como tabla de minterminos
    minterms = [int(x) for x in args.minterms.split(',')]
    num_vars = args.num_vars
    def func(*bits: int) -> int:
        index = sum(bit << (num_vars - i - 1) for i, bit in enumerate(bits))
        return 1 if index in minterms else 0
    derivative = bl.boolean_derivative(func, args.var_index, num_vars)
    # mostrarmos la tabla de verdad de la derivada
    print("Derivada booleana:")
    for bits in itertools.product([0, 1], repeat=num_vars):
        print(bits, '→', derivative(*bits))


def cmd_timing(args) -> None:
    margin = bl.timing_margin(args.launch, args.delay, args.setup)
    print(f"Margen temporal = {margin:.6f} (positivo indica margen disponible)")


def cmd_fsm_check(args) -> None:
    with open(args.file) as f:
        adj = json.load(f)
    reachable, nodes = fu.bfs_reachability(adj, args.start, args.target)
    print(f"Alcanzable: {reachable}, nodos visitados: {nodes}")
    cycle = fu.has_cycle(adj, args.start)
    print(f"Contiene ciclos accesibles: {cycle}")


def cmd_miter(args) -> None:
    # definimos las funciones a partir de minterminos
    mt1 = [int(x) for x in args.minterms1.split(',')]
    mt2 = [int(x) for x in args.minterms2.split(',')]
    num_vars = args.num_vars
    def f1(*bits: int) -> int:
        index = sum(bit << (num_vars - i - 1) for i, bit in enumerate(bits))
        return 1 if index in mt1 else 0
    def f2(*bits: int) -> int:
        index = sum(bit << (num_vars - i - 1) for i, bit in enumerate(bits))
        return 1 if index in mt2 else 0
    equivalent = fu.check_equivalence(f1, f2, num_vars)
    print("Equivalencia funcional:", equivalent)


def cmd_parity(args) -> None:
    bits = [int(b) for b in args.bits.split(',')]
    p = fu.parity_bit(bits)
    print("Bit de paridad:", p)


def cmd_doublepulse(args) -> None:
    out = fu.double_pulse(args.valid1, args.valid2, args.state)
    print("Salida de doble pulsación:", out)


def main() -> None:
    parser = argparse.ArgumentParser(description="CLI para lógica booleana y FSM")
    parser.add_argument('--dry-run', action='store_true', default=True, help='Muestra lo que se haría sin efectos')
    parser.add_argument('--confirm', action='store_true', help='Confirma operaciones que modifican ficheros')
    parser.add_argument('--log-level', default='INFO', help='Nivel de logging (INFO, DEBUG, ... )')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # minimize
    p_min = subparsers.add_parser('minimize', help='Minimiza una función booleana')
    p_min.add_argument('--minterms', required=True, help='Lista de minterminos separados por coma')
    p_min.add_argument('--dont-cares', default='', help='Lista de estados no influyentes')
    p_min.add_argument('--num-vars', type=int, required=True, help='Número de variables')
    p_min.add_argument('--variables', default='', help='Nombres de variables separados por coma')
    p_min.add_argument('--method', default='quine_mccluskey', help='Método de minimización')
    p_min.set_defaults(func=cmd_minimize)

    # sum of products
    p_sop = subparsers.add_parser('sop', help='Calcula la forma suma de productos')
    p_sop.add_argument('--minterms', required=True, help='Lista de minterminos separados por coma')
    p_sop.add_argument('--num-vars', type=int, required=True, help='Número de variables')
    p_sop.add_argument('--variables', default='', help='Nombres de variables')
    p_sop.set_defaults(func=cmd_sop)

    # product of sums
    p_pos = subparsers.add_parser('pos', help='Calcula la forma producto de sumas')
    p_pos.add_argument('--zeros', required=True, help='Lista de estados donde la función vale 0')
    p_pos.add_argument('--num-vars', type=int, required=True, help='Número de variables')
    p_pos.add_argument('--variables', default='', help='Nombres de variables')
    p_pos.set_defaults(func=cmd_pos)

    # consensus
    p_con = subparsers.add_parser('consensus', help='Calcula término de consenso entre dos productos')
    p_con.add_argument('--term1', required=True, help='Primer producto como literales separados por coma (A,~B,...)')
    p_con.add_argument('--term2', required=True, help='Segundo producto como literales separados por coma')
    p_con.set_defaults(func=cmd_consensus)

    # derivative
    p_der = subparsers.add_parser('derivative', help='Calcula la derivada booleana de una función')
    p_der.add_argument('--minterms', required=True, help='Minterminos de la función original')
    p_der.add_argument('--num-vars', type=int, required=True, help='Número de variables')
    p_der.add_argument('--var-index', type=int, required=True, help='Índice de la variable sobre la que derivar (0..n-1)')
    p_der.set_defaults(func=cmd_derivative)

    # timing
    p_time = subparsers.add_parser('timing', help='Calcula el margen temporal')
    p_time.add_argument('--launch', type=float, required=True, help='Tiempo de lanzamiento')
    p_time.add_argument('--delay', type=float, required=True, help='Retardo de ruta')
    p_time.add_argument('--setup', type=float, required=True, help='Tiempo de preparación')
    p_time.set_defaults(func=cmd_timing)

    # fsm check
    p_fsm = subparsers.add_parser('fsm-check', help='Verifica alcanzabilidad y ciclos en una FSM')
    p_fsm.add_argument('--file', required=True, help='Ruta a un JSON con la matriz de adyacencia (estado → lista de estados)')
    p_fsm.add_argument('--start', required=True, help='Estado inicial')
    p_fsm.add_argument('--target', required=True, help='Estado objetivo')
    p_fsm.set_defaults(func=cmd_fsm_check)

    # miter equivalence
    p_mit = subparsers.add_parser('miter', help='Comprueba equivalencia entre dos funciones por minterminos')
    p_mit.add_argument('--minterms1', required=True, help='Minterminos de la primera función')
    p_mit.add_argument('--minterms2', required=True, help='Minterminos de la segunda función')
    p_mit.add_argument('--num-vars', type=int, required=True, help='Número de variables')
    p_mit.set_defaults(func=cmd_miter)

    # parity
    p_par = subparsers.add_parser('parity', help='Calcula bit de paridad')
    p_par.add_argument('--bits', required=True, help='Bits separados por coma')
    p_par.set_defaults(func=cmd_parity)

    # double pulse
    p_dp = subparsers.add_parser('doublepulse', help='Evalúa doble pulsación segura')
    p_dp.add_argument('--valid1', type=int, choices=[0, 1], required=True, help='Estado del primer canal (0/1)')
    p_dp.add_argument('--valid2', type=int, choices=[0, 1], required=True, help='Estado del segundo canal (0/1)')
    p_dp.add_argument('--state', type=int, choices=[0, 1], required=True, help='Estado previo de armado (0/1)')
    p_dp.set_defaults(func=cmd_doublepulse)

    args = parser.parse_args()
    configure_logging(args.log_level)
    # Actualmente no hay operaciones de escritura; `--confirm` se reserva para futuros comandos
    args.func(args)


if __name__ == '__main__':
    main()