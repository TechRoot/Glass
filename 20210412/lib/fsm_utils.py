"""
Utilidades para trabajar con máquinas de estados finitos (FSM).

Este módulo proporciona funciones para representar una FSM mediante una
matriz o diccionario de adyacencia, explorar su espacio de estados con
búsqueda en anchura (BFS), verificar que un estado objetivo es alcanzable
y que no existen ciclos en las trayectorias de interés, medir la
complejidad temporal de la exploración y comparar lógicas mediante un
circuito miter.
"""

from __future__ import annotations

import itertools
from collections import deque
from typing import Dict, Iterable, List, Tuple, Callable


def bfs_reachability(adj: Dict[str, List[str]], start: str, target: str) -> Tuple[bool, int]:
    """Realiza una búsqueda en anchura sobre un grafo dirigido para determinar
    si `target` es alcanzable desde `start`.

    Args:
        adj: diccionario que mapea cada estado a la lista de estados sucesores.
        start: estado inicial.
        target: estado objetivo.

    Returns:
        Una tupla (alcanzable, nodos_visitados). La bandera es verdadera si se
        alcanza el objetivo. `nodos_visitados` indica cuántos estados han
        sido extraídos de la cola (aproximación de la complejidad temporal).
    """
    visited = set()
    queue = deque([start])
    visited.add(start)
    nodes_visited = 0
    while queue:
        current = queue.popleft()
        nodes_visited += 1
        if current == target:
            return True, nodes_visited
        for succ in adj.get(current, []):
            if succ not in visited:
                visited.add(succ)
                queue.append(succ)
    return False, nodes_visited


def has_cycle(adj: Dict[str, List[str]], start: str) -> bool:
    """Detecta si existe un ciclo accesible desde `start`.

    Usa un algoritmo DFS con pila de recorrido.
    """
    visited = set()
    stack = set()
    def dfs(node: str) -> bool:
        if node in stack:
            return True
        if node in visited:
            return False
        visited.add(node)
        stack.add(node)
        for succ in adj.get(node, []):
            if dfs(succ):
                return True
        stack.remove(node)
        return False
    return dfs(start)


def build_miter(func1: Callable[..., int], func2: Callable[..., int], num_vars: int) -> Callable[..., int]:
    """Construye una función miter que devuelve 1 si las implementaciones difieren.

    El miter calcula la disyunción exclusiva (XOR) de las salidas de ambas
    funciones para un vector de entrada.
    """
    def miter_func(*args: int) -> int:
        return func1(*args) ^ func2(*args)
    return miter_func


def check_equivalence(func1: Callable[..., int], func2: Callable[..., int], num_vars: int) -> bool:
    """Comprueba la equivalencia funcional de dos funciones booleanas exhaustivamente.

    Evalúa ambos sobre todas las combinaciones posibles de entradas y retorna
    True solo si todas las salidas coinciden.
    """
    for bits in itertools.product([0, 1], repeat=num_vars):
        if func1(*bits) != func2(*bits):
            return False
    return True


def estimated_bfs_complexity(num_states: int, branching_factor: int) -> str:
    """Devuelve una cadena con la complejidad temporal O(b^d) de un BFS.

    Este cálculo es teórico: b es el factor de ramificación y d es la
    profundidad máxima. Se usa únicamente con fines informativos.
    """
    return f"O({branching_factor}^d)"


def parity_bit(data_bits: Iterable[int]) -> int:
    """Reexporta el cálculo de paridad del módulo boolean_logic.

    Esta función se incluye aquí por comodidad de importación cruzada.
    """
    from .boolean_logic import parity_bit as _pb
    return _pb(data_bits)


def double_pulse(valid1: bool, valid2: bool, state: bool) -> bool:
    """Reexporta la doble pulsación del módulo boolean_logic.

    Permite usar la lógica de doble pulsación directamente desde este módulo.
    """
    from .boolean_logic import double_pulse as _dp
    return _dp(valid1, valid2, state)
