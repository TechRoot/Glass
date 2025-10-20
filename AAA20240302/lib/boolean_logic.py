"""
Utilidades de álgebra booleana: minimización, derivadas y márgenes temporales.

Este módulo contiene implementaciones didácticas de varias operaciones de
lógica booleana: minimización mediante Quine–McCluskey, conversión entre
formas canónicas suma‑de‑productos (SOP) y producto‑de‑sumas (POS), cálculo
de términos de consenso, derivadas booleanas, márgenes temporales para
análisis de hazards y comprobaciones simples de paridad y doble impulso.
Estas funciones pueden utilizarse como bloques de construcción en scripts
de lógica.

Dependencias: solo la biblioteca estándar de Python

Autor: A. Alonso
Commit: 849862447569455
Fecha: 2024-06-12
"""

from __future__ import annotations

from typing import Iterable, List, Tuple


def _int_to_bin_tuple(value: int, n: int) -> Tuple[int, ...]:
    return tuple((value >> i) & 1 for i in reversed(range(n)))


def _combine_terms(a: Tuple[int, ...], b: Tuple[int, ...]) -> Tuple[int, ...]:
    diff = 0
    combo = []
    for x, y in zip(a, b):
        if x != y:
            diff += 1
            combo.append(-1)  # -1 representa un don't care
        else:
            combo.append(x)
    if diff == 1:
        return tuple(combo)
    return None  # type: ignore


def quine_mccluskey(minterms: Iterable[int], n_vars: int) -> List[Tuple[int, ...]]:
    """Minimiza una función booleana usando el algoritmo de Quine–McCluskey."""
    groups = {}
    for m in minterms:
        bits = _int_to_bin_tuple(m, n_vars)
        ones = sum(bits)
        groups.setdefault(ones, []).append(bits)
    prime = set()
    while groups:
        next_groups = {}
        used = set()
        keys = sorted(groups.keys())
        for i in range(len(keys) - 1):
            for a in groups[keys[i]]:
                for b in groups[keys[i + 1]]:
                    c = _combine_terms(a, b)
                    if c is not None:
                        next_groups.setdefault(keys[i], []).append(c)
                        used.add(a)
                        used.add(b)
        for group in groups.values():
            for term in group:
                if term not in used:
                    prime.add(term)
        groups = {}
        for key, terms in next_groups.items():
            groups[key] = list(set(terms))
    return list(prime)


def term_to_expr(term: Tuple[int, ...], var_names: List[str]) -> str:
    """Convierte un término binario con comodines (−1) en una expresión literal."""
    expr_parts = []
    for bit, var in zip(term, var_names):
        if bit == 1:
            expr_parts.append(var)
        elif bit == 0:
            expr_parts.append(f"!{var}")
    return ' & '.join(expr_parts) if expr_parts else '1'


def minimise_sop(minterms: Iterable[int], var_names: List[str]) -> List[str]:
    """Devuelve las expresiones mínimas en forma de suma‑de‑productos que cubren los minterminos."""
    prime_terms = quine_mccluskey(minterms, len(var_names))
    return [term_to_expr(term, var_names) for term in prime_terms]


def boolean_derivative(expr_minterms: Iterable[int], var_index: int, n_vars: int) -> List[int]:
    """Calcula la derivada booleana ∂f/∂x_i como la diferencia simétrica de minterminos."""
    mask = 1 << (n_vars - var_index - 1)
    result = []
    for m in expr_minterms:
        if (m ^ mask) in expr_minterms:
            result.append(m & ~mask)
    return sorted(set(result))


def consensus(a: Iterable[int], b: Iterable[int]) -> List[int]:
    """Calcula los términos de consenso entre dos términos producto (índices de minterminos)."""
    return sorted(set(a) & set(b))


def timing_margin(delays: List[float]) -> float:
    """Calcula un margen temporal de caso peor sencillo.

    Para una lista de retardos de puertas, el margen es la diferencia entre
    el máximo y el segundo máximo retardo【849862447569455†L1-L233】.
    """
    if len(delays) < 2:
        return 0.0
    d_sorted = sorted(delays)
    return float(d_sorted[-1] - d_sorted[-2])


def parity_bit(data: Iterable[int]) -> int:
    """Devuelve el bit de paridad (par)."""
    return sum(data) % 2


def double_pulse(signal: Iterable[int]) -> bool:
    """Comprueba la presencia de un doble impulso en una señal binaria."""
    prev = 0
    for bit in signal:
        if bit and prev:
            return True
        prev = bit
    return False
