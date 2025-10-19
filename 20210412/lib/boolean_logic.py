"""
Módulo de utilidades para lógica booleana y minimización.

Incluye una implementación básica del algoritmo de Quine‑McCluskey para simplificar
funciones a partir de minterminos, conversión a formas normal conjuntas y
productos de sumas, cálculo de derivadas booleanas, adición de términos de
consenso para eliminar *hazards* estáticos y evaluación de márgenes temporales.

Las funciones están diseñadas para su uso en un contexto didáctico. No se
pretende competir con bibliotecas especializadas y su rendimiento se orienta a
casos de pocos variables.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Set, Tuple, Dict, Optional, Callable


def _group_terms(terms: Iterable[int], num_vars: int) -> Dict[int, List[str]]:
    """Agrupa minterminos por el número de bits a uno.

    Devuelve un diccionario clave → lista de términos en representación binaria.
    """
    groups: Dict[int, List[str]] = {}
    for term in terms:
        bits = bin(term)[2:].zfill(num_vars)
        count = bits.count('1')
        groups.setdefault(count, []).append(bits)
    return groups


def _combine_pairs(a: str, b: str) -> Optional[str]:
    """Combina dos términos binarios si difieren en un solo bit.

    Devuelve la cadena combinada con un guion ('-') en la posición que difiere,
    o None si no se pueden combinar.
    """
    diff = 0
    combined = []
    for ca, cb in zip(a, b):
        if ca != cb:
            diff += 1
            combined.append('-')
        else:
            combined.append(ca)
        if diff > 1:
            return None
    return ''.join(combined) if diff == 1 else None


def _quine_mccluskey(minterms: Iterable[int], dont_cares: Iterable[int], num_vars: int) -> Set[str]:
    """Implementa el algoritmo de Quine‑McCluskey para obtener implicantes primos.

    Devuelve un conjunto de términos simplificados con guiones representando
    variables indiferentes.
    """
    terms = set(minterms) | set(dont_cares)
    groups = _group_terms(terms, num_vars)
    prime_implicants: Set[str] = set()
    while groups:
        next_groups: Dict[int, List[str]] = {}
        used = set()
        all_terms = []
        for count in sorted(groups.keys()):
            all_terms.extend(groups[count])
            if count + 1 not in groups:
                continue
            for a in groups[count]:
                for b in groups[count + 1]:
                    combined = _combine_pairs(a, b)
                    if combined:
                        used.add(a)
                        used.add(b)
                        next_groups.setdefault(combined.count('1'), []).append(combined)
        # añadir los términos que no se pudieron combinar
        for group_terms in groups.values():
            for t in group_terms:
                if t not in used:
                    prime_implicants.add(t)
        # normalizar y eliminar duplicados en next_groups
        dedup: Dict[str, None] = {}
        for terms_list in next_groups.values():
            for t in terms_list:
                dedup[t] = None
        groups = _group_terms([k for k in dedup.keys()], num_vars) if dedup else {}
    # eliminar implicantes que cubren únicamente don't cares
    filtered = set()
    for implicant in prime_implicants:
        covers_only_dc = True
        for m in minterms:
            if _covers(implicant, m):
                covers_only_dc = False
                break
        if not covers_only_dc:
            filtered.add(implicant)
    return filtered


def _covers(pattern: str, value: int) -> bool:
    """Indica si un patrón (con guiones) cubre un mintermino concreto."""
    bits = bin(value)[2:].zfill(len(pattern))
    for p, b in zip(pattern, bits):
        if p == '-':
            continue
        if p != b:
            return False
    return True


def minimize_function(
    minterms: Iterable[int],
    num_vars: int,
    dont_cares: Iterable[int] = (),
    method: str = "quine_mccluskey",
) -> List[str]:
    """Simplifica una función booleana dada por sus minterminos.

    Args:
        minterms: Iterable de índices donde la función vale 1.
        num_vars: número de variables.
        dont_cares: índices que pueden ser ignorados.
        method: actualmente solo se implementa `quine_mccluskey`.

    Returns:
        Lista de patrones simplificados (p.ej. '1-0-') que representan cada
        implicante primo esencial.
    """
    if method.lower() != "quine_mccluskey":
        raise ValueError("Método de minimización no soportado")
    return sorted(_quine_mccluskey(minterms, dont_cares, num_vars))


def pattern_to_expression(pattern: str, variables: List[str]) -> str:
    """Convierte un patrón de bits/guiones en una expresión booleana en suma de productos.

    Por ejemplo, '1-0' con variables ['A','B','C'] produce 'A & ~C'.
    """
    terms = []
    for bit, var in zip(pattern, variables):
        if bit == '1':
            terms.append(var)
        elif bit == '0':
            terms.append(f"~{var}")
    return ' & '.join(terms) if terms else '1'


def sum_of_products(minterms: Iterable[int], num_vars: int, variables: List[str]) -> str:
    """Devuelve la forma suma de productos (SOP) a partir de minterminos."""
    exprs = []
    for m in minterms:
        bits = bin(m)[2:].zfill(num_vars)
        terms = []
        for bit, var in zip(bits, variables):
            terms.append(var if bit == '1' else f"~{var}")
        exprs.append(' & '.join(terms))
    return ' | '.join(exprs) if exprs else '0'


def product_of_sums(zeros: Iterable[int], num_vars: int, variables: List[str]) -> str:
    """Devuelve la forma producto de sumas (POS) a partir de ceros (donde la función vale 0)."""
    exprs = []
    for z in zeros:
        bits = bin(z)[2:].zfill(num_vars)
        terms = []
        for bit, var in zip(bits, variables):
            terms.append(var if bit == '0' else f"~{var}")
        exprs.append(' | '.join(terms))
    return ' & '.join(f"({e})" for e in exprs) if exprs else '1'


def add_consensus(term1: Tuple[str, str], term2: Tuple[str, str]) -> str:
    """Añade el término de consenso entre dos productos para eliminar un hazard estático.

    Cada término se representa como una tupla de literales (e.g., ('A', '~B')).
    Para F = XY + X'Z, el término de consenso es YZ.
    """
    literals1 = set(term1)
    literals2 = set(term2)
    # identificar la variable complementaria
    vars1 = {lit.strip('~') for lit in literals1}
    vars2 = {lit.strip('~') for lit in literals2}
    common = vars1 & vars2
    consensus_literals: Set[str] = set()
    for v in common:
        # añadir v si aparece sin complemento en ambos
        pos1 = v in literals1
        pos2 = v in literals2
        neg1 = f"~{v}" in literals1
        neg2 = f"~{v}" in literals2
        if pos1 and pos2:
            consensus_literals.add(v)
        elif neg1 and neg2:
            consensus_literals.add(f"~{v}")
        # si aparece complementado en uno y sin complementar en otro, no se añade
    return ' & '.join(sorted(consensus_literals))


def boolean_derivative(func: Callable[..., int], var_index: int, num_vars: int) -> Callable[..., int]:
    """Devuelve una función que representa la derivada booleana de `func` respecto a la variable en `var_index`.

    La derivada booleana se define como f(x_i=1) XOR f(x_i=0).
    """
    def derivative(*args: int) -> int:
        args1 = list(args)
        args0 = list(args)
        args1[var_index] = 1
        args0[var_index] = 0
        return func(*args1) ^ func(*args0)

    return derivative


def timing_margin(launch: float, path_delay: float, setup: float) -> float:
    """Calcula el margen temporal entre lanzamiento, retardo de ruta y tiempo de preparación.

    Si el margen es negativo, indica una violación del presupuesto de tiempo.
    """
    return launch - path_delay - setup


def parity_bit(data_bits: Iterable[int]) -> int:
    """Calcula el bit de paridad de un conjunto de bits (paridad impar)."""
    return sum(data_bits) % 2


def double_pulse(valid1: bool, valid2: bool, state: bool) -> bool:
    """Evalúa la condición de doble pulsación segura.

    Se requiere que ambas entradas estén activas para que la salida sea verdadera.
    El estado indica si ya se ha armado la orden (evita repeticiones).
    """
    return not state and valid1 and valid2
