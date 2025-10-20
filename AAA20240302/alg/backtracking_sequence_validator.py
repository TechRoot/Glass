"""
Generador de secuencias por retroceso con poda por invariante.

Dado un conjunto finito de eventos y un predicado (invariante) que debe cumplirse
para cualquier prefijo de la secuencia, esta función genera todas las
secuencias de una longitud dada que satisfacen el invariante.  Se utiliza
para explorar secuencias de eventos en máquinas de estados finitos
mientras se podan ramas que violan condiciones de seguridad【943327264349420†L1-L75】.
"""

from __future__ import annotations

from typing import Callable, Iterable, List


def generate_sequences(events: Iterable[str], length: int,
                       invariant: Callable[[List[str]], bool]) -> List[List[str]]:
    """Genera todas las secuencias de eventos de la longitud indicada que cumplen un invariante."""
    events = list(events)
    result: List[List[str]] = []
    def backtrack(prefix: List[str]) -> None:
        if len(prefix) == length:
            result.append(prefix.copy())
            return
        for ev in events:
            prefix.append(ev)
            if invariant(prefix):
                backtrack(prefix)
            prefix.pop()
    backtrack([])
    return result
