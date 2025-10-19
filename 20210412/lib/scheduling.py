"""Heurísticas de planificación para sistemas de dos máquinas.

Este módulo contiene una implementación de la regla de Johnson para
minimizar el tiempo de terminación (makespan) en problemas de flujo
con dos máquinas. También proporciona funciones auxiliares para
calcular el tiempo total de procesamiento dado un orden específico.

Cada trabajo se define por sus tiempos de procesamiento en la máquina
1 y en la máquina 2. La regla de Johnson ordena los trabajos
seleccionando sucesivamente el trabajo con el menor tiempo entre
ambas máquinas: si el tiempo mínimo está en la primera máquina, el
trabajo se coloca al inicio de la secuencia; si el tiempo mínimo está
en la segunda máquina, el trabajo se coloca al final. La complejidad
es O(n log n) debido a las operaciones de búsqueda repetidas.

Funciones principales:
* `johnson_schedule(jobs)` → devuelve una lista de índices en el
  orden que minimiza el makespan.
* `calculate_makespan(order, jobs)` → calcula el makespan del orden
  dado.

Los trabajos pueden ser representados como tuplas `(id, t1, t2)` o
listas de tiempos `(t1, t2)`; en el segundo caso los índices se
derivan de la posición en la lista.
"""

from __future__ import annotations

from typing import List, Tuple, Sequence


def johnson_schedule(jobs: Sequence[Tuple[float, float]] | Sequence[Tuple[str, float, float]]) -> List[int]:
    """Aplica la regla de Johnson a una lista de trabajos para dos máquinas.

    Args:
        jobs: lista de trabajos; cada elemento puede ser una tupla
          `(t1, t2)` para tiempos en máquinas 1 y 2, o `(id, t1, t2)`.

    Returns:
        Lista de índices de los trabajos en el orden óptimo. Si los
        trabajos incluyen un campo de identificación, se ordena por
        la posición en la lista original.
    """
    # Normalizar a (index, t1, t2)
    norm_jobs: List[Tuple[int, float, float]] = []
    for idx, job in enumerate(jobs):
        if len(job) == 2:
            t1, t2 = job
            norm_jobs.append((idx, float(t1), float(t2)))
        else:
            _, t1, t2 = job  # se ignora id
            norm_jobs.append((idx, float(t1), float(t2)))
    order_start: List[int] = []
    order_end: List[int] = []
    remaining = norm_jobs.copy()
    while remaining:
        # encontrar trabajo con tiempo mínimo entre ambas máquinas
        min_job = min(remaining, key=lambda x: min(x[1], x[2]))
        remaining.remove(min_job)
        idx, t1, t2 = min_job
        if t1 <= t2:
            order_start.append(idx)
        else:
            order_end.insert(0, idx)
    return order_start + order_end


def calculate_makespan(order: Sequence[int], jobs: Sequence[Tuple[float, float]] | Sequence[Tuple[str, float, float]]) -> float:
    """Calcula el makespan de un orden dado para dos máquinas.

    Args:
        order: secuencia de índices de los trabajos.
        jobs: lista de trabajos en el mismo formato que en
          `johnson_schedule`.

    Returns:
        Tiempo total (makespan) necesario para procesar todos los trabajos
        según el orden dado.
    """
    # Normalizar a (t1, t2)
    norm: List[Tuple[float, float]] = []
    for job in jobs:
        if len(job) == 2:
            t1, t2 = job
            norm.append((float(t1), float(t2)))
        else:
            _, t1, t2 = job
            norm.append((float(t1), float(t2)))
    time_m1 = 0.0
    time_m2 = 0.0
    for idx in order:
        t1, t2 = norm[idx]
        time_m1 += t1
        # empieza en máquina 2 cuando termine en m1 y la m2 esté libre
        if time_m2 < time_m1:
            time_m2 = time_m1 + t2
        else:
            time_m2 += t2
    return time_m2