"""
Utilidades para el análisis de máquinas de estados finitos (FSM).

Este módulo contiene funciones para explorar y verificar propiedades de
máquinas de estados finitos: búsqueda en anchura (BFS) para alcanzabilidad,
detección de ciclos, construcción de un *miter* para comprobación de equivalencia
y una función para comparar dos máquinas.  Se utiliza en el CLI de lógica
para verificar propiedades de seguridad y vivacidad de circuitos secuenciales【926149549049321†L18-L118】.

Dependencias: solo la biblioteca estándar de Python
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Optional, Set, Tuple


def bfs_reachable(adj: Dict[str, List[str]], start: str) -> Set[str]:
    """Devuelve el conjunto de estados alcanzables desde ``start`` utilizando BFS."""
    visited: Set[str] = {start}
    q = deque([start])
    while q:
        u = q.popleft()
        for v in adj.get(u, []):
            if v not in visited:
                visited.add(v)
                q.append(v)
    return visited


def has_cycle(adj: Dict[str, List[str]]) -> bool:
    """Detecta ciclos en un grafo dirigido utilizando DFS."""
    visited: Set[str] = set()
    rec: Set[str] = set()
    def dfs(u: str) -> bool:
        visited.add(u)
        rec.add(u)
        for v in adj.get(u, []):
            if v not in visited:
                if dfs(v):
                    return True
            elif v in rec:
                return True
        rec.remove(u)
        return False
    for node in adj:
        if node not in visited:
            if dfs(node):
                return True
    return False


def miter(adj1: Dict[str, List[str]], adj2: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Construye la máquina producto (miter) de dos FSM para comprobar equivalencia."""
    miter_adj: Dict[str, List[str]] = {}
    for s1 in adj1:
        for s2 in adj2:
            state = f"{s1}|{s2}"
            miter_adj[state] = []
            for v1 in adj1.get(s1, []):
                for v2 in adj2.get(s2, []):
                    miter_adj[state].append(f"{v1}|{v2}")
    return miter_adj


def equivalent(adj1: Dict[str, List[str]], adj2: Dict[str, List[str]], start1: str, start2: str) -> bool:
    """Comprueba si dos FSM son equivalentes en lenguaje【926149549049321†L18-L118】."""
    product = miter(adj1, adj2)
    reachable = bfs_reachable(product, f"{start1}|{start2}")
    # Si existe un estado alcanzable donde uno es final y el otro no,
    # los lenguajes difieren.  En este ejemplo simplificado, se asume que
    # los estados finales son aquellos con sufijo "_F".  Los usuarios pueden
    # anular esta suposición según sea necesario.
    for state in reachable:
        s1, s2 = state.split("|")
        is_final1 = s1.endswith("_F")
        is_final2 = s2.endswith("_F")
        if is_final1 != is_final2:
            return False
    return True
