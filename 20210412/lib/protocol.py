"""Protocolo de implantación y evaluación global.

Este módulo define un protocolo genérico para seleccionar técnicas
algebraicas y algorítmicas en función de un problema concreto y para
evaluar soluciones mediante matrices de ponderación.  Cada técnica
disponible se describe mediante una serie de criterios (por ejemplo,
precisión, coste, complejidad computacional y robustez) con puntuaciones
normalizadas entre 0 y 1.  El usuario puede suministrar un archivo de
pesos (YAML) para ajustar la importancia relativa de cada criterio.
"""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# Inventario de técnicas disponibles por dominio.  Para cada técnica se
# asignan puntuaciones en diferentes criterios: precision, cost,
# complexity y robustness.  Estos valores son puramente ilustrativos y
# pueden ajustarse según el contexto industrial.
DOMAIN_TO_TECHNIQUES: Dict[str, Dict[str, Dict[str, float]]] = {
    "boolean_logic": {
        "quine_mccluskey": {"precision": 0.9, "cost": 0.5, "complexity": 0.7, "robustness": 0.8},
        "karnaugh_map": {"precision": 0.8, "cost": 0.6, "complexity": 0.6, "robustness": 0.7},
    },
    "linear_algebra": {
        "svd": {"precision": 0.9, "cost": 0.7, "complexity": 0.8, "robustness": 0.9},
        "kabsch": {"precision": 0.85, "cost": 0.5, "complexity": 0.7, "robustness": 0.8},
        "homography_dlt": {"precision": 0.8, "cost": 0.4, "complexity": 0.6, "robustness": 0.7},
    },
    "modular_arithmetic": {
        "extended_euclid": {"precision": 0.9, "cost": 0.3, "complexity": 0.5, "robustness": 0.8},
        "crt": {"precision": 0.85, "cost": 0.4, "complexity": 0.5, "robustness": 0.7},
    },
    "graphs": {
        "dijkstra": {"precision": 0.9, "cost": 0.6, "complexity": 0.7, "robustness": 0.8},
        "topological_sort": {"precision": 0.8, "cost": 0.3, "complexity": 0.4, "robustness": 0.8},
        "fiedler_value": {"precision": 0.7, "cost": 0.5, "complexity": 0.6, "robustness": 0.7},
    },
    "optimization": {
        "hungarian": {"precision": 0.9, "cost": 0.7, "complexity": 0.8, "robustness": 0.9},
        "johnson": {"precision": 0.85, "cost": 0.5, "complexity": 0.5, "robustness": 0.8},
        "cutting_heuristic": {"precision": 0.6, "cost": 0.4, "complexity": 0.3, "robustness": 0.6},
    },
    "interpolation": {
        "lagrange": {"precision": 0.8, "cost": 0.5, "complexity": 0.6, "robustness": 0.7},
        "newton": {"precision": 0.8, "cost": 0.5, "complexity": 0.7, "robustness": 0.7},
        "least_squares": {"precision": 0.7, "cost": 0.4, "complexity": 0.7, "robustness": 0.8},
    },
    "correctness": {
        "nilpotent_check": {"precision": 0.7, "cost": 0.3, "complexity": 0.4, "robustness": 0.7},
        "rotation_invariants": {"precision": 0.8, "cost": 0.3, "complexity": 0.4, "robustness": 0.8},
        "primal_dual": {"precision": 0.85, "cost": 0.4, "complexity": 0.5, "robustness": 0.8},
    },
}


def load_weights(path: Optional[str] = None) -> Dict[str, float]:
    """Carga un diccionario de pesos desde un archivo YAML.

    Si no se especifica `path`, se carga el archivo `etc/weights_default.yml`.
    El YAML debe definir una clave `criteria` con pesos para cada
    criterio (precision, cost, complexity, robustness).  Si faltan
    criterios, se rellenan con valores iguales para mantener la suma a 1.
    """
    if path is None:
        path = Path(__file__).resolve().parents[1] / "etc" / "weights_default.yml"
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    weights = data.get("criteria", {})
    # Normalizar pesos: cualquier clave no especificada toma valor cero antes de normalizar
    all_keys = {"precision", "cost", "complexity", "robustness"}
    total = 0.0
    norm_weights: Dict[str, float] = {}
    for k in all_keys:
        v = float(weights.get(k, 0.0))
        norm_weights[k] = v
        total += v
    if total == 0:
        # Distribuir equitativamente
        for k in all_keys:
            norm_weights[k] = 1.0 / len(all_keys)
    else:
        for k in all_keys:
            norm_weights[k] /= total
    return norm_weights


def evaluate_domain(domain: str, weights: Dict[str, float]) -> List[Tuple[str, float]]:
    """Calcula el índice global para cada técnica de un dominio.

    El índice se obtiene como la suma ponderada de las puntuaciones de
    precisión, coste, complejidad y robustez según los pesos
    suministrados.  Se devuelve una lista de pares (técnica, índice)
    ordenada de mayor a menor.

    Parámetros
    ----------
    domain : str
        Nombre del dominio (por ejemplo, 'optimization', 'graphs').
    weights : dict
        Pesos normalizados para cada criterio.

    Retorna
    -------
    List[Tuple[str, float]]
        Lista de (técnica, índice) ordenada por índice descendente.
    """
    domain = domain.lower()
    if domain not in DOMAIN_TO_TECHNIQUES:
        raise ValueError(f"Dominio desconocido: {domain}")
    techniques = DOMAIN_TO_TECHNIQUES[domain]
    results: List[Tuple[str, float]] = []
    for name, scores in techniques.items():
        idx = 0.0
        for crit, w in weights.items():
            idx += scores.get(crit, 0.0) * w
        results.append((name, idx))
    # Ordenar de mayor a menor índice
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def recommend_techniques(domain: str, weights_path: Optional[str] = None) -> List[Tuple[str, float]]:
    """Devuelve la lista de técnicas recomendadas para un dominio.

    Carga los pesos desde el archivo indicado o desde la configuración
    por defecto, calcula los índices y devuelve la lista ordenada.
    """
    weights = load_weights(weights_path)
    return evaluate_domain(domain, weights)


def global_index(domain: str, weights_path: Optional[str] = None) -> float:
    """Calcula un índice global de madurez para un dominio.

    Se define como el máximo índice entre las técnicas disponibles en
    dicho dominio bajo los pesos especificados.  Proporciona una medida
    de la capacidad potencial del conjunto de herramientas para ese
    problema.
    """
    results = recommend_techniques(domain, weights_path)
    return results[0][1] if results else 0.0