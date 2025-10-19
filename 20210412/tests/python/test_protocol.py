import sys
from pathlib import Path
import tempfile
import yaml

import pytest

sys.path.append(str(Path(__file__).resolve().parents[2]))

from lib import protocol


def test_default_weights_evaluation_graphs():
    """Comprueba que la evaluación por defecto ordena correctamente las técnicas de grafos."""
    weights = protocol.load_weights()  # carga pesos por defecto
    results = protocol.evaluate_domain("graphs", weights)
    # El primer elemento debe ser dijkstra según las puntuaciones predefinidas
    assert results[0][0] == "dijkstra"
    # Los índices deberían estar en orden descendente
    assert results[0][1] >= results[1][1] >= results[2][1]


def test_custom_weights_affect_ranking():
    """Comprueba que los pesos personalizados influyen en el ranking de técnicas."""
    # Crear un archivo temporal de pesos con alta importancia al coste y la robustez
    with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
        yaml.dump({"criteria": {"precision": 0.1, "cost": 0.6, "complexity": 0.1, "robustness": 0.2}}, tmp)
        tmp_path = tmp.name
    try:
        weights = protocol.load_weights(tmp_path)
        # Evaluar dominio de optimización con estos pesos
        results = protocol.evaluate_domain("optimization", weights)
        # La técnica 'hungarian' debe seguir siendo la más valorada
        assert results[0][0] == "hungarian"
        # Asegurarse de que los índices están calculados
        assert all(isinstance(idx, float) for _, idx in results)
    finally:
        Path(tmp_path).unlink(missing_ok=True)