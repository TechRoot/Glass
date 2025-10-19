"""Paquete de bibliotecas compartidas.

Este paquete agrupa utilidades para lógica booleana, máquinas de estados,
transformaciones geométricas, estimación de homografías, cálculo de
alineamientos rígidos y aritmética modular entre otros. Los módulos pueden
importarse individualmente según la necesidad.
"""

from . import boolean_logic  # noqa: F401
from . import fsm_utils  # noqa: F401
from . import transformations  # noqa: F401
from . import homography  # noqa: F401
from . import kabsch  # noqa: F401
from . import crc  # noqa: F401
from . import lfsr  # noqa: F401
from . import reed_solomon  # noqa: F401

# Algoritmos de grafos para cálculo de Laplaciana, Fiedler, Dijkstra, orden
# topológico y centralidad. Se importa aquí para facilitar el acceso desde
# otros módulos y scripts.
from . import graph_algorithms  # noqa: F401

# Algoritmos de optimización combinatoria: método Húngaro y regla de Johnson
from . import assignment  # noqa: F401
from . import scheduling  # noqa: F401

# Interpolación y mínimos cuadrados
from . import interpolation  # noqa: F401
from . import least_squares  # noqa: F401
from . import correctness  # noqa: F401

# Protocolo de evaluación global
from . import protocol  # noqa: F401