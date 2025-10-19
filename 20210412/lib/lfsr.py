"""
Generador de secuencias pseudoaleatorias mediante registros de desplazamiento con
realimentación lineal (LFSR).

Un LFSR de longitud n mantiene un vector de bits y en cada paso calcula un
bit de retroalimentación como la suma módulo 2 (XOR) de ciertos bits
especificados por `taps`. Luego desplaza todos los bits hacia la derecha y
coloca el bit calculado en la primera posición.

Esta implementación permite generar secuencias y estimar el periodo (longitud
hasta repetir el estado inicial) de forma sencilla.
"""
from __future__ import annotations

from typing import List, Iterable, Tuple, Optional, Set


class LFSR:
    """Clase para gestionar un registro de desplazamiento con realimentación lineal.

    Args:
        taps: índices de los bits que se retroalimentan (0 corresponde al primer bit).
        seed: lista de bits iniciales (valores 0/1).
    """
    def __init__(self, taps: Iterable[int], seed: Iterable[int]):
        self.taps: List[int] = list(taps)
        self.state: List[int] = list(seed)
        if not self.state:
            raise ValueError("La semilla no puede estar vacía")
        if any(b not in (0, 1) for b in self.state):
            raise ValueError("La semilla debe contener solo 0 y 1")
        if any(t < 0 or t >= len(self.state) for t in self.taps):
            raise ValueError("Los taps deben estar entre 0 y n-1")

    def step(self) -> int:
        """Realiza un paso del LFSR y devuelve el bit de salida (último bit)."""
        # calcular el nuevo bit como XOR de los bits en las posiciones de taps
        new_bit = 0
        for t in self.taps:
            new_bit ^= self.state[t]
        # salida actual (último bit)
        out_bit = self.state[-1]
        # desplazar y anteponer el nuevo bit
        self.state = [new_bit] + self.state[:-1]
        return out_bit

    def generate(self, length: int) -> List[int]:
        """Genera una secuencia de bits de longitud `length`."""
        return [self.step() for _ in range(length)]

    def period(self, max_iterations: Optional[int] = None) -> int:
        """Calcula el periodo de la secuencia hasta que se repite el estado inicial.

        Args:
            max_iterations: número máximo de iteraciones a explorar (por defecto 2^n).

        Returns:
            Longitud del ciclo antes de regresar al estado inicial. Si no se
            encuentra el periodo dentro de `max_iterations`, se devuelve -1.
        """
        initial = tuple(self.state)
        if max_iterations is None:
            max_iterations = 2 ** len(self.state)
        for i in range(1, max_iterations + 1):
            self.step()
            if tuple(self.state) == initial:
                return i
        return -1