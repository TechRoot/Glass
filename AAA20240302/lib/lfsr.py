"""
Implementación de un registro de desplazamiento con realimentación lineal (LFSR).

Un LFSR genera secuencias pseudorandom de bits desplazando un registro y
retroalimentando una combinación lineal de sus bits.  Este módulo implementa
un LFSR de n bits dado un polinomio de taps y proporciona métodos para
avanzar el estado y calcular el período de la secuencia【486514497268836†L17-L67】.

Dependencias: solo la biblioteca estándar de Python.
"""

from __future__ import annotations

from typing import List


class LFSR:
    """Registro de desplazamiento con realimentación lineal definido por un polinomio de taps."""

    def __init__(self, seed: int, taps: List[int], length: int):
        if seed <= 0 or seed >= (1 << length):
            raise ValueError("Seed must be non-zero and fit in the register length")
        self.state = seed
        self.taps = taps
        self.length = length

    def step(self) -> int:
        """Avanza el LFSR un paso y devuelve el bit de salida."""
        xor = 0
        for t in self.taps:
            xor ^= (self.state >> t) & 1
        out = self.state & 1
        self.state = (self.state >> 1) | (xor << (self.length - 1))
        return out

    def generate(self, n: int) -> List[int]:
        """Genera ``n`` bits a partir del LFSR."""
        return [self.step() for _ in range(n)]

    def period(self) -> int:
        """Calcula el período de la secuencia generada por el LFSR【486514497268836†L17-L67】."""
        seen = {}
        count = 0
        while self.state not in seen:
            seen[self.state] = True
            self.step()
            count += 1
        return count
