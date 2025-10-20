"""
Utilidades de aritmética modular: máximo común divisor extendido, inversos modulares y TCR.

Este módulo implementa operaciones básicas en aritmética modular: cálculo del
máximo común divisor (mcd) y coeficientes de Bézout mediante el algoritmo de
Euclides extendido, obtención de inversos modulares y resolución de sistemas de
congruencias utilizando el teorema chino del resto (TCR).

Dependencias: solo la biblioteca estándar de Python

Autor: A. Alonso
commit: 849862447569455
Fecha: 2019-19-12
"""

from __future__ import annotations

from typing import List, Tuple


def egcd(a: int, b: int) -> Tuple[int, int, int]:
    """Devuelve ``(g, x, y)`` tal que ``g = mcd(a, b)`` y ``a*x + b*y = g``."""
    if b == 0:
        return (a, 1, 0)
    else:
        g, x1, y1 = egcd(b, a % b)
        return (g, y1, x1 - (a // b) * y1)


def modinv(a: int, m: int) -> int:
    """Devuelve el inverso modular de ``a`` módulo ``m``.

    Lanza ``ValueError`` si el inverso no existe (es decir, si ``a`` y
    ``m`` no son coprimos).
    """
    g, x, _ = egcd(a, m)
    if g != 1:
        raise ValueError(f"{a} has no inverse modulo {m}")
    return x % m


def crt(remainders: List[int], moduli: List[int]) -> Tuple[int, int]:
    """Resuelve el sistema x ≡ r_i (mod m_i) para i=0..k−1.

    Devuelve un par ``(x, M)`` tal que todas las soluciones son congruentes a
    x módulo M (el producto de los módulos), suponiendo que los módulos son
    coprimos dos a dos.  Si los módulos no son coprimos, se lanza ``ValueError``.
    """
    if len(remainders) != len(moduli):
        raise ValueError("remainders and moduli must have the same length")
    x = 0
    M = 1
    for r, m in zip(remainders, moduli):
        g, _, _ = egcd(M, m)
        if g != 1:
            raise ValueError("Moduli must be pairwise coprime for CRT")
        inv = modinv(M, m)
        k = ((r - x) * inv) % m
        x += k * M
        M *= m
    return x % M, M
