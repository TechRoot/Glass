"""
Códigos Reed–Solomon sencillos sobre GF(2^8).

Este módulo implementa un codificador y decodificador básico de Reed–Solomon (RS)
sobre el campo de Galois GF(2^8) utilizando un polinomio primitivo.  Los códigos
RS proporcionan detección y corrección de errores  utilizados en comunicaciones
digitales y almacenamiento.  【980447337874345†L24-L176】.

Nota: esta implementación opera sobre arreglos de bytes y devuelve síndromes
para la detección de errores.  La decodificación completa (localización y
corrección de errores) requiere los algoritmos de Berlekamp–Massey y la
búsqueda de Chien, que están fuera del alcance de este ejemplo.
"""

from __future__ import annotations

from typing import List

# Primitive polynomial for GF(2^8)
PRIM_POLY = 0x11d

# Precalcula
EXP_TABLE = [0] * 512
LOG_TABLE = [0] * 256
def _init_tables() -> None:
    x = 1
    for i in range(255):
        EXP_TABLE[i] = x
        LOG_TABLE[x] = i
        x <<= 1
        if x & 0x100:
            x ^= PRIM_POLY
    for i in range(255, 512):
        EXP_TABLE[i] = EXP_TABLE[i - 255]

_init_tables()


def gf_mul(a: int, b: int) -> int:
    """Multiplica dos elementos en GF(2^8)【980447337874345†L24-L176】."""
    if a == 0 or b == 0:
        return 0
    return EXP_TABLE[LOG_TABLE[a] + LOG_TABLE[b]]


def gf_pow(a: int, power: int) -> int:
    """Eleva ``a`` a la ``power`` en GF(2^8)."""
    if a == 0:
        return 0
    return EXP_TABLE[(LOG_TABLE[a] * power) % 255]


def gf_inverse(a: int) -> int:
    """Calcula el inverso multiplicativo de ``a`` en GF(2^8)."""
    if a == 0:
        raise ZeroDivisionError
    return EXP_TABLE[255 - LOG_TABLE[a]]


def poly_mul(p: List[int], q: List[int]) -> List[int]:
    """Multiplica dos polinomios sobre GF(2^8)."""
    r = [0] * (len(p) + len(q) - 1)
    for i, coeff_p in enumerate(p):
        for j, coeff_q in enumerate(q):
            r[i + j] ^= gf_mul(coeff_p, coeff_q)
    return r


def poly_eval(poly: List[int], x: int) -> int:
    """Evalúa un polinomio en ``x`` sobre GF(2^8)."""
    result = 0
    for coeff in poly:
        result = gf_mul(result, x) ^ coeff
    return result


def generate_generator_poly(nsym: int) -> List[int]:
    """Genera un polinomio generador para un código RS (n,k).

    ``nsym`` es el número de símbolos de corrección de errores【980447337874345†L24-L176】.
    """
    g = [1]
    for i in range(nsym):
        g = poly_mul(g, [1, gf_pow(2, i)])
    return g


def rs_encode(data: List[int], nsym: int) -> List[int]:
    """Codifica los datos con un código Reed–Solomon de ``nsym`` símbolos de verificación."""
    gen = generate_generator_poly(nsym)
    out = list(data) + [0] * nsym
    for i in range(len(data)):
        coef = out[i]
        if coef != 0:
            for j in range(len(gen)):
                out[i + j] ^= gf_mul(gen[j], coef)
    code = data + out[-nsym:]
    return code


def rs_calc_syndromes(code: List[int], nsym: int) -> List[int]:
    """Calcula los síndromes de un código Reed–Solomon."""
    synd = []
    for i in range(nsym):
        synd.append(poly_eval(code, gf_pow(2, i)))
    return synd


def rs_check(code: List[int], nsym: int) -> bool:
    """Devuelve True si todos los síndromes son cero (es decir, sin errores)."""
    return max(rs_calc_syndromes(code, nsym)) == 0
