"""
Codificación y verificación simplificada de códigos Reed–Solomon sobre GF(2^8).

Esta implementación proporciona las funciones esenciales para generar el
polinomio generador de un código (n, k), codificar un mensaje mediante
división polinómica y calcular los síndromes de un código recibido. La
    decodificación se limita a comprobar si los síndromes son nulos; la
    corrección de errores requiere el uso de algoritmos adicionales como los de
    Berlekamp–Massey y Forney, que no están incluidos en este módulo.

Referencia: se utiliza el polinomio primitivo 0x11D para GF(2^8).
"""
from __future__ import annotations

from typing import List, Tuple

try:
    # Importar excepción personalizada para dependencias externas. Este módulo puede no
    # existir en sistemas reducidos; en ese caso se recurre a RuntimeError.
    from .exceptions import ExternalDependencyMissing  # type: ignore
except Exception:
    ExternalDependencyMissing = RuntimeError  # type: ignore

# Inicialización de tablas de logaritmos y antilogaritmos para GF(2^8)
GF_EXP = [0] * 512
GF_LOG = [0] * 256

def _init_tables() -> None:
    primitive = 0x11D
    x = 1
    for i in range(255):
        GF_EXP[i] = x
        GF_LOG[x] = i
        x <<= 1
        if x & 0x100:
            x ^= primitive
    for i in range(255, 512):
        GF_EXP[i] = GF_EXP[i - 255]
_init_tables()


def gf_add(x: int, y: int) -> int:
    return x ^ y


def gf_sub(x: int, y: int) -> int:
    return x ^ y


def gf_mul(x: int, y: int) -> int:
    if x == 0 or y == 0:
        return 0
    return GF_EXP[GF_LOG[x] + GF_LOG[y]]


def gf_div(x: int, y: int) -> int:
    if y == 0:
        raise ZeroDivisionError
    if x == 0:
        return 0
    return GF_EXP[(GF_LOG[x] - GF_LOG[y]) % 255]


def gf_pow(x: int, p: int) -> int:
    return GF_EXP[(GF_LOG[x] * p) % 255] if x != 0 else 0


def gf_inverse(x: int) -> int:
    if x == 0:
        raise ZeroDivisionError
    return GF_EXP[255 - GF_LOG[x]]


def poly_add(p: List[int], q: List[int]) -> List[int]:
    # Ajustar longitud
    length = max(len(p), len(q))
    res = [0] * length
    for i in range(len(p)):
        res[i + length - len(p)] ^= p[i]
    for i in range(len(q)):
        res[i + length - len(q)] ^= q[i]
    return res


def poly_mul(p: List[int], q: List[int]) -> List[int]:
    res = [0] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            res[i + j] ^= gf_mul(a, b)
    return res


def poly_div(dividend: List[int], divisor: List[int]) -> Tuple[List[int], List[int]]:
    # Realiza la división polinómica y devuelve (cociente, resto).
    tmp = list(dividend)
    div_len = len(divisor)
    for i in range(len(dividend) - div_len + 1):
        coef = tmp[i]
        if coef != 0:
            for j in range(1, div_len):
                if divisor[j] != 0:
                    tmp[i + j] ^= gf_mul(divisor[j], coef)
    separator = -(div_len - 1)
    return tmp[:separator], tmp[separator:]


def rs_generator_poly(nsym: int) -> List[int]:
    """Genera el polinomio generador de un código RS con nsym símbolos de paridad."""
    g = [1]
    for i in range(nsym):
        g = poly_mul(g, [1, gf_pow(2, i)])
    return g


def rs_encode_msg(msg: List[int], nsym: int) -> List[int]:
    """Codifica un mensaje añadiendo `nsym` símbolos de paridad.

    Este algoritmo realiza una división polinómica en GF(2^8) sobre el
    polinomio mensaje. Se implementa siguiendo la forma de silla: se
    multiplica el polinomio generador por el coeficiente correspondiente y
    se resta (XOR) del mensaje extendido.
    """
    if nsym <= 0:
        raise ValueError("El número de símbolos de paridad debe ser positivo")
    gen = rs_generator_poly(nsym)
    # Copiar el mensaje y añadir espacio para la paridad
    msg_out = list(msg) + [0] * nsym
    for i in range(len(msg)):
        coef = msg_out[i]
        if coef != 0:
            for j in range(len(gen)):
                msg_out[i + j] ^= gf_mul(gen[j], coef)
    # Los últimos nsym coeficientes son el resto de la división
    return msg + msg_out[-nsym:]


def rs_calc_syndromes(codeword: List[int], nsym: int) -> List[int]:
    """Calcula la lista de síndromes de un codeword de longitud n+k.

    Cada síndrome s_i es la evaluación del polinomio c(x) en x = α^i, donde α es el
    elemento generador de GF(2^8). Se utiliza una evaluación horner.
    """
    def poly_eval(poly: List[int], x: int) -> int:
        """Evalúa un polinomio en el campo GF(2^8) usando Horner."""
        res = 0
        for coeff in poly:
            res = gf_mul(res, x) ^ coeff
        return res
    # Los síndromes se evalúan en α^i para i = 0..nsym-1
    return [poly_eval(codeword, gf_pow(2, i)) for i in range(nsym)]


def rs_check(codeword: List[int], nsym: int) -> bool:
    """Devuelve True si el código no presenta errores detectables."""
    synd = rs_calc_syndromes(codeword, nsym)
    return all(s == 0 for s in synd)


def rs_decode(codeword: List[int], nsym: int) -> List[int]:
    """Decodificación simplificada de códigos Reed–Solomon.

    Devuelve el mensaje original (sin los símbolos de paridad) cuando los
    síndromes indican ausencia de errores.  Si se detecta un error de
    transmisión, lanza una excepción para indicar que la corrección de
    errores requiere la implementación de algoritmos adicionales no
    incluidos en esta biblioteca.  Esta interfaz permite detectar fallos
    sin proporcionar una rutina de corrección completa.
    """
    if not rs_check(codeword, nsym):
        # La corrección de errores completa requiere la implementación de
        # algoritmos específicos como Berlekamp–Massey, que no están presentes
        # en esta biblioteca básica.  Se informa mediante la excepción
        # ExternalDependencyMissing con un mensaje sobrio.
        raise ExternalDependencyMissing(
            "se detectó un error en el codeword; requiere módulo externo de corrección"
        )
    return codeword[:-nsym]