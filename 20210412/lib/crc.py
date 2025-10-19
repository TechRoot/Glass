"""
Cálculo de códigos de redundancia cíclica (CRC) y verificación.

Este módulo implementa una versión simple del algoritmo CRC-8. A partir de
una secuencia de bytes y un polinomio generador se calcula el resto de la
división polinómica bit a bit.  También ofrece una función de verificación
que comprueba si un mensaje acompañado de su CRC es divisible por el
polinomio generador (lo que indica que no hay errores detectables).
"""
from __future__ import annotations

from typing import Iterable


def crc8(data: Iterable[int] | bytes, poly: int = 0x07, init: int = 0x00) -> int:
    """Calcula un CRC-8 sobre una secuencia de bytes.

    Args:
        data: iterable de enteros en el rango 0..255 o `bytes`.
        poly: polinomio generador representado como entero (por defecto x^8 + x^2 + x + 1).
        init: valor inicial del registro CRC.

    Returns:
        Resto de la división polinómica (valor CRC entre 0 y 255).
    """
    if isinstance(data, bytes):
        data_iter = data
    else:
        data_iter = bytes(data)
    crc = init
    for byte in data_iter:
        crc ^= byte
        for _ in range(8):
            # si el bit más significativo está a 1, aplicar el polinomio
            if crc & 0x80:
                crc = ((crc << 1) & 0xFF) ^ poly
            else:
                crc = (crc << 1) & 0xFF
    return crc & 0xFF


def verify_crc8(data: Iterable[int] | bytes, crc_value: int, poly: int = 0x07, init: int = 0x00) -> bool:
    """Comprueba si la concatenación de datos y su CRC es divisible por el polinomio.

    Args:
        data: datos originales como bytes o lista de enteros.
        crc_value: valor CRC a verificar.
        poly: polinomio generador utilizado.
        init: valor inicial del registro CRC utilizado.

    Returns:
        True si el CRC es válido, False en caso contrario.
    """
    full_data = bytes(list(data) + [crc_value])
    return crc8(full_data, poly, init) == 0