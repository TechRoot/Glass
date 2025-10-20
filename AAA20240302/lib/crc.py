"""
Utilidades de código de redundancia cíclica (CRC).

Este módulo implementa un algoritmo CRC‑8 sencillo y proporciona una función
para verificar un mensaje con su CRC.  Los CRC se utilizan para detectar
errores en la transmisión y almacenamiento de datos.  La implementación
utiliza el polinomio 0x07, común en estándares de comunicaciones.

Dependencias: solo la biblioteca estándar de Python.

Autor: A. Alonso
"""

from __future__ import annotations

from typing import Iterable


CRC8_POLY = 0x07


def crc8(data: Iterable[int]) -> int:
    """Calcula el CRC‑8 de la secuencia de bytes dada.

    Parámetros
    ----------
    data : iterable de int
        Bytes (0–255) para los que se va a calcular el CRC.

    Devuelve
    -------
    crc : int
        El valor CRC calculado (0–255).
    """
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) & 0xFF) ^ CRC8_POLY
            else:
                crc = (crc << 1) & 0xFF
    return crc


def verify_crc8(data: Iterable[int], crc_value: int) -> bool:
    """Verifica que ``crc8(data)`` sea igual a ``crc_value``.

    Devuelve ``True`` si el CRC coincide, ``False`` en caso contrario【846800301731874†L14-L53】.
    """
    return crc8(data) == crc_value
