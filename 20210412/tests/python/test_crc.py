import pytest

from ...lib import crc, lfsr, reed_solomon


def test_crc8_known_vector():
    # Ejemplo estándar: CRC-8 de "123456789" con polinomio 0x07 da 0xF4
    data = b"123456789"
    crc_val = crc.crc8(data, poly=0x07, init=0x00)
    assert crc_val == 0xF4
    assert crc.verify_crc8(data, crc_val, poly=0x07, init=0x00)


def test_lfsr_period_and_sequence():
    taps = [0, 2]  # realimentación desde bit 0 y 2
    seed = [1, 0, 0]  # longitud 3
    l = lfsr.LFSR(taps, seed)
    seq = l.generate(7)
    # Para taps [0,2] y semilla [1,0,0] el periodo debe ser 7 (2^3-1)
    assert l.period() == 7
    # La secuencia generada no debe contener todos ceros hasta el periodo completo
    assert any(bit == 1 for bit in seq)


def test_reed_solomon_encode_and_check():
    msg = [1, 2, 3, 4, 5]
    nsym = 2
    code = reed_solomon.rs_encode_msg(msg, nsym)
    # longitudes: original + paridad
    assert len(code) == len(msg) + nsym
    # código correcto no debe presentar síndromes
    assert reed_solomon.rs_check(code, nsym)
    # decodificar un mensaje sin errores devuelve el original
    decoded = reed_solomon.rs_decode(code, nsym)
    assert decoded == msg
    # inducir un error y comprobar que se detecta
    corrupted = code.copy()
    corrupted[1] ^= 1  # cambiar un bit del segundo símbolo
    assert not reed_solomon.rs_check(corrupted, nsym)
    with pytest.raises(ValueError):
        reed_solomon.rs_decode(corrupted, nsym)