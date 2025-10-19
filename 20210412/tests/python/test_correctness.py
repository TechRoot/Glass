import math

import numpy as np

from 20210412.lib import correctness
from 20210412.lib import assignment
from 20210412.lib import crc


def test_nilpotent_matrix():
    # Matriz de adyacencia de un autómata que termina en 2 pasos
    A = [
        [0, 1, 0],  # S -> T
        [0, 0, 1],  # T -> E
        [0, 0, 0],  # E sin transiciones
    ]
    is_nilp, steps = correctness.check_nilpotent_and_steps(A)
    assert is_nilp is True
    assert steps == 3  # A^3 = 0, aunque A^2 ya no tiene 1s en primeras filas


def test_rotation_invariants():
    # Matriz de rotación 90 grados alrededor del eje z
    R = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    assert correctness.check_rotation_invariants(R)
    # Matriz que no es rotación (det = -1)
    R_bad = np.array([
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ])
    assert correctness.check_rotation_invariants(R_bad) is False


def test_verify_primal_dual():
    # Cost matrix para asignación 2x2
    C = [
        [4.0, 1.0],
        [2.0, 3.0],
    ]
    # Asignación propuesta (fila,columna)
    assign, opt_cost = assignment.hungarian(C)
    assert correctness.verify_primal_dual(C, assign)
    # Asignación subóptima
    bad_assign = [(0, 0), (1, 1)]
    assert correctness.verify_primal_dual(C, bad_assign) is False


def test_verify_inverse_mod():
    # Inverso de 3 mod 11 es 4 (3*4=12 ≡1 mod 11)
    assert correctness.verify_inverse_mod(3, 11, 4)
    assert correctness.verify_inverse_mod(10, 17, 12)
    assert correctness.verify_inverse_mod(3, 11, 5) is False


def test_verify_crc_error():
    data = b"123456789"
    crc_val = crc.crc8(data)
    assert correctness.verify_crc_error(data, crc_val)
    # Introducir un error en los datos: debería fallar
    bad_data = b"123456788"
    assert correctness.verify_crc_error(bad_data, crc_val) is False