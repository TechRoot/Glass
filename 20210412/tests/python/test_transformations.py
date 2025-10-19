import numpy as np
import pytest

from ...lib import transformations as tf
from ...lib import homography
from ...lib import kabsch


def test_homography_estimate_and_apply():
    # correspondencias para escalado en el eje x por 2
    src = [(0, 0), (1, 0), (0, 1), (1, 1)]
    dst = [(0, 0), (2, 0), (0, 1), (2, 1)]
    H = homography.estimate_homography(src, dst)
    expected = np.array([[2.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [0.0, 0.0, 1.0]])
    assert np.allclose(H, expected, atol=1e-6)
    # aplicar homografía a un punto interior
    point = [(0.5, 0.5)]
    trans = homography.apply_homography(point, H)
    assert np.allclose(trans[0], [1.0, 0.5], atol=1e-6)


def test_kabsch_alignment():
    # definimos tres puntos en referencia y los rotamos + trasladamos
    P = np.array([[0.0, 0.0, 0.0],
                  [1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0]])
    angle = np.pi / 4
    R_true = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                       [np.sin(angle),  np.cos(angle), 0.0],
                       [0.0,            0.0,           1.0]])
    t_true = np.array([1.0, 2.0, 3.0])
    Q = (R_true @ P.T).T + t_true
    R, t, rmsd = kabsch.kabsch(P, Q)
    assert np.allclose(R, R_true, atol=1e-5)
    assert np.allclose(t, t_true, atol=1e-5)
    assert rmsd < 1e-12
    # probar incremental
    inc = kabsch.KabschIncremental()
    for p, q in zip(P, Q):
        inc.add_points([p], [q])
    result = inc.get_transform()
    assert result is not None
    R_inc, t_inc = result
    assert np.allclose(R_inc, R_true, atol=1e-4)
    assert np.allclose(t_inc, t_true, atol=1e-4)


def test_quaternion_conversion():
    # cuaternión correspondiente a rotación de 45° alrededor de z
    angle = np.pi / 4
    q = np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])
    R = tf.quaternion_to_rotation(q)
    q2 = tf.rotation_to_quaternion(R)
    # el cuaternión recuperado puede diferir en signo pero representa la misma rotación
    assert np.allclose(np.abs(q2), np.abs(q), atol=1e-5)


def test_spectral_radius():
    A = np.array([[0.5, 0.0], [0.0, 0.2]])
    radius = tf.spectral_radius(A)
    assert abs(radius - 0.5) < 1e-9