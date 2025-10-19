import math

import numpy as np

from 20210412.lib import interpolation as interp
from 20210412.lib import least_squares as ls


def test_lagrange_and_newton_equivalence():
    # Tres puntos definen un polinomio cuadrático f(x) = x^2 + 1
    xs = [0.0, 1.0, 2.0]
    ys = [1.0, 2.0, 5.0]
    # Valor esperado en x=1.5: 1 + 1.5^2 = 3.25
    x_eval = 1.5
    expected = 3.25
    y_lagrange = interp.lagrange_interpolate(xs, ys, x_eval)
    coeffs_newton = interp.newton_divided_differences(xs, ys)
    y_newton = interp.newton_eval(xs, coeffs_newton, x_eval)
    # La diferencia debe ser pequeña
    assert math.isclose(y_lagrange, expected, abs_tol=1e-9)
    assert math.isclose(y_newton, expected, abs_tol=1e-9)
    # Barycentric evaluation
    weights = interp.barycentric_weights(xs)
    y_bar = interp.barycentric_eval(xs, ys, weights, x_eval)
    assert math.isclose(y_bar, expected, abs_tol=1e-9)
    # Coeficientes Vandermonde deberían ser [1,0,1]
    coeffs_vand = interp.vandermonde_coeffs(xs, ys)
    assert np.allclose(coeffs_vand, [1.0, 0.0, 1.0], atol=1e-9)


def test_least_squares_methods():
    # Datos lineales exactos: y = 2x + 1
    xs = [0, 1, 2, 3]
    ys = [1, 3, 5, 7]
    # Construir la matriz de diseño de grado 1
    A = [[x, 1] for x in xs]
    # Normal equations
    x_normal, cond_normal = ls.normal_equations(A, ys)
    # SVD
    x_svd, cond_svd = ls.svd_least_squares(A, ys)
    # QR
    x_qr, cond_qr = ls.qr_least_squares(A, ys)
    # Todas las soluciones deben ser [2,1]
    for x_sol in (x_normal, x_svd, x_qr):
        assert math.isclose(x_sol[0], 2.0, rel_tol=1e-9, abs_tol=1e-9)
        assert math.isclose(x_sol[1], 1.0, rel_tol=1e-9, abs_tol=1e-9)
    # MSE debe ser cero (ajuste perfecto)
    for x_sol in (x_normal, x_svd, x_qr):
        mse = ls.mean_squared_error(A, x_sol, ys)
        assert math.isclose(mse, 0.0, abs_tol=1e-12)


def test_ridge_regression_regularization():
    # Datos lineales con ligero ruido
    xs = [0, 1, 2, 3]
    ys = [1.0, 3.1, 5.0, 7.05]
    A = [[x, 1] for x in xs]
    # Ridge con alpha positivo debería producir coeficientes cercanos
    x_ridge, cond_ridge = ls.ridge_regression(A, ys, alpha=0.1)
    # Esperamos pendiente y ordenada cercanas a 2 y 1
    assert abs(x_ridge[0] - 2.0) < 0.1
    assert abs(x_ridge[1] - 1.0) < 0.2