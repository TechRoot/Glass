// Pseudoinversa y número de condición
//
// Este módulo implementa una aproximación simple de la pseudoinversa de Moore–Penrose
// para matrices de dimensión arbitraria utilizando la fórmula
// A⁺ = (Aᵀ A)⁻¹ Aᵀ si A tiene rango completo por columnas, o
// A⁺ = Aᵀ (A Aᵀ)⁻¹ si tiene rango completo por filas. El cálculo de la
// inversión se realiza mediante eliminación de Gauss–Jordan.
// También se ofrece una estimación del número de condición basada en las
// normas de Frobenius: κ ≈ ‖A‖_F · ‖A⁺‖_F.

#pragma once

#include <vector>

namespace lin {

using Matrix = std::vector<std::vector<double>>;

// Calcula la inversa de una matriz cuadrada utilizando eliminación de Gauss.
// Lanza std::runtime_error si la matriz es singular.
Matrix inverse(const Matrix &A);

// Devuelve la pseudoinversa de Moore–Penrose de una matriz A.
Matrix pseudoinverse(const Matrix &A);

// Devuelve una estimación del número de condición de A.
double condition_number(const Matrix &A);

}