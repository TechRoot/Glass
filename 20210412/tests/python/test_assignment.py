import csv
from pathlib import Path

from ...lib import assignment
from ...lib import scheduling


def test_hungarian_algorithm_small_matrix():
    # Matriz de ejemplo (3x3) con solución conocida
    matrix = [
        [4.0, 1.0, 3.0],
        [2.0, 0.0, 5.0],
        [3.0, 2.0, 2.0],
    ]
    assign, total = assignment.hungarian(matrix)
    # La asignación óptima es (fila0,col1), (fila1,col0), (fila2,col2) con coste 1+2+2=5
    assign_set = set(assign)
    assert (0, 1) in assign_set and (1, 0) in assign_set and (2, 2) in assign_set
    assert abs(total - 5.0) < 1e-6


def test_hungarian_with_non_square():
    # Matriz rectangular 2x3
    matrix = [
        [10.0, 2.0, 9.0],
        [7.0, 5.0, 3.0],
    ]
    assign, total = assignment.hungarian(matrix)
    # Solución manual: fila0->col1 (2), fila1->col2 (3) = 5
    assign_set = set(assign)
    assert (0, 1) in assign_set and (1, 2) in assign_set
    assert abs(total - 5.0) < 1e-6


def test_johnson_schedule_and_makespan():
    # Trabajos: id, t1, t2
    jobs = [
        ('A', 2.0, 4.0),
        ('B', 3.0, 1.0),
        ('C', 4.0, 3.0),
        ('D', 2.0, 2.0),
    ]
    order = scheduling.johnson_schedule(jobs)
    # Esperamos el orden [0(A), 3(D), 2(C), 1(B)]
    assert order == [0, 3, 2, 1]
    makespan = scheduling.calculate_makespan(order, jobs)
    # Calcular manualmente el makespan: A(0-2,2-6), D(2-4,6-8), C(4-8,8-11), B(8-11,11-12) => 12
    assert abs(makespan - 12.0) < 1e-6