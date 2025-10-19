import sys
import json
from pathlib import Path

import pytest

# Añadir ruta del proyecto para importación de módulos locales
sys.path.append(str(Path(__file__).resolve().parents[2]))

from lib import boolean_logic as bl
from lib import fsm_utils as fu


def test_quine_mccluskey_majority():
    # Minterminos para la función de mayoría 2-de-3
    minterms = [3, 5, 6, 7]
    patterns = bl.minimize_function(minterms, num_vars=3)
    # Los implicantes primos para mayoría deben ser tres términos con un guion
    assert set(patterns) == {"11-", "1-1", "-11"}
    vars_list = ['A', 'B', 'C']
    expressions = [bl.pattern_to_expression(p, vars_list) for p in patterns]
    assert set(expressions) == {"A & B", "A & C", "B & C"}


def test_consensus_term():
    term1 = ('A', '~B')
    term2 = ('~A', 'C')
    consensus = bl.add_consensus(term1, term2)
    # el término de consenso esperado es C & ~B (orden no importa)
    expected_literals = {'C', '~B'}
    result_literals = {x.strip() for x in consensus.split('&')}
    assert result_literals == expected_literals


def test_boolean_derivative_and_timing():
    # f(A,B) = A AND B
    def func(a, b):
        return a & b
    deriv = bl.boolean_derivative(func, var_index=0, num_vars=2)
    results = {bits: deriv(*bits) for bits in [(0,0),(0,1),(1,0),(1,1)]}
    # derivada respecto a A es B
    assert results[(0,0)] == 0
    assert results[(0,1)] == 1
    assert results[(1,0)] == 0
    assert results[(1,1)] == 1
    # margen temporal
    assert bl.timing_margin(10.0, 3.0, 2.0) == pytest.approx(5.0)


def test_fsm_reachability_and_cycle():
    adj = {
        "S": ["T"],
        "T": ["E"],
        "E": []
    }
    reachable, nodes = fu.bfs_reachability(adj, "S", "E")
    assert reachable is True and nodes == 3
    assert fu.has_cycle(adj, "S") is False
    # introducimos un ciclo
    adj_cycle = {"A": ["B"], "B": ["C"], "C": ["A"]}
    assert fu.has_cycle(adj_cycle, "A") is True


def test_miter_equivalence_and_parity():
    # Función de paridad de 2 bits vs XOR
    def f1(a, b):
        return a ^ b
    def f2(a, b):
        return bl.parity_bit([a, b])  # reusa la paridad impar
    assert fu.check_equivalence(f1, f2, 2) is True
    # paridad con 3 bits
    assert fu.parity_bit([1,0,1]) == 0  # 1+0+1=2 -> paridad par
    assert fu.parity_bit([1,1,1]) == 1  # 3 -> paridad impar


def test_doublepulse():
    # no armado y dos pulsaciones activas -> salida verdadera
    assert fu.double_pulse(True, True, False) is True
    # ya armado -> salida falsa
    assert fu.double_pulse(True, True, True) is False
    # faltan pulsaciones -> salida falsa
    assert fu.double_pulse(True, False, False) is False