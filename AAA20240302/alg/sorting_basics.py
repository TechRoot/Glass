"""
Algoritmos clásicos de ordenación y búsqueda binaria.

Este módulo implementa merge sort, quick sort y heap sort, junto con una
rutina de búsqueda binaria.  Estos ejemplos ilustran el paradigma de
divide y vencerás【121111711656550†L8-L59】.

Dependencias: solo la biblioteca estándar de Python
"""

from __future__ import annotations

from typing import Iterable, List, TypeVar

T = TypeVar('T')


def merge_sort(arr: List[T]) -> List[T]:
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    merged: List[T] = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i]); i += 1
        else:
            merged.append(right[j]); j += 1
    merged.extend(left[i:]); merged.extend(right[j:])
    return merged


def quick_sort(arr: List[T]) -> List[T]:
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    lesser = [x for x in arr if x < pivot]
    equal = [x for x in arr if x == pivot]
    greater = [x for x in arr if x > pivot]
    return quick_sort(lesser) + equal + quick_sort(greater)


def heap_sort(arr: List[T]) -> List[T]:
    import heapq
    heap = list(arr)
    heapq.heapify(heap)
    return [heapq.heappop(heap) for _ in range(len(heap))]


def binary_search(arr: List[T], target: T) -> int:
    """Devuelve el índice de ``target`` en ``arr`` ordenado o −1 si no se encuentra."""
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
