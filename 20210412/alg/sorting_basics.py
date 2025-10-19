"""Ordenación clásica y búsqueda binaria (n log n).

Uso previsto: preprocesamiento de lotes, eventos o pedidos para
pipelines de optimización; binsearch para umbrales/parametrización.
"""
from typing import List, Callable
import heapq, random, time

def mergesort(a: List[int]) -> List[int]:
    if len(a) <= 1: return a[:]
    mid = len(a)//2
    L = mergesort(a[:mid]); R = mergesort(a[mid:])
    i=j=0; out=[]
    while i < len(L) and j < len(R):
        if L[i] <= R[j]: out.append(L[i]); i+=1
        else: out.append(R[j]); j+=1
    out.extend(L[i:]); out.extend(R[j:])
    return out

def partition(a: List[int], lo: int, hi: int) -> int:
    pivot = a[hi]
    i = lo
    for j in range(lo, hi):
        if a[j] <= pivot:
            a[i], a[j] = a[j], a[i]
            i += 1
    a[i], a[hi] = a[hi], a[i]
    return i

def quicksort_inplace(a: List[int], lo: int=0, hi: int|None=None) -> None:
    if hi is None: hi = len(a)-1
    if lo < hi:
        p = partition(a, lo, hi)
        quicksort_inplace(a, lo, p-1)
        quicksort_inplace(a, p+1, hi)

def heapsort(a: List[int]) -> List[int]:
    h = list(a)
    heapq.heapify(h)
    out = [heapq.heappop(h) for _ in range(len(h))]
    return out

def binary_search(a: List[int], x: int) -> int:
    lo, hi = 0, len(a)-1
    while lo <= hi:
        mid = (lo+hi)//2
        if a[mid] == x: return mid
        if a[mid] < x: lo = mid+1
        else: hi = mid-1
    return -1

if __name__ == "__main__":
    data = [random.randint(0, 10**6) for _ in range(10_000)]
    for name, fn in [("mergesort", mergesort), ("heapsort", heapsort)]:
        t0=time.time(); s=fn(data); t1=time.time()
        assert s == sorted(data)
        print(name, "tiempo_s=", round(t1-t0, 4))
    arr = data[:]
    t0=time.time(); quicksort_inplace(arr); t1=time.time()
    assert arr == sorted(data)
    print("quicksort", "tiempo_s=", round(t1-t0, 4))
    idx = binary_search(arr, arr[len(arr)//2])
    print("binary_search_idx=", idx)
