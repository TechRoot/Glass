# Repositorio privado, Extracto para UTAMED
**Licencia privada — “Prohibido su uso”.**  Queda prohibida la copia, distribución o modificación no autorizadas.

---

## 1) Estructura (orientativa)
```
20210412/
  ├─ alg/          # Bibliotecas algoritmicas
  ├─ lib/          # Bibliotecas Python (lógica, lineal/geom., modular/códigos, grafos, verificación)
  ├─ scripts/      # CLIs de ejemplo (ayuda integrada con -h/--help)
  ├─ cpp/          # Código C++ (CMake) para utilidades/algoritmos
  ├─ data/
  │   ├─ samples/  # Datos de ejemplo (CSV/JSON) si se usan
  │   └─ traces/   # Salidas opcionales de ejecución (--confirm)
  └─ tests/        # Pruebas mínimas si están disponibles
```
> Nota: la estructura exacta puede variar..

---

## 2) Requisitos y entorno
- **Python** ≥ 3.10 (recomendado 3.11+).
- **CMake** ≥ 3.20 + compilador **C++17** (si compilas lo de `cpp/`).
- Dependencias Python: (mayor **fija**; minor/patch **acotados**).

**Entorno Python recomendado**
```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
python -m pip install -U pip
# Si existe:
python -m pip install -r requirements.txt
# o:
python -m pip install .
```

**Compilación C++ **
```bash
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
# Si hay tests:
ctest --test-dir build --output-on-failure
```

---

## 3) CLIs y bibliotecas
> Todos los CLIs soportan `-h/--help`. Muchos aceptan `--dry-run` para simular y `--confirm` para escribir salidas.

### 3.1 Lógica booleana (minimización)
- **Script:** `scripts/logic_cli.py`
- **Biblioteca base:** `lib/boolean_logic.py`
- **Qué hace:** minimización de funciones booleanas (p. ej., Quine–McCluskey) y utilidades de verificación.
- **Ejemplo**
  ```bash
  python -m scripts.logic_cli minimize       --minterms 3,5,6 --num-vars 3 --method quine_mccluskey --dry-run
  ```
- **Check rápido:** equivalencia con la tabla de verdad (total o muestreo grande).

### 3.2 Álgebra lineal y geometría (homografía, Kabsch, LS)
- **Script:** `scripts/calibration_cli.py`
- **Bibliotecas:** `lib/homography.py`, `lib/transformations.py`, `lib/least_squares.py`, `lib/kabsch.py`
- **Utilidad:** homografías 2D, alineamiento rígido (Kabsch) y ajustes por mínimos cuadrados (QR/SVD/ridge).
- **Ejemplos**
  ```bash
  python -m scripts.calibration_cli homography --input data/samples/homography_points.csv --dry-run
  python -m scripts.calibration_cli kabsch --input data/samples/kabsch_points.csv --dry-run
  ```
- **Check rápido:** reproyección < ε; RMSD/||A−(R·B+t)||_F/N < umbral; consistencia QR vs SVD.

### 3.3 Aritmética modular y códigos (CRC, LFSR, Reed–Solomon)
- **Bibliotecas:** `lib/crc.py`, `lib/lfsr.py`, `lib/reed_solomon.py`, `lib/modular.py`
- **Utilidad** CRC-8 (cálculo/verificación), generación con LFSR y codificación/verificación básica RS.
- **Ejemplo mínimo**
  ```python
  from 20210412.lib import crc, lfsr, reed_solomon
  val = crc.crc8(b"123456789")     # valor canónico 0xF4
  L = lfsr.LFSR(taps=[0,2], seed=[1,0,0]); seq = L.generate(7); per = L.period()
  msg = [1,2,3,4,5]; code = reed_solomon.rs_encode_msg(msg, nsym=2)
  assert reed_solomon.rs_check(code, 2)
  ```
- **Check rápido:** CRC("123456789") = 0xF4; período LFSR según taps/semilla; RS válida y decodifica si el error ≤ capacidad.

### 3.4 Grafos (Laplaciana, Fiedler, Dijkstra, orden topológico)
- **Script:** `scripts/graphs_cli.py` (o subcomandos equivalentes)
- **Biblioteca:** `lib/graph_algorithms.py`
- **Utilidad:** Laplaciana y valor de Fiedler; Dijkstra; orden topológico; centralidad por eigenvector.
- **Ejemplos**
  ```bash
  python -m scripts.graphs_cli laplacian --input data/samples/graph.json --dry-run
  python -m scripts.graphs_cli dijkstra --input data/samples/graph.json --source A --target Z --dry-run
  python -m scripts.graphs_cli topo --input data/samples/dag.json --dry-run
  ```
- **Check rápido:** topológico válido; coste Dijkstra consistente; Fiedler>0 si conexo, 0 si disconexo.

### 3.5 Utilidades C++ (Dijkstra / álgebra numérica)
- **Ubicación:** `cpp/` (CMake)
- **Marco** ejecutables de prueba para Dijkstra y pseudoinversa/condición.
- **Ejemplo**
  ```bash
  cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build --target test_dijkstra
  ./build/test_dijkstra
  ```
- **Check rápido:** ruta/coste esperados; ||A·A⁺·A − A||_F pequeño.

---

## 4) Datos de ejemplo y formatos
- **CSV**: coma, cabeceras (`x,y` / `x,y,z` / `src,dst,weight`).
- **JSON**: para grafos, `{"nodes":[...],"edges":[{"u":"A","v":"B","w":1.2}]}`.
- Si `data/samples/` no existe, cree mini-datasets con esos formatos para reproducir los ejemplos.

---

## 5) Verificación rápida
- **Lógica**: igualdad de tabla de verdad entre función original y minimizada.
- **Geom./Lineal**: reproyección (homografía) y RMSD (Kabsch) por debajo de ε; QR/SVD consistentes.
- **Modular/Códigos**: CRC “123456789”=0xF4; RS valida y decodifica dentro de capacidad; inverso modular cumple `a*x ≡ 1 (mod m)`.
- **Grafos**: orden topológico válido; coste y ruta de Dijkstra correctos; Fiedler coherente con conectividad.
- **C++**: binarios compilan en Release; tests reportan resultados esperados.

---

## 6) Buenas prácticas locales
- Use `--dry-run` para explorar y `--confirm` para persistir en `data/traces/`.
- Mantenga dependencias con mayor **fija** y minor/patch **acotados**.

---

## 7) Licencia
**Licencia privada — “Prohibido su uso”.**  Queda prohibida la copia, distribución o modificación no autorizadas.
