#!/bin/bash
# Heurística simple para corte bidimensional
#
# Este script lee un archivo CSV con pedidos de corte (ancho,alto) y
# calcula cuántas piezas de cada pedido caben en una lámina rectangular
# de dimensiones suministradas por el usuario (por defecto 100x100). La
# heurística se limita a calcular el número máximo de piezas idénticas
# que encajan en la lámina, sin combinar diferentes pedidos en un mismo
# patrón. También calcula el área de desperdicio. Los resultados se
# muestran por pantalla y, si se especifica `--confirm`, se guardan en
# un CSV en `data/traces/block5_trace.csv`.

# Activar opciones de salida inmediata ante errores y variables no definidas.
# 'pipefail' no está disponible en /bin/sh, por lo que no se usa aquí.
set -eu

usage() {
  cat <<'USAGE'
Uso: cutting_heuristic.sh --input archivo.csv [--board-size W,H] [--output fichero.csv] [--dry-run] [--confirm]

  --input        Ruta al fichero CSV de pedidos. Cada línea debe contener 'ancho,alto'.
  --board-size   Dimensiones de la lámina en formato 'ancho,alto'. Por defecto 100,100.
  --output       Fichero donde guardar el resultado. Si no se especifica, se imprime en pantalla.
  --dry-run      No escribe salidas; muestra lo que haría. Activado por defecto.
  --confirm      Ejecuta la operación y guarda el resultado en el fichero indicado o en data/traces.
USAGE
}

INPUT=""
BOARD_SIZE="100,100"
OUTPUT=""
DRY_RUN=1

while [ $# -gt 0 ]; do
  case "$1" in
    --input)
      INPUT="$2"; shift 2;;
    --board-size)
      BOARD_SIZE="$2"; shift 2;;
    --output)
      OUTPUT="$2"; shift 2;;
    --confirm)
      DRY_RUN=0; shift;;
    --dry-run)
      DRY_RUN=1; shift;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Opción desconocida: $1" >&2; usage; exit 1;;
  esac
done

if [ -z "$INPUT" ]; then
  echo "Se requiere --input" >&2
  usage
  exit 1
fi

WIDTH=$(echo "$BOARD_SIZE" | cut -d, -f1)
HEIGHT=$(echo "$BOARD_SIZE" | cut -d, -f2)
if ! echo "$WIDTH" | grep -Eq '^[0-9]+(\.[0-9]+)?$' || ! echo "$HEIGHT" | grep -Eq '^[0-9]+(\.[0-9]+)?$'; then
  echo "Las dimensiones de la lámina deben ser numéricas" >&2
  exit 1
fi

# Función para calcular corte para una pieza
calc_piece() {
  local w="$1" h="$2"
  if [ "$w" = 0 ] || [ "$h" = 0 ]; then
    echo "0,0"
    return
  fi
  local per_row per_col count waste
  # Calcular cuántas piezas caben por fila y por columna usando awk para
  # evitar la dependencia de la utilidad `bc`. Se trunca la división hacia
  # abajo para obtener la cantidad entera de piezas.
  per_row=$(awk -v W="$WIDTH" -v w="$w" 'BEGIN { if (w == 0) print 0; else print int(W / w) }')
  per_col=$(awk -v H="$HEIGHT" -v h="$h" 'BEGIN { if (h == 0) print 0; else print int(H / h) }')
  count=$((per_row * per_col))
  # Calcular área de desperdicio con awk: área de la lámina menos el área ocupada por las piezas
  waste=$(awk -v W="$WIDTH" -v H="$HEIGHT" -v c="$count" -v w="$w" -v h="$h" 'BEGIN { printf "%.6f", (W*H - c*w*h) }')
  echo "$count,$waste"
}

# Preparar salida
RESULTS=""
LINE=0
while IFS=, read -r w h; do
  LINE=$((LINE + 1))
  # saltar líneas vacías o encabezados no numéricos
  if [ -z "$w" ] || [ -z "$h" ]; then
    continue
  fi
  if ! echo "$w" | grep -Eq '^[0-9]+(\.[0-9]+)?$' || ! echo "$h" | grep -Eq '^[0-9]+(\.[0-9]+)?$'; then
    continue
  fi
  res=$(calc_piece "$w" "$h")
  pieces=$(echo "$res" | cut -d, -f1)
  waste=$(echo "$res" | cut -d, -f2)
  RESULTS="$RESULTS$LINE,$w,$h,$pieces,$waste\n"
done < "$INPUT"

if [ "$DRY_RUN" -eq 1 ]; then
    # Indicar que se está en modo simulación sin comenzar la línea con '--' para evitar conflictos con utilidades como grep.
    echo "dry-run activado. Resultados que se generarían (fila,ancho,alto,cantidad,desperdicio):"
  printf "%b" "$RESULTS"
else
  # determinar ruta de salida
  if [ -z "$OUTPUT" ]; then
    mkdir -p "$(dirname "$0")/../data/traces" 2>/dev/null || true
    OUTPUT="$(dirname "$0")/../data/traces/block5_trace.csv"
  fi
  printf "%b" "$RESULTS" > "$OUTPUT"
  echo "Resultados guardados en $OUTPUT"
fi