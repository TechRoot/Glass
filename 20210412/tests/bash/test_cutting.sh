#!/bin/sh
# Test básico para cutting_heuristic.sh
set -eu

# Crear un archivo temporal con pedidos de corte
TMP_DIR=$(mktemp -d)
ORDERS="$TMP_DIR/orders.csv"
cat > "$ORDERS" <<'EOF_ORD'
4,2
6,2
5,5
EOF_ORD

# Ejecutar la heurística en modo dry-run con lámina 10x10
SCRIPT="$(dirname "$0")/../../lib/cutting_heuristic.sh"
output=$(sh "$SCRIPT" --input "$ORDERS" --board-size 10,10 --dry-run)
echo "$output" | grep -q "--dry-run activado" || { echo "No se detectó dry-run"; exit 1; }
# Comprobar que el script devuelve tres líneas de resultados
num_lines=$(echo "$output" | tail -n +2 | wc -l)
[ "$num_lines" -eq 3 ] || { echo "Número incorrecto de líneas: $num_lines"; exit 1; }
# Verificar los valores de recuento para cada pedido (board 10x10):
# 4x2 => floor(10/4)=2, floor(10/2)=5 => 10 piezas
# 6x2 => floor(10/6)=1, floor(10/2)=5 => 5 piezas
# 5x5 => floor(10/5)=2, floor(10/5)=2 => 4 piezas
counts=$(echo "$output" | tail -n +2 | cut -d, -f4 | tr '\n' ' ')
set -- $counts
[ "$1" -eq 10 ] || { echo "Esperado 10 piezas para pedido 1, obtenido $1"; exit 1; }
[ "$2" -eq 5 ] || { echo "Esperado 5 piezas para pedido 2, obtenido $2"; exit 1; }
[ "$3" -eq 4 ] || { echo "Esperado 4 piezas para pedido 3, obtenido $3"; exit 1; }

echo "Prueba de cutting_heuristic.sh completada con éxito"