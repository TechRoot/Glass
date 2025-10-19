#!/usr/bin/env python3
"""CLI para evaluar dominios y seleccionar técnicas mediante el protocolo.

Este script utiliza el módulo ``lib.protocol`` para calcular índices
globales en función de pesos definidos para diferentes criterios.  El
usuario puede especificar el dominio a evaluar (por ejemplo,
``optimization`` o ``graphs``) y opcionalmente un archivo de pesos en
formato YAML.  Por defecto se emplea el archivo ``etc/weights_default.yml``.

El comando ``evaluate`` muestra una tabla ordenada de técnicas con sus
índices.  Con ``--confirm`` y ``--output`` se escribe el resultado en
``data/traces/`` en formato CSV.  Un subcomando adicional ``list``
presenta los dominios disponibles.

Ejemplos de uso
---------------

Evaluar el dominio de grafos con pesos por defecto::

    python3 scripts/evaluate_cli.py evaluate --domain graphs --dry-run

Evaluar el dominio de optimización con un archivo de pesos personalizado y
guardar el resultado::

    python3 scripts/evaluate_cli.py evaluate --domain optimization \
      --weights my_weights.yml --output results.csv --confirm
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import List

from 20210412.lib import protocol  # type: ignore


def list_domains() -> List[str]:
    """Devuelve la lista de dominios disponibles en el protocolo."""
    return sorted(protocol.DOMAIN_TO_TECHNIQUES.keys())


def cmd_list(args: argparse.Namespace) -> None:
    """Muestra los dominios disponibles."""
    print("Dominios disponibles:")
    for d in list_domains():
        print(f" - {d}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evalúa un dominio concreto y muestra las técnicas recomendadas."""
    domain = args.domain.lower()
    if domain not in protocol.DOMAIN_TO_TECHNIQUES:
        logging.error("Dominio desconocido: %s", domain)
        return
    try:
        results = protocol.recommend_techniques(domain, args.weights)
    except Exception as exc:
        logging.error("Error al cargar pesos: %s", exc)
        return
    # Mostrar resultados por pantalla
    print(f"Recomendaciones para el dominio '{domain}':")
    for name, idx in results:
        print(f" - {name}: {idx:.4f}")
    # Guardar si se confirma y se especifica salida
    if args.confirm and args.output:
        out_path = Path(args.output)
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["technique", "index"])
                for name, idx in results:
                    writer.writerow([name, f"{idx:.6f}"])
            print(f"Resultados guardados en {out_path}")
        except Exception as exc:
            logging.error("No se pudo escribir el archivo de salida: %s", exc)
    elif args.confirm and not args.output:
        logging.warning("Se especificó --confirm sin --output; no se guardará ningún archivo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evalúa técnicas para un dominio utilizando un protocolo de ponderación",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Nivel de detalle del registro",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # Subcomando list
    parser_list = subparsers.add_parser(
        "list", help="Muestra los dominios disponibles en el protocolo"
    )
    parser_list.set_defaults(func=cmd_list)

    # Subcomando evaluate
    parser_eval = subparsers.add_parser(
        "evaluate", help="Evalúa un dominio y muestra las técnicas recomendadas"
    )
    parser_eval.add_argument(
        "--domain", required=True, help="Dominio a evaluar (por ejemplo, graphs, optimization)"
    )
    parser_eval.add_argument(
        "--weights", help="Ruta a un archivo YAML con pesos personalizados"
    )
    parser_eval.add_argument(
        "--output",
        help="Archivo CSV donde se guardarán los resultados si se confirma",
    )
    parser_eval.add_argument(
        "--confirm",
        action="store_true",
        default=False,
        help="Confirma y guarda los resultados en el archivo indicado",
    )
    parser_eval.set_defaults(func=cmd_evaluate)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    args.func(args)


if __name__ == "__main__":
    main()