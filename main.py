"""
Punto de entrada unico (CLI Global) para la Entrega 1.

Permite ejecutar:
  python main.py                    # Ejecucion integral (Parte 1 y Parte 2)
  python main.py --demo             # Demostracion rapida en vivo (< 30 segundos)
  python main.py --parte 1          # Ejecuta unicamente Parte 1 (Optimizacion Numerica)
  python main.py --parte 2          # Ejecuta unicamente Parte 2 (Optimizacion Combinatoria TSP)
  python main.py --parte 2 --algoritmo ga    # Ejecuta el GA en Parte 2
  python main.py --parte 2 --comparar-oropt # Comparativa Con vs Sin Or-opt en Parte 2
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PARTE1_DIR = BASE_DIR / "parte_1"
PARTE2_DIR = BASE_DIR / "parte_2"


def correr_comando(comando: list[str], cwd: Path) -> int:
    print(f"\n[EJECUTANDO] {' '.join(comando)} en {cwd.name}/")
    proceso = subprocess.run([sys.executable] + comando, cwd=str(cwd))
    return proceso.returncode


def main():
    parser = argparse.ArgumentParser(
        description="CLI Global del Proyecto: Parte 1 (Numerica) y Parte 2 (Combinatoria)"
    )
    parser.add_argument("--parte", choices=["1", "2", "ambas"], default="ambas",
                        help="Parte a ejecutar (1: Numerica, 2: Combinatoria, ambas por defecto)")
    parser.add_argument("--demo", action="store_true",
                        help="Perfil rapido para sustentacion en vivo (<30 s)")
    parser.add_argument("--gif", action="store_true",
                        help="Genera animaciones GIF")
    parser.add_argument("--algoritmo", type=str, default=None,
                        help="Algoritmo especifico para Parte 2 (aco, ga, ambos)")
    parser.add_argument("--comparar-oropt", action="store_true",
                        help="Ejecuta comparativa Con vs Sin Or-opt en Parte 2")
    parser.add_argument("--barrido", action="store_true",
                        help="Ejecuta barrido completo en Parte 2")
    parser.add_argument("--corridas", type=int, default=None,
                        help="Numero de corridas independientes en Parte 1")
    args = parser.parse_args()

    t_inicio = time.time()
    print("=" * 75)
    print("PROYECTO REDES Y ALGORITMOS BIOINSPIRADOS - ENTREGA 1")
    print(f"Modo: {'DEMOSTRACION EN VIVO (<30 s)' if args.demo else 'COMPLETO'}")
    print(f"Parte seleccionada: {args.parte}")
    print("=" * 75)

    # 1. PARTE 1
    if args.parte in ("1", "ambas"):
        cmd_p1 = ["main.py"]
        if args.demo:
            cmd_p1.append("--demo")
        if args.gif:
            cmd_p1.append("--gif")
        if args.corridas:
            cmd_p1.extend(["--corridas", str(args.corridas)])

        codigo_p1 = correr_comando(cmd_p1, PARTE1_DIR)
        if codigo_p1 != 0:
            print(f"[ERROR] Parte 1 finalizo con codigo {codigo_p1}")

    # 2. PARTE 2
    if args.parte in ("2", "ambas"):
        cmd_p2 = ["main.py"]
        if args.demo:
            cmd_p2.append("--demo")
        if args.algoritmo:
            cmd_p2.extend(["--algoritmo", args.algoritmo])
        if args.comparar_oropt:
            cmd_p2.append("--comparar-oropt")
        if args.barrido:
            cmd_p2.append("--barrido")

        codigo_p2 = correr_comando(cmd_p2, PARTE2_DIR)
        if codigo_p2 != 0:
            print(f"[ERROR] Parte 2 finalizo con codigo {codigo_p2}")

    dt_total = time.time() - t_inicio
    print("\n" + "=" * 75)
    print(f"[OK] EJECUCION FINALIZADA EN {dt_total:.2f} segundos.")
    print("Resultados disponibles en:")
    print(f"  - Parte 1 : {PARTE1_DIR / 'resultados'}")
    print(f"  - Parte 2 : {PARTE2_DIR / 'resultados'}")
    print("=" * 75)


if __name__ == "__main__":
    main()
