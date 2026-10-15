"""
Punto de Entrada Unico (CLI Global y Runner de Ejercicios) para la Entrega 1.
Curso: Redes Neuronales y Algoritmos Bioinspirados - UNAL Medellin

Permite ejecutar TODO o CUALQUIER ejercicio/algoritmo en UN SOLO COMANDO:

COMANDOS GENERALES:
  python main.py                           # Ejecucion completa (Parte 1 y Parte 2)
  python main.py --demo                    # Demostracion rapida en vivo (< 30 s)
  python main.py --menu                    # Menu interactivo por consola

EJECUCION DE EJERCICIOS INDIVIDUALES:
  python main.py -e gd                     # Descenso por Gradiente (Rosenbrock y Rastrigin)
  python main.py -e pso                    # Enjambre de Particulas (PSO)
  python main.py -e ea                     # Algoritmo Evolutivo Continuo (EA)
  python main.py -e de                     # Evolucion Diferencial (DE)
  python main.py -e aco                    # Colonia de Hormigas con Or-opt (TSP 47 capitales)
  python main.py -e ga                     # Algoritmo Genetico DEAP (TSP 47 capitales)
  python main.py -e oropt                  # Comparativa Con vs Sin Or-opt en ACO
  python main.py -e sensibilidad           # Estudio de sensibilidad valor de la hora [0, 100] €/h
  python main.py -e graficas               # Genera todas las figuras (3D, barras, convergencia)
  python main.py -e gifs                   # Genera las animaciones GIF de Parte 1 y Parte 2

MODULAR:
  python main.py --parte 1                 # Solo Parte 1 (Optimizacion Numerica)
  python main.py --parte 2                 # Solo Parte 2 (Optimizacion Combinatoria TSP)
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
    """Ejecuta un comando en un subproceso con el interprete de Python actual."""
    print(f"\n[EJECUTANDO] python {' '.join(comando)}  (en {cwd.name}/)")
    proceso = subprocess.run([sys.executable] + comando, cwd=str(cwd))
    return proceso.returncode


def mostrar_menu_interactivo():
    """Menu interactivo por consola para seleccionar y correr cualquier ejercicio."""
    while True:
        print("\n" + "=" * 78)
        print("    PROYECTO REDES NEURONALES - ENTREGA 1 (UNAL MEDELLIN)")
        print("    MENU INTERACTIVO DE EJECUCION EN UN SOLO COMANDO")
        print("=" * 78)
        print("  [1]  Ejecutar TODO el Proyecto (Parte 1 + Parte 2 completa)")
        print("  [2]  Demostracion Rapida en Vivo (< 30 s) para Sustentacion")
        print("  --- PARTE 1: OPTIMIZACION NUMERICA CONTINUA ---")
        print("  [3]  Descenso por Gradiente (GD) - Rosenbrock y Rastrigin")
        print("  [4]  Enjambre de Particulas (PSO)")
        print("  [5]  Algoritmo Evolutivo Continuo (EA)")
        print("  [6]  Evolucion Diferencial (DE)")
        print("  [7]  Protocolo Comparativo Completo Parte 1 (30 corridas)")
        print("  --- PARTE 2: OPTIMIZACION COMBINATORIA (TSP 47 CAPITALES) ---")
        print("  [8]  Colonia de Hormigas (ACO + Or-opt) - Ruta optima peninsular")
        print("  [9]  Algoritmo Genetico (DEAP) para el TSP")
        print("  [10] Comparativa Con vs Sin Or-opt en ACO")
        print("  [11] Estudio de Sensibilidad: Valor de la Hora del Vendedor [0 - 100 EUR/h]")
        print("  --- RECURSOS GRAFICOS Y VISUALIZACIONES ---")
        print("  [12] Generar Figuras 3D, Graficas de Barras y Convergencia")
        print("  [13] Generar Animaciones GIF de Optimizacion")
        print("  [0]  Salir")
        print("=" * 78)

        opc = input("Seleccione una opcion (0-13): ").strip()
        if opc == "0":
            print("\nFinalizando. ¡Hasta luego!")
            sys.exit(0)
        elif opc == "1":
            correr_ejercicio("todo")
        elif opc == "2":
            correr_ejercicio("demo")
        elif opc == "3":
            correr_ejercicio("gd")
        elif opc == "4":
            correr_ejercicio("pso")
        elif opc == "5":
            correr_ejercicio("ea")
        elif opc == "6":
            correr_ejercicio("de")
        elif opc == "7":
            correr_ejercicio("parte1")
        elif opc == "8":
            correr_ejercicio("aco")
        elif opc == "9":
            correr_ejercicio("ga")
        elif opc == "10":
            correr_ejercicio("oropt")
        elif opc == "11":
            correr_ejercicio("sensibilidad")
        elif opc == "12":
            correr_ejercicio("graficas")
        elif opc == "13":
            correr_ejercicio("gifs")
        else:
            print("Opcion no valida. Ingrese un numero entre 0 y 13.")


def correr_ejercicio(ejercicio: str, demo: bool = False, corridas: int | None = None,
                     funcion: str = "ambas", dim: int = 0):
    ej = ejercicio.lower()
    t_inicio = time.time()

    if ej in ("todo", "todas", "ambas"):
        cmd_p1 = ["main.py"]
        if demo:
            cmd_p1.append("--demo")
        if corridas:
            cmd_p1.extend(["--corridas", str(corridas)])
        correr_comando(cmd_p1, PARTE1_DIR)

        cmd_p2 = ["main.py"]
        if demo:
            cmd_p2.append("--demo")
        correr_comando(cmd_p2, PARTE2_DIR)

        correr_comando(["generar_graficas.py"], BASE_DIR)

    elif ej == "demo":
        correr_comando(["main.py", "--demo"], PARTE1_DIR)
        correr_comando(["main.py", "--demo"], PARTE2_DIR)

    elif ej in ("gd", "gradiente"):
        cmd = ["main.py", "--metodo", "gd", "--funcion", funcion, "--dim", str(dim)]
        if demo:
            cmd.append("--demo")
        if corridas:
            cmd.extend(["--corridas", str(corridas)])
        correr_comando(cmd, PARTE1_DIR)

    elif ej == "pso":
        cmd = ["main.py", "--metodo", "pso", "--funcion", funcion, "--dim", str(dim)]
        if demo:
            cmd.append("--demo")
        if corridas:
            cmd.extend(["--corridas", str(corridas)])
        correr_comando(cmd, PARTE1_DIR)

    elif ej in ("ea", "evolutivo"):
        cmd = ["main.py", "--metodo", "ea", "--funcion", funcion, "--dim", str(dim)]
        if demo:
            cmd.append("--demo")
        if corridas:
            cmd.extend(["--corridas", str(corridas)])
        correr_comando(cmd, PARTE1_DIR)

    elif ej in ("de", "diferencial"):
        cmd = ["main.py", "--metodo", "de", "--funcion", funcion, "--dim", str(dim)]
        if demo:
            cmd.append("--demo")
        if corridas:
            cmd.extend(["--corridas", str(corridas)])
        correr_comando(cmd, PARTE1_DIR)

    elif ej in ("parte1", "numerica", "1"):
        cmd = ["main.py"]
        if demo:
            cmd.append("--demo")
        if corridas:
            cmd.extend(["--corridas", str(corridas)])
        correr_comando(cmd, PARTE1_DIR)

    elif ej == "aco":
        cmd = ["main.py", "--algoritmo", "aco"]
        if demo:
            cmd.append("--demo")
        correr_comando(cmd, PARTE2_DIR)

    elif ej in ("ga", "genetico", "deap"):
        cmd = ["main.py", "--algoritmo", "ga"]
        if demo:
            cmd.append("--demo")
        correr_comando(cmd, PARTE2_DIR)

    elif ej in ("oropt", "or_opt", "comparar_oropt"):
        cmd = ["main.py", "--comparar-oropt"]
        if demo:
            cmd.append("--demo")
        correr_comando(cmd, PARTE2_DIR)

    elif ej in ("sensibilidad", "valor_hora"):
        cmd = ["sensibilidad.py"]
        if demo:
            cmd.append("--demo")
        correr_comando(cmd, PARTE2_DIR)

    elif ej in ("parte2", "combinatoria", "tsp", "2"):
        cmd = ["main.py"]
        if demo:
            cmd.append("--demo")
        correr_comando(cmd, PARTE2_DIR)

    elif ej in ("graficas", "figuras", "plots"):
        correr_comando(["generar_graficas.py"], BASE_DIR)

    elif ej in ("gifs", "animaciones"):
        correr_comando(["main.py", "--gif", "--demo"], PARTE1_DIR)
        correr_comando(["main.py", "--demo"], PARTE2_DIR)
        correr_comando(["generar_graficas.py"], BASE_DIR)

    else:
        print(f"[ERROR] Ejercicio no reconocido: '{ejercicio}'")
        print("Opciones validas: gd, pso, ea, de, aco, ga, oropt, sensibilidad, graficas, gifs, parte1, parte2, todo, demo")
        sys.exit(1)

    dt = time.time() - t_inicio
    print(f"\n[OK] Ejercicio '{ejercicio}' finalizado en {dt:.2f} segundos.")


def main():
    parser = argparse.ArgumentParser(
        description="CLI Maestro: Ejecuta cualquier ejercicio o parte del proyecto en un solo comando."
    )
    parser.add_argument("-e", "--ejercicio", type=str, default=None,
                        help="Ejercicio especifico: gd, pso, ea, de, aco, ga, oropt, sensibilidad, graficas, gifs, parte1, parte2, todo")
    parser.add_argument("--menu", "-i", action="store_true",
                        help="Abre el menu interactivo por consola")
    parser.add_argument("--parte", choices=["1", "2", "ambas"], default=None,
                        help="Parte a ejecutar (1: Numerica, 2: Combinatoria, ambas por defecto)")
    parser.add_argument("--demo", action="store_true",
                        help="Perfil rapido para sustentacion en vivo (<30 s)")
    parser.add_argument("--gif", action="store_true",
                        help="Genera animaciones GIF")
    parser.add_argument("--algoritmo", type=str, default=None,
                        help="Algoritmo especifico para Parte 2 (aco, ga, ambos)")
    parser.add_argument("--comparar-oropt", action="store_true",
                        help="Ejecuta comparativa Con vs Sin Or-opt en Parte 2")
    parser.add_argument("--sensibilidad", action="store_true",
                        help="Ejecuta estudio de sensibilidad del valor de hora")
    parser.add_argument("--barrido", action="store_true",
                        help="Ejecuta barrido completo en Parte 2")
    parser.add_argument("--corridas", type=int, default=None,
                        help="Numero de corridas independientes en Parte 1")
    parser.add_argument("--funcion", choices=["Rosenbrock", "Rastrigin", "ambas"], default="ambas",
                        help="Funcion para Parte 1")
    parser.add_argument("--dim", choices=[2, 3, 0], type=int, default=0,
                        help="Dimension para Parte 1 (2, 3 o 0 para ambas)")
    args = parser.parse_args()

    if args.menu:
        mostrar_menu_interactivo()
        return

    # Si se pasa --ejercicio especifico
    if args.ejercicio:
        correr_ejercicio(args.ejercicio, demo=args.demo, corridas=args.corridas,
                         funcion=args.funcion, dim=args.dim)
        return

    # Compatibilidad con flags clasicos
    if args.sensibilidad:
        correr_ejercicio("sensibilidad", demo=args.demo)
        return

    if args.comparar_oropt:
        correr_ejercicio("oropt", demo=args.demo)
        return

    if args.parte == "1":
        correr_ejercicio("parte1", demo=args.demo, corridas=args.corridas)
        return
    elif args.parte == "2":
        if args.algoritmo:
            correr_ejercicio(args.algoritmo, demo=args.demo)
        else:
            correr_ejercicio("parte2", demo=args.demo)
        return

    # Si no se paso ningun argumento, o se especifico --parte ambas
    t_inicio = time.time()
    print("=" * 78)
    print("PROYECTO REDES Y ALGORITMOS BIOINSPIRADOS - ENTREGA 1 (UNAL MEDELLIN)")
    print(f"Modo: {'DEMOSTRACION EN VIVO (<30 s)' if args.demo else 'COMPLETO'}")
    print("Ejecutando Parte 1 (Numerica) + Parte 2 (Combinatoria TSP)...")
    print("=" * 78)

    # 1. PARTE 1
    cmd_p1 = ["main.py"]
    if args.demo:
        cmd_p1.append("--demo")
    if args.gif:
        cmd_p1.append("--gif")
    if args.corridas:
        cmd_p1.extend(["--corridas", str(args.corridas)])
    correr_comando(cmd_p1, PARTE1_DIR)

    # 2. PARTE 2
    cmd_p2 = ["main.py"]
    if args.demo:
        cmd_p2.append("--demo")
    if args.algoritmo:
        cmd_p2.extend(["--algoritmo", args.algoritmo])
    if args.barrido:
        cmd_p2.append("--barrido")
    correr_comando(cmd_p2, PARTE2_DIR)

    dt_total = time.time() - t_inicio
    print("\n" + "=" * 78)
    print(f"[OK] EJECUCION INTEGRAL FINALIZADA EN {dt_total:.2f} segundos.")
    print("Resultados disponibles en:")
    print(f"  - Parte 1 : {PARTE1_DIR / 'resultados'}")
    print(f"  - Parte 2 : {PARTE2_DIR / 'resultados'}")
    print(f"  - Assets  : {BASE_DIR / 'assets' / 'img'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
