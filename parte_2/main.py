"""
Runner unificado para la Parte 2: Optimizacion Combinatoria (TSP 47 Capitales).

Permite ejecutar:
  python main.py                     # ACO con Or-opt (por defecto) + GIF + Mapa
  python main.py --sin-or-opt        # ACO puro sin Or-opt
  python main.py --algoritmo ga      # Algoritmo Genetico (DEAP) para el TSP
  python main.py --comparar-oropt    # Comparativa Con vs Sin Or-opt
  python main.py --barrido           # Barrido completo de hiperparametros ACO
  python main.py --demo              # Corrida demo rapida (<15 s)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Asegurar importacion local de parte_2
DIR_PARTE2 = Path(__file__).resolve().parent
if str(DIR_PARTE2) not in sys.path:
    sys.path.insert(0, str(DIR_PARTE2))

from config import cargar_config
from costes_comunes import BASE_DIR, VALOR_HORA_POR_DEFECTO, cargar_costes
from colonia_hormigas import ejecutar_aco, guardar_resultados
from algoritmo_genetico import ejecutar_ga

DIR_RESULTADOS = DIR_PARTE2 / "resultados"
DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)


def correr_aco_individual(datos, M, cfg, vh: float, usar_or_opt: bool, hacer_gif: bool):
    aco = cfg.get("aco", {})
    t0 = time.time()
    sufijo = "con_oropt" if usar_or_opt else "sin_oropt"

    print(f"\n[ACO {'CON' if usar_or_opt else 'SIN'} Or-opt] "
          f"Ejecutando ({aco.get('hormigas', 20)} hormigas, "
          f"{aco.get('iteraciones', 150)} iteraciones, semilla={aco.get('semilla', 42)})...")

    res = ejecutar_aco(
        M,
        hormigas=aco.get("hormigas", 20),
        iteraciones=aco.get("iteraciones", 150),
        alfa=aco.get("alfa", 1.0),
        beta=aco.get("beta", 3.0),
        evaporacion=aco.get("evaporacion", 0.4),
        semilla=aco.get("semilla", 42),
        usar_or_opt=usar_or_opt,
        verbose=True,
    )
    ruta_csv, resumen = guardar_resultados(datos, res["tour"], vh, res, directorio=DIR_RESULTADOS)
    dt = time.time() - t0
    print(f"[ACO {sufijo}] Coste: {res['coste']:.2f} EUR en {dt:.2f}s")
    print(f"[ACO {sufijo}] Desglose: Distancia={resumen['distancia_total_km']:.1f} km, "
          f"Tiempo={resumen['tiempo_total_h']:.2f} h, Peajes={resumen['peajes_total_eur']:.2f} EUR, "
          f"Gasolina={resumen['combustible_total_eur']:.2f} EUR")

    if hacer_gif:
        from generar_gif import construir_animacion
        from matplotlib.animation import PillowWriter
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        gif_c = cfg.get("gif", {})
        cada = gif_c.get("cada", 2)
        fps = gif_c.get("fps", 8)

        print(f"[ACO {sufijo}] Generando animacion GIF y mapa...")
        fig, anim, reg, _ = construir_animacion(
            datos, vh, algoritmo="aco",
            semilla=aco.get("semilla", 42),
            cada=cada, fps=fps,
            hormigas=aco.get("hormigas", 20),
            iteraciones=aco.get("iteraciones", 150),
            alfa=aco.get("alfa", 1.0),
            beta=aco.get("beta", 3.0),
            evaporacion=aco.get("evaporacion", 0.4),
            usar_or_opt=usar_or_opt,
        )
        out_gif = DIR_RESULTADOS / f"evolucion_aco_{sufijo}.gif"
        out_png = DIR_RESULTADOS / f"mapa_aco_{sufijo}.png"
        anim.save(str(out_gif), writer=PillowWriter(fps=fps), dpi=110)
        fig.savefig(out_png, dpi=130)
        plt.close(fig)
        print(f"[ACO {sufijo}] GIF: {out_gif}")
        print(f"[ACO {sufijo}] Mapa: {out_png}")

    return resumen


def correr_ga_individual(datos, M, cfg, vh: float, hacer_gif: bool = True):
    ga = cfg.get("ga", {})
    t0 = time.time()
    pob = ga.get("poblacion", 100)
    gen = ga.get("generaciones", 500)
    semilla = ga.get("semilla", 42)

    print(f"\n[GA - Algoritmo Genetico] Iniciando (pob={pob}, gen={gen}, semilla={semilla})...")
    res = ejecutar_ga(
        M,
        tam_poblacion=pob,
        generaciones=gen,
        prob_cruce=ga.get("probabilidad_cruce", 0.8),
        prob_mutacion=ga.get("probabilidad_mutacion", 0.2),
        semilla=semilla,
        verbose=True,
    )
    desglose = datos.desglose_tour(res["tour"], vh)
    dt = time.time() - t0
    print(f"[GA] Finalizado en {dt:.2f}s | Coste: {desglose['coste_eur']:.2f} EUR | Distancia: {desglose['distancia_km']:.1f} km")

    resumen_ga = {
        "algoritmo": "ga",
        "valor_hora_eur": vh,
        "ciudades": datos.n,
        "parametros": ga,
        "coste_total_eur": round(desglose["coste_eur"], 2),
        "distancia_total_km": round(desglose["distancia_km"], 1),
        "tiempo_total_h": round(desglose["tiempo_h"], 2),
        "tiempo_ejecucion_s": round(dt, 2),
        "historia_mejor": res["historia_mejor"],
        "orden_ciudades": datos.tour_a_nombres(res["tour"]),
    }
    with open(DIR_RESULTADOS / "resumen_ga.json", "w", encoding="utf-8") as f:
        json.dump(resumen_ga, f, ensure_ascii=False, indent=2)

    if hacer_gif:
        from generar_gif import construir_animacion
        from matplotlib.animation import PillowWriter
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        gif_c = cfg.get("gif", {})
        cada = max(gif_c.get("cada", 2), gen // 40)
        fps = gif_c.get("fps", 8)

        print(f"[GA] Generando animacion GIF y mapa...")
        fig, anim, reg, _ = construir_animacion(
            datos, vh, algoritmo="ga",
            semilla=semilla,
            cada=cada, fps=fps,
            poblacion=pob,
            generaciones=gen,
            prob_cruce=ga.get("probabilidad_cruce", 0.8),
            prob_mutacion=ga.get("probabilidad_mutacion", 0.2),
        )
        out_gif = DIR_RESULTADOS / "evolucion_ga.gif"
        out_png = DIR_RESULTADOS / "mapa_ga.png"
        anim.save(str(out_gif), writer=PillowWriter(fps=fps), dpi=110)
        fig.savefig(out_png, dpi=130)
        plt.close(fig)
        print(f"[GA] GIF: {out_gif}")
        print(f"[GA] Mapa: {out_png}")

    return resumen_ga


def generar_grafica_comparativa_parte2(res_aco, res_ga, vh: float, salida: Path):
    """Genera grafica comparativa de convergencia entre ACO y GA."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Historia ACO
    aco_hist = res_aco.get("historia_mejor", [])
    if aco_hist:
        ax.plot(range(1, len(aco_hist) + 1), aco_hist,
                label=f"ACO (con Or-opt) - Final: {res_aco['coste_total_eur']:.2f} EUR",
                color="#c0392b", linewidth=2.5)

    # Historia GA
    ga_hist = res_ga.get("historia_mejor", [])
    if ga_hist:
        ax.plot(range(1, len(ga_hist) + 1), ga_hist,
                label=f"Algoritmo Genético (DEAP) - Final: {res_ga['coste_total_eur']:.2f} EUR",
                color="#2980b9", linewidth=2.2, linestyle="--")

    ax.set_title(f"Convergencia Comparativa: ACO vs Algoritmo Genético (TSP 47 Capitales)\n"
                 f"Valor hora = {vh:.2f} EUR/h | Semilla = 42", fontsize=12, fontweight="bold")
    ax.set_xlabel("Iteraciones / Generaciones", fontsize=11)
    ax.set_ylabel("Costo del Tour (EUR)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=10, loc="upper right")
    fig.tight_layout()
    fig.savefig(salida, dpi=130)
    plt.close(fig)
    print(f"[Parte 2] Grafica comparativa guardada: {salida}")


def main():
    parser = argparse.ArgumentParser(description="Runner Parte 2: Optimizacion Combinatoria (TSP 47 Capitales)")
    parser.add_argument("--demo", action="store_true", help="Perfil rapido (<15 s)")
    parser.add_argument("--algoritmo", choices=["aco", "ga", "ambos"], default="ambos",
                        help="Algoritmo a ejecutar (por defecto: ambos)")
    parser.add_argument("--valor-hora", type=float, default=None, help="Valor de la hora del vendedor (EUR/h)")
    parser.add_argument("--sin-or-opt", action="store_true", help="Desactiva Or-opt en ACO")
    parser.add_argument("--comparar-oropt", action="store_true", help="Ejecuta comparativa Con vs Sin Or-opt")
    parser.add_argument("--barrido", action="store_true", help="Barrido completo de parametros ACO")
    parser.add_argument("--sin-gif", action="store_true", help="Omite generacion de GIF")
    args = parser.parse_args()

    cfg = cargar_config(parte="parte_2", demo=args.demo)
    vh = args.valor_hora if args.valor_hora is not None else cfg.get("valor_hora", VALOR_HORA_POR_DEFECTO)

    datos = cargar_costes()
    M = datos.matriz_coste(vh)

    print("=" * 70)
    print("PARTE 2: OPTIMIZACION COMBINATORIA - TSP 47 CAPITALES PENINSULARES")
    print(f"Valor hora: {vh:.2f} EUR/h | Semilla: {cfg.get('semilla', 42)} | Demo: {args.demo}")
    print("=" * 70)

    t0 = time.time()

    if args.barrido:
        from experimentos_aco import main as ejecutar_barrido_aco
        ejecutar_barrido_aco()
        return

    if args.comparar_oropt:
        from experimentos_aco import comparar_con_y_sin_oropt
        comparar_con_y_sin_oropt(M, datos, vh, cfg.get("aco", {}))
        print(f"\nComparativa guardada en {DIR_RESULTADOS} ({time.time()-t0:.2f}s)")
        return

    resultados = {}
    if args.algoritmo in ("aco", "ambos"):
        resultados["aco"] = correr_aco_individual(datos, M, cfg, vh, usar_or_opt=not args.sin_or_opt, hacer_gif=not args.sin_gif)

    if args.algoritmo in ("ga", "ambos"):
        resultados["ga"] = correr_ga_individual(datos, M, cfg, vh, hacer_gif=not args.sin_gif)

    if "aco" in resultados and "ga" in resultados:
        generar_grafica_comparativa_parte2(resultados["aco"], resultados["ga"], vh,
                                           DIR_RESULTADOS / "comparacion_convergencia.png")

    print(f"\nTOTAL PARTE 2: {time.time()-t0:.2f}s | Resultados en {DIR_RESULTADOS}")


if __name__ == "__main__":
    main()
