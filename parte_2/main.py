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
import shutil
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
ASSETS_DIR = BASE_DIR.parent / "assets" / "img"


def _sincronizar_asset(ruta_origen: Path) -> None:
    """Copia un recurso generado en parte_2/resultados/ hacia assets/img/."""
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    destino = ASSETS_DIR / ruta_origen.name
    shutil.copy2(ruta_origen, destino)
    print(f"[Assets] Sincronizado: {destino}")


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
        if usar_or_opt:
            _sincronizar_asset(out_gif)
            _sincronizar_asset(out_png)

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
    print(f"[GA] Desglose: Tiempo={desglose['tiempo_h']:.2f} h, "
          f"Peajes={desglose['tax_eur']:.2f} EUR, Gasolina={desglose['gas_eur']:.2f} EUR")

    resumen_ga = {
        "algoritmo": "ga",
        "valor_hora_eur": vh,
        "ciudades": datos.n,
        "parametros": ga,
        "coste_total_eur": round(desglose["coste_eur"], 2),
        "distancia_total_km": round(desglose["distancia_km"], 1),
        "tiempo_total_h": round(desglose["tiempo_h"], 2),
        "peajes_total_eur": round(desglose["tax_eur"], 2),
        "combustible_total_eur": round(desglose["gas_eur"], 2),
        "evaluaciones_tour": int(res.get("evaluaciones_tour", 0)),
        "pasadas_or_opt": int(res.get("pasadas_or_opt", 0)),
        "tiempo_ejecucion_s": round(dt, 2),
        "historia_mejor": res["historia_mejor"],
        "historia_evals_tour": res.get("historia_evals_tour", []),
        "historia_tiempos": res.get("historia_tiempos", []),
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
        _sincronizar_asset(out_gif)
        _sincronizar_asset(out_png)

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


def comparar_aco_vs_ga_justo(datos, M, cfg: dict, vh: float):
    """Compara ACO y GA bajo igualdad de esfuerzo (evaluaciones de tour, Or-opt y tiempo)."""
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    aco_cfg = cfg.get("aco", {})
    ga_cfg = cfg.get("ga", {})
    semilla = int(cfg.get("semilla", 42))
    alfa = float(aco_cfg.get("alfa", 1.0))
    beta = float(aco_cfg.get("beta", 3.0))
    evap = float(aco_cfg.get("evaporacion", 0.4))
    pc = float(ga_cfg.get("probabilidad_cruce", 0.8))
    pm = float(ga_cfg.get("probabilidad_mutacion", 0.2))

    print("\n" + "=" * 78)
    print("PARTE 2: COMPARATIVA ACO vs GA EN IGUALDAD DE ESFUERZO")
    print("=" * 78)

    escenarios = [
        # 1. Sin Or-opt, 3.000 tours (20x150 ACO)
        ("Sin Or-opt (3.000 tours)", "ACO", dict(
            hormigas=20, iteraciones=150, alfa=alfa, beta=beta, evaporacion=evap,
            semilla=semilla, usar_or_opt=False,
        )),
        ("Sin Or-opt (3.000 tours)", "GA", dict(
            tam_poblacion=50, generaciones=100, prob_cruce=pc, prob_mutacion=pm,
            semilla=semilla, usar_or_opt=False, max_evaluaciones=3000,
        )),
        # 2. Sin Or-opt, 15.000 tours (50x300 ACO)
        ("Sin Or-opt (15.000 tours)", "ACO", dict(
            hormigas=50, iteraciones=300, alfa=alfa, beta=beta, evaporacion=evap,
            semilla=semilla, usar_or_opt=False,
        )),
        ("Sin Or-opt (15.000 tours)", "GA", dict(
            tam_poblacion=100, generaciones=250, prob_cruce=pc, prob_mutacion=pm,
            semilla=semilla, usar_or_opt=False, max_evaluaciones=15000,
        )),
        # 3. Con Or-opt, 3.000 tours + ~150 pasadas Or-opt
        ("Con Or-opt (3.000 tours + Or-opt)", "ACO", dict(
            hormigas=20, iteraciones=150, alfa=alfa, beta=beta, evaporacion=evap,
            semilla=semilla, usar_or_opt=True,
        )),
        ("Con Or-opt (3.000 tours + Or-opt)", "GA", dict(
            tam_poblacion=20, generaciones=150, prob_cruce=pc, prob_mutacion=pm,
            semilla=semilla, usar_or_opt=True, max_evaluaciones=3000,
        )),
        # 4. Tiempo de computo comparable (configuracion principal)
        ("Configuracion principal (tiempo ~6-7s)", "ACO+Or-opt", dict(
            hormigas=int(aco_cfg.get("hormigas", 20)),
            iteraciones=int(aco_cfg.get("iteraciones", 150)),
            alfa=alfa, beta=beta, evaporacion=evap, semilla=semilla, usar_or_opt=True,
        )),
        ("Configuracion principal (tiempo ~6-7s)", "GA (100x1000)", dict(
            tam_poblacion=int(ga_cfg.get("poblacion", 100)),
            generaciones=int(ga_cfg.get("generaciones", 1000)),
            prob_cruce=pc, prob_mutacion=pm, semilla=semilla, usar_or_opt=False,
        )),
    ]

    filas = []
    curvas_3k_15k = {}
    for regimen, metodo, kwargs in escenarios:
        t0 = time.perf_counter()
        if metodo.startswith("ACO"):
            res = ejecutar_aco(M, **kwargs, verbose=False)
        else:
            res = ejecutar_ga(M, **kwargs, verbose=False)
        dt = time.perf_counter() - t0
        desg = datos.desglose_tour(res["tour"], vh)
        fila = {
            "regimen": regimen,
            "algoritmo": metodo,
            "usar_or_opt": bool(res.get("usar_or_opt", False)),
            "evaluaciones_tour": int(res.get("evaluaciones_tour", 0)),
            "pasadas_or_opt": int(res.get("pasadas_or_opt", 0)),
            "coste_eur": round(desg["coste_eur"], 2),
            "distancia_km": round(desg["distancia_km"], 2),
            "tiempo_h": round(desg["tiempo_h"], 2),
            "peajes_eur": round(desg["tax_eur"], 2),
            "gas_eur": round(desg["gas_eur"], 2),
            "tiempo_computo_s": round(dt, 2),
        }
        filas.append(fila)
        curvas_3k_15k[(regimen, metodo)] = (
            res.get("historia_evals_tour", []),
            res.get("historia_mejor", []),
            res.get("historia_tiempos", []),
            fila["coste_eur"],
        )
        print(f"  [{regimen:36s}] {metodo:12s}: {fila['coste_eur']:7.2f} EUR | "
              f"evals={fila['evaluaciones_tour']:6d} | oropt={fila['pasadas_or_opt']:3d} | "
              f"t={fila['tiempo_computo_s']:5.2f}s")

    out_json = DIR_RESULTADOS / "comparacion_justa_aco_ga.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"valor_hora_eur": vh, "semilla": semilla, "resultados": filas}, f, ensure_ascii=False, indent=2)

    out_csv = DIR_RESULTADOS / "comparacion_justa_aco_ga.csv"
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)

    # Grafica comparativa en eje comun de esfuerzo: evaluaciones de tour y tiempo de computo
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    series_evals = [
        ("Sin Or-opt (15.000 tours)", "ACO", "ACO sin Or-opt (15k tours)", "#d62728", "--"),
        ("Sin Or-opt (15.000 tours)", "GA", "GA sin Or-opt (15k tours)", "#2980b9", "--"),
        ("Con Or-opt (3.000 tours + Or-opt)", "ACO", "ACO + Or-opt (3k tours)", "#27ae60", "-"),
        ("Con Or-opt (3.000 tours + Or-opt)", "GA", "GA + Or-opt (3k tours)", "#8e44ad", "-."),
    ]
    for reg, met, lbl, col, ls in series_evals:
        he, hm, _, cf = curvas_3k_15k[(reg, met)]
        if he and hm:
            axes[0].plot(he, hm, label=f"{lbl}: {cf:.2f} €", color=col, linestyle=ls, linewidth=2.1)

    series_tiempo = [
        ("Configuracion principal (tiempo ~6-7s)", "ACO+Or-opt", "ACO + Or-opt (3k tours)", "#27ae60", "-"),
        ("Configuracion principal (tiempo ~6-7s)", "GA (100x1000)", "GA sin Or-opt (84.1k tours)", "#2980b9", "--"),
        ("Sin Or-opt (3.000 tours)", "ACO", "ACO sin Or-opt (3k tours)", "#d62728", ":"),
        ("Con Or-opt (3.000 tours + Or-opt)", "GA", "GA + Or-opt (2.6k tours)", "#8e44ad", "-."),
    ]
    for reg, met, lbl, col, ls in series_tiempo:
        _, hm, ht, cf = curvas_3k_15k[(reg, met)]
        if ht and hm:
            axes[1].plot(ht, hm, label=f"{lbl}: {cf:.2f} €", color=col, linestyle=ls, linewidth=2.1)

    axes[0].set_title("Convergencia por Evaluaciones de Tour", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Evaluaciones de recorrido construidas / evaluadas", fontsize=10)
    axes[0].set_ylabel("Mejor Costo del Tour (EUR)", fontsize=10)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=8.5, loc="upper right")

    axes[1].set_title("Convergencia por Tiempo de Cómputo (<= 7.5 s)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Tiempo acumulado (s)", fontsize=10)
    axes[1].set_ylabel("Mejor Costo del Tour (EUR)", fontsize=10)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(fontsize=8.5, loc="upper right")

    fig.suptitle(
        f"Comparación ACO vs GA en Igualdad de Esfuerzo (TSP 47 Capitales, h={vh:.2f} EUR/h, semilla={semilla})",
        fontsize=12.5, fontweight="bold", y=0.98,
    )
    fig.tight_layout()
    out_png = DIR_RESULTADOS / "comparacion_convergencia_aco_ga.png"
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    _sincronizar_asset(out_png)
    print(f"[Parte 2] Comparativa justa guardada en {out_json} y {out_png}")
    return filas


def main():
    parser = argparse.ArgumentParser(description="Runner Parte 2: Optimizacion Combinatoria (TSP 47 Capitales)")
    parser.add_argument("--demo", action="store_true", help="Perfil rapido (<15 s)")
    parser.add_argument("--algoritmo", choices=["aco", "ga", "ambos"], default="ambos",
                        help="Algoritmo a ejecutar (por defecto: ambos)")
    parser.add_argument("--valor-hora", type=float, default=None, help="Valor de la hora del vendedor (EUR/h)")
    parser.add_argument("--sin-or-opt", action="store_true", help="Desactiva Or-opt en ACO")
    parser.add_argument("--comparar-oropt", action="store_true", help="Ejecuta comparativa Con vs Sin Or-opt")
    parser.add_argument("--comparar-justo", action="store_true", help="Ejecuta comparativa ACO vs GA en igualdad de esfuerzo")
    parser.add_argument("--sensibilidad", action="store_true", help="Ejecuta estudio de sensibilidad del valor de hora")
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

    if args.sensibilidad:
        from sensibilidad import analizar_sensibilidad
        analizar_sensibilidad(demo=args.demo)
        return

    if args.barrido:
        from experimentos_aco import main as ejecutar_barrido_aco
        ejecutar_barrido_aco()
        return

    if args.comparar_oropt:
        from experimentos_aco import comparar_con_y_sin_oropt
        comparar_con_y_sin_oropt(M, datos, vh, cfg.get("aco", {}))
        print(f"\nComparativa guardada en {DIR_RESULTADOS} ({time.time()-t0:.2f}s)")
        return

    if args.comparar_justo:
        comparar_aco_vs_ga_justo(datos, M, cfg, vh)
        print(f"\nComparativa justa guardada en {DIR_RESULTADOS} ({time.time()-t0:.2f}s)")
        return

    resultados = {}
    if args.algoritmo in ("aco", "ambos"):
        resultados["aco"] = correr_aco_individual(datos, M, cfg, vh, usar_or_opt=not args.sin_or_opt, hacer_gif=not args.sin_gif)

    if args.algoritmo in ("ga", "ambos"):
        resultados["ga"] = correr_ga_individual(datos, M, cfg, vh, hacer_gif=not args.sin_gif)

    if "aco" in resultados and "ga" in resultados:
        generar_grafica_comparativa_parte2(resultados["aco"], resultados["ga"], vh,
                                           DIR_RESULTADOS / "comparacion_convergencia.png")
        if not args.demo:
            comparar_aco_vs_ga_justo(datos, M, cfg, vh)

    print(f"\nTOTAL PARTE 2: {time.time()-t0:.2f}s | Resultados en {DIR_RESULTADOS}")


if __name__ == "__main__":
    main()
