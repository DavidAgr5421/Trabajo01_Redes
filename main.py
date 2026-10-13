"""
Runner unico: cada experimento principal se lanza con un solo comando.

Ejemplos (sustentacion, sin editar codigo):
  python3 main.py --demo                    # corrida demo <2 min (GA+ACO+GIF)
  python3 main.py --algoritmo ga            # GA completo + CSV + GIF
  python3 main.py --algoritmo aco           # ACO completo + CSV + GIF
  python3 main.py --algoritmo ambos         # GA + ACO + comparativa
  python3 main.py --sensibilidad            # barrido valor_hora + umbrales
  python3 main.py --demo --valor-hora 35    # override puntual por CLI
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from config import cargar_config
from costes_comunes import BASE_DIR, cargar_costes

DIR_RESULTADOS = BASE_DIR / "resultados"


def correr_ga(datos, M, cfg, vh, hacer_gif: bool):
    from algoritmo_genetico import ejecutar_ga, guardar_resultados
    ga = cfg["ga"]
    t0 = time.time()
    res = ejecutar_ga(M, tam_poblacion=ga["poblacion"],
                      generaciones=ga["generaciones"],
                      prob_mutacion=ga.get("prob_mutacion", 0.3),
                      semilla=ga.get("semilla", 42), verbose=True)
    _, _, resumen = guardar_resultados(datos, res["tour"], vh, res)
    print(f"[GA] coste {res['coste']:.2f} EUR en {time.time()-t0:.1f}s")
    if hacer_gif:
        from generar_gif import construir_animacion
        from matplotlib.animation import PillowWriter
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        gif_c = cfg.get("gif", {})
        fig, anim, reg, r = construir_animacion(
            datos, vh, algoritmo="ga",
            tam_poblacion=ga["poblacion"], generaciones=ga["generaciones"],
            semilla=ga.get("semilla", 42),
            cada=gif_c.get("cada", 4), fps=gif_c.get("fps", 8))
        out = DIR_RESULTADOS / "evolucion_ga.gif"
        anim.save(str(out), writer=PillowWriter(fps=gif_c.get("fps", 8)), dpi=110)
        fig.savefig(DIR_RESULTADOS / "evolucion_ga_final.png", dpi=140)
        plt.close(fig)
        print(f"[GA] GIF: {out} ({len(reg)} frames)")
    return resumen


def correr_aco(datos, M, cfg, vh, hacer_gif: bool):
    from colonia_hormigas import ejecutar_aco, guardar_resultados
    aco = cfg["aco"]
    t0 = time.time()
    res = ejecutar_aco(M, hormigas=aco["hormigas"],
                       iteraciones=aco["iteraciones"],
                       alfa=aco.get("alfa", 1.0), beta=aco.get("beta", 3.0),
                       evaporacion=aco.get("evaporacion", 0.2),
                       semilla=aco.get("semilla", 42),
                       usar_or_opt=aco.get("usar_or_opt", True),
                       verbose=True)
    _, resumen = guardar_resultados(datos, res["tour"], vh, res)
    print(f"[ACO] coste {res['coste']:.2f} EUR en {time.time()-t0:.1f}s")
    if hacer_gif:
        from generar_gif import construir_animacion
        from matplotlib.animation import PillowWriter
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        gif_c = cfg.get("gif", {})
        fig, anim, reg, r = construir_animacion(
            datos, vh, algoritmo="aco",
            semilla=aco.get("semilla", 42),
            cada=gif_c.get("cada", 4), fps=gif_c.get("fps", 8),
            hormigas=aco["hormigas"], iteraciones=aco["iteraciones"],
            alfa=aco.get("alfa", 1.0), beta=aco.get("beta", 3.0),
            evaporacion=aco.get("evaporacion", 0.2),
            usar_or_opt=aco.get("usar_or_opt", True))
        out = DIR_RESULTADOS / "evolucion_aco.gif"
        anim.save(str(out), writer=PillowWriter(fps=gif_c.get("fps", 8)), dpi=110)
        fig.savefig(DIR_RESULTADOS / "evolucion_aco_final.png", dpi=140)
        plt.close(fig)
        print(f"[ACO] GIF: {out} ({len(reg)} frames)")
    return resumen


def main():
    p = argparse.ArgumentParser(description="Runner unico TSP 47 capitales")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--demo", action="store_true",
                   help="Perfil rapido <2 min para sustentacion")
    p.add_argument("--algoritmo", choices=["ga", "aco", "ambos"],
                   default="ambos")
    p.add_argument("--valor-hora", type=float, default=None)
    p.add_argument("--sin-gif", action="store_true")
    p.add_argument("--sensibilidad", action="store_true",
                   help="Tambien corre el barrido de valor_hora")
    args = p.parse_args()

    cfg = cargar_config(args.config, demo=args.demo)
    vh = args.valor_hora if args.valor_hora is not None else cfg["valor_hora"]
    print(f"valor_hora={vh:.2f} EUR/h | algoritmo={args.algoritmo} "
          f"| demo={args.demo}")

    datos = cargar_costes()
    M = datos.matriz_coste(vh)
    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    hacer_gif = not args.sin_gif

    t0 = time.time()
    resultados = {}
    if args.algoritmo in ("ga", "ambos"):
        resultados["ga"] = correr_ga(datos, M, cfg, vh, hacer_gif)
    if args.algoritmo in ("aco", "ambos"):
        resultados["aco"] = correr_aco(datos, M, cfg, vh, hacer_gif)

    if args.sensibilidad:
        from sensibilidad_valor_hora import (barrido, envolvente_inferior,
                                             guardar_salidas)
        s = cfg["sensibilidad"]
        import numpy as np
        valores = np.arange(s["desde"], s["hasta"] + 1e-9, s["paso"])
        cand, regs, _ = barrido(datos, valores,
                                tam_poblacion=s.get("poblacion", 40),
                                generaciones=s.get("generaciones", 40),
                                semilla=s.get("semilla", 7))
        tramos, cambios = envolvente_inferior(cand, float(valores[0]),
                                              float(valores[-1]))
        resumen, _ = guardar_salidas(datos, regs, cand, tramos, cambios)
        print(f"[SENS] cambios en v={resumen['valores_cambio_eur_h']}")

    with open(DIR_RESULTADOS / "comparativa.json", "w",
              encoding="utf-8") as f:
        json.dump({"valor_hora": vh, **resultados}, f, ensure_ascii=False,
                  indent=2)
    print(f"TOTAL {time.time()-t0:.1f}s | Salidas en resultados/")


if __name__ == "__main__":
    main()
