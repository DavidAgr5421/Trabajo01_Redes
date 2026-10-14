"""
Runner unico para el TSP de las 47 capitales peninsulares espanolas con ACO.

Permite ejecutar todas las variantes desde un solo comando:
  python main.py                    # Ejecuta ACO (con Or-opt) + CSV + mapa + GIF
  python main.py --sin-or-opt       # Ejecuta ACO sin busqueda local Or-opt
  python main.py --comparar-oropt   # Ejecuta comparativa Con Or-opt vs Sin Or-opt
  python main.py --barrido          # Ejecuta el barrido completo de configuraciones
  python main.py --demo             # Corrida rapida (<15 s) para demostracion/sustentacion
  python main.py --sin-gif          # Ejecuta ACO sin generar archivo GIF (mas rapido)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from config import cargar_config
from costes_comunes import BASE_DIR, VALOR_HORA_POR_DEFECTO, cargar_costes

DIR_RESULTADOS = BASE_DIR / "resultados"


def correr_aco_individual(datos, M, cfg, vh: float, usar_or_opt: bool, hacer_gif: bool):
    from colonia_hormigas import ejecutar_aco, guardar_resultados
    aco = cfg.get("aco", {})
    t0 = time.time()
    sufijo = "con_oropt" if usar_or_opt else "sin_oropt"

    print(f"\n[ACO {'CON' if usar_or_opt else 'SIN'} Or-opt] "
          f"Iniciando ejecucion ({aco.get('hormigas', 20)} hormigas, "
          f"{aco.get('iteraciones', 150)} iteraciones, semilla={aco.get('semilla', 42)})...")

    res = ejecutar_aco(
        M,
        hormigas=aco.get("hormigas", 20),
        iteraciones=aco.get("iteraciones", 150),
        alfa=aco.get("alfa", 1.0),
        beta=aco.get("beta", 3.0),
        evaporacion=aco.get("evaporacion", 0.2),
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

        print(f"[ACO {sufijo}] Generando animacion GIF...")
        fig, anim, reg, _ = construir_animacion(
            datos, vh, algoritmo="aco",
            semilla=aco.get("semilla", 42),
            cada=cada, fps=fps,
            hormigas=aco.get("hormigas", 20),
            iteraciones=aco.get("iteraciones", 150),
            alfa=aco.get("alfa", 1.0),
            beta=aco.get("beta", 3.0),
            evaporacion=aco.get("evaporacion", 0.2),
            usar_or_opt=usar_or_opt,
        )
        out_gif = DIR_RESULTADOS / f"evolucion_aco_{sufijo}.gif"
        out_png = DIR_RESULTADOS / f"mapa_aco_{sufijo}.png"
        anim.save(str(out_gif), writer=PillowWriter(fps=fps), dpi=110)
        fig.savefig(out_png, dpi=130)
        plt.close(fig)
        print(f"[ACO {sufijo}] GIF guardado: {out_gif}")
        print(f"[ACO {sufijo}] Mapa final: {out_png}")

    return resumen


def main():
    parser = argparse.ArgumentParser(description="Runner central para el TSP de 47 capitales con ACO")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--demo", action="store_true",
                        help="Perfil rapido para sustentacion (<15 s)")
    parser.add_argument("--valor-hora", type=float, default=None,
                        help="Sobreescribe el valor de la hora del vendedor (EUR/h)")
    parser.add_argument("--sin-or-opt", action="store_true",
                        help="Ejecuta ACO sin la busqueda local Or-opt")
    parser.add_argument("--comparar-oropt", action="store_true",
                        help="Ejecuta y compara ACO Con Or-opt vs Sin Or-opt")
    parser.add_argument("--barrido", action="store_true",
                        help="Ejecuta el barrido completo de configuraciones con semilla fija 42")
    parser.add_argument("--sin-gif", action="store_true",
                        help="Omite la generacion de animaciones GIF")
    args = parser.parse_args()

    cfg = cargar_config(args.config, demo=args.demo)
    vh = args.valor_hora if args.valor_hora is not None else cfg.get("valor_hora", VALOR_HORA_POR_DEFECTO)
    semilla = cfg.get("semilla", cfg.get("aco", {}).get("semilla", 42))

    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    datos = cargar_costes()
    M = datos.matriz_coste(vh)

    print("=" * 65)
    print(f"SISTEMA DE RUTAS OPTIMAS ACO - 47 CAPITALES ESPAÑOLAS")
    print(f"Valor hora : {vh:.2f} EUR/h | Semilla : {semilla} | Demo : {args.demo}")
    print("=" * 65)

    t0 = time.time()

    if args.barrido:
        from experimentos_aco import main as ejecutar_barrido
        ejecutar_barrido()
        return

    if args.comparar_oropt:
        from experimentos_aco import comparar_con_y_sin_oropt
        cfg_aco = cfg.get("aco", {})
        comparar_con_y_sin_oropt(M, datos, vh, cfg_aco)

        # Generar ambos GIFs si no se especifico --sin-gif
        if not args.sin_gif:
            from generar_gif import construir_animacion
            from matplotlib.animation import PillowWriter
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            gif_c = cfg.get("gif", {})
            cada = gif_c.get("cada", 2)
            fps = gif_c.get("fps", 8)

            for opt, sufijo in [(True, "con_oropt"), (False, "sin_oropt")]:
                print(f"Generando GIF ({sufijo})...")
                fig, anim, _, _ = construir_animacion(
                    datos, vh, algoritmo="aco",
                    semilla=semilla, cada=cada, fps=fps,
                    hormigas=cfg_aco.get("hormigas", 20),
                    iteraciones=cfg_aco.get("iteraciones", 150),
                    alfa=cfg_aco.get("alfa", 1.0),
                    beta=cfg_aco.get("beta", 3.0),
                    evaporacion=cfg_aco.get("evaporacion", 0.2),
                    usar_or_opt=opt,
                )
                out_gif = DIR_RESULTADOS / f"evolucion_aco_{sufijo}.gif"
                out_png = DIR_RESULTADOS / f"mapa_aco_{sufijo}.png"
                anim.save(str(out_gif), writer=PillowWriter(fps=fps), dpi=110)
                fig.savefig(out_png, dpi=130)
                plt.close(fig)

        print(f"\n[OK] Comparativa completa guardada en {DIR_RESULTADOS} en {time.time()-t0:.1f}s")
        return

    # Ejecucion normal de una variante
    usar_or_opt = not args.sin_or_opt
    resumen = correr_aco_individual(datos, M, cfg, vh, usar_or_opt=usar_or_opt, hacer_gif=not args.sin_gif)

    # Actualizar comparativa.json
    res_json_path = DIR_RESULTADOS / "comparativa.json"
    with open(res_json_path, "w", encoding="utf-8") as f:
        json.dump({"valor_hora": vh, "semilla": semilla, "aco": resumen}, f, ensure_ascii=False, indent=2)

    print(f"\nTOTAL: {time.time()-t0:.2f}s | Resultados en {DIR_RESULTADOS}")


if __name__ == "__main__":
    main()
