"""
Runner unificado para la Parte 1: Optimizacion Numerica.

Ejecuta los cuatro metodos requeridos sobre las funciones de prueba (Rosenbrock y Rastrigin):
  1. Descenso por Gradiente (GD)
  2. Optimizacion por Enjambre de Particulas (PSO)
  3. Algoritmo Evolutivo (EA)
  4. Evolucion Diferencial (DE)

Cumple con la Regla de Comparacion Justa:
  - Conteo transparente de evaluaciones de f(x) y de gradiente grad_f(x).
  - Equivalencia analitica: aproximar gradiente en dimension n cuesta 2n evaluaciones de f.
    evals_equivalentes = evals_f + (2 * n * evals_gradiente)
  - 30 corridas independientes con resumen estadistico (Media, Std, Min, Max, Tasa de Exito).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import pandas as pd

from config import cargar_config
from funciones import grad_rastrigin, grad_rosenbrock, rastrigin, rosenbrock
from gradienteDecendente import gradient_descent
from PSO import pso
from algoritmo_evolutivo import optimizar_algoritmo_evolutivo
from evolucion_diferencial import optimizar_evolucion_diferencial

DIR_RESULTADOS = Path(__file__).resolve().parent / "resultados"
DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)

FUNCIONES_INFO = {
    "Rosenbrock": {
        "func": rosenbrock,
        "grad": grad_rosenbrock,
        "limites": (-5.0, 5.0),
        "optimo_f": 0.0,
        "optimo_x": lambda d: np.ones(d),
        "alpha": 0.001,
        "umbral_exito": 1e-3,
    },
    "Rastrigin": {
        "func": rastrigin,
        "grad": grad_rastrigin,
        "limites": (-5.12, 5.12),
        "optimo_f": 0.0,
        "optimo_x": lambda d: np.zeros(d),
        "alpha": 0.002,
        "umbral_exito": 1e-3,
    },
}


def ejecutar_metodo(
    metodo: str,
    f_info: dict,
    dim: int,
    semilla: int,
    cfg: dict,
    nombre_f: str = "Rosenbrock",
    respetar_presupuesto: bool = True,
) -> dict:
    """Ejecuta una corrida individual del metodo seleccionado."""
    func = f_info["func"]
    grad = f_info["grad"]
    lb, ub = f_info["limites"]
    rng = np.random.default_rng(semilla)
    presupuesto_equiv = int(cfg.get("presupuesto_evaluaciones_equiv", 15000))

    if metodo == "GD":
        c_gd = cfg.get("gradiente_descendente", {})
        punto_configurado = c_gd.get("punto_inicial")
        if punto_configurado is None:
            x0 = rng.uniform(lb, ub, size=dim)
        else:
            punto_configurado = np.asarray(punto_configurado, dtype=float)
            if punto_configurado.ndim != 1 or punto_configurado.size not in (2, 3):
                raise ValueError("El punto inicial de GD debe tener 2 o 3 coordenadas.")
            if not np.all(np.isfinite(punto_configurado)):
                raise ValueError("Las coordenadas del punto inicial de GD deben ser finitas.")
            x0 = (
                punto_configurado
                if punto_configurado.size == dim
                else rng.uniform(lb, ub, size=dim)
            )
        alpha_key = f"alpha_{nombre_f.lower()}"
        alpha = float(c_gd.get(alpha_key, f_info["alpha"]))
        max_iter_cfg = int(c_gd.get("max_iter", 10000))
        if respetar_presupuesto and punto_configurado is None:
            max_iter_gd = min(max_iter_cfg, max(1, (presupuesto_equiv - 1) // (1 + 2 * dim)))
        else:
            max_iter_gd = max_iter_cfg

        res = gradient_descent(
            func, grad, x0,
            alpha=alpha,
            max_iter=max_iter_gd,
            tolerancia=float(c_gd.get("tolerancia", 1e-8)),
        )
        evals_f = res["evaluaciones_f"]
        evals_grad = res["evaluaciones_gradiente"]
        f_val = res["f_final"]
        historial_f = np.asarray(res["historial_f"], dtype=float)
        historial_x = res["historial_x"]
        evals_equiv_hist = 1 + np.arange(len(historial_f)) * (1 + 2 * dim)

    elif metodo == "PSO":
        c_pso = cfg.get("pso", {})
        n_part = int(c_pso.get("n_particulas", 30))
        max_it = int(c_pso.get("max_iter", 499))
        res = pso(
            func, dim,
            n_particulas=n_part,
            max_iter=max_it,
            limites=(lb, ub),
            w=float(c_pso.get("w", 0.7)),
            c1=float(c_pso.get("c1", 1.5)),
            c2=float(c_pso.get("c2", 1.5)),
            semilla=semilla,
        )
        evals_f = res["evaluaciones_f"]
        evals_grad = 0
        f_val = res["f_final"]
        historial_f = np.asarray(res["historial_f"], dtype=float)
        historial_x = res["historial_x"]
        evals_equiv_hist = (np.arange(len(historial_f)) + 1) * n_part

    elif metodo == "EA":
        c_ea = cfg.get("algoritmo_evolutivo", {})
        n_pop = int(c_ea.get("poblacion", 40))
        max_gen = int(c_ea.get("generaciones", 440))
        frac_el = float(c_ea.get("elitismo", 0.15))
        res = optimizar_algoritmo_evolutivo(
            func, dim, lb, ub,
            n_pop=n_pop,
            max_generaciones=max_gen,
            frac_elitismo=frac_el,
            frac_mutacion=float(c_ea.get("mutacion", 0.2)),
            frac_cruzamiento=float(c_ea.get("cruzamiento", 0.65)),
            semilla=semilla,
        )
        evals_f = res["evaluaciones"]
        evals_grad = 0
        f_val = res["mejor_f"]
        historial_f = np.asarray(res["historial_mejor_f"], dtype=float)
        historial_x = [res["mejor_x"]]
        n_elite = int(np.round(n_pop * frac_el))
        evals_equiv_hist = n_pop + np.arange(len(historial_f)) * (n_pop - n_elite)

    elif metodo == "DE":
        c_de = cfg.get("evolucion_diferencial", {})
        n_pop = int(c_de.get("poblacion", 40))
        max_gen = int(c_de.get("generaciones", 374))
        res = optimizar_evolucion_diferencial(
            func, dim, lb, ub,
            n_pop=n_pop,
            max_generaciones=max_gen,
            F=float(c_de.get("F", 0.7)),
            CR=float(c_de.get("CR", 0.8)),
            semilla=semilla,
        )
        evals_f = res["evaluaciones"]
        evals_grad = 0
        f_val = res["mejor_f"]
        historial_f = np.asarray(res["historial_mejor_f"], dtype=float)
        historial_x = [res["mejor_x"]]
        evals_equiv_hist = (np.arange(len(historial_f)) + 1) * n_pop
    else:
        raise ValueError(f"Metodo no reconocido: {metodo}")

    # Regla de equivalencia justa: cada evaluacion de gradiente equivale a 2n evaluaciones de f
    evals_equiv = evals_f + (2 * dim * evals_grad)

    return {
        "metodo": metodo,
        "f_val": float(f_val),
        "evals_f": int(evals_f),
        "evals_grad": int(evals_grad),
        "evals_equiv": int(evals_equiv),
        "historial_f": historial_f,
        "evals_equiv_hist": evals_equiv_hist,
        "historial_x": historial_x,
    }


def ejecutar_estudio_estadistico(cfg: dict, n_corridas: int = 30) -> tuple[pd.DataFrame, dict]:
    """Ejecuta las corridas independientes por metodo/funcion/dimension con el mismo presupuesto."""
    metodos = ["GD", "PSO", "EA", "DE"]
    filas = []
    curvas_convergencia: dict[tuple[str, int], dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    presupuesto_equiv = int(cfg.get("presupuesto_evaluaciones_equiv", 15000))
    umbral_global = float(cfg.get("umbral_exito", 1e-3))

    print("\n" + "=" * 78)
    print(f"PARTE 1: PROTOCOLO EN IGUALDAD DE ESFUERZO ({n_corridas} CORRIDAS, TOPE {presupuesto_equiv:,} E_equiv)")
    print("=" * 78)

    eje_comun = np.linspace(50, presupuesto_equiv, 150)

    for nombre_f in cfg.get("funciones_seleccionadas", ["Rosenbrock", "Rastrigin"]):
        f_info = FUNCIONES_INFO[nombre_f]
        for dim in cfg.get("dimensiones", [2, 3]):
            print(f"\n>>> Evaluando {nombre_f} en {dim}D...")
            curvas_convergencia[(nombre_f, dim)] = {}
            for met in metodos:
                valores = []
                evals_f_list = []
                evals_grad_list = []
                evals_eq_list = []
                curvas_interp = []
                exitos = 0
                t0 = time.time()

                for corrida in range(n_corridas):
                    semilla = cfg.get("semilla_base", 2026) + corrida * 37
                    res = ejecutar_metodo(
                        met, f_info, dim, semilla, cfg,
                        nombre_f=nombre_f, respetar_presupuesto=True
                    )
                    val = res["f_val"]
                    valores.append(val)
                    evals_f_list.append(res["evals_f"])
                    evals_grad_list.append(res["evals_grad"])
                    evals_eq_list.append(res["evals_equiv"])

                    if np.isfinite(val) and abs(val - f_info["optimo_f"]) <= umbral_global:
                        exitos += 1

                    # Curva del mejor acumulado vs E_equiv para graficar en eje comun
                    hf = np.asarray(res["historial_f"], dtype=float)
                    he = np.asarray(res["evals_equiv_hist"], dtype=float)
                    if len(hf) > 0 and np.isfinite(hf[0]):
                        mejor_acum = np.minimum.accumulate(np.where(np.isfinite(hf), hf, np.inf))
                        curvas_interp.append(np.interp(eje_comun, he, mejor_acum))

                dt = time.time() - t0
                val_arr = np.array(valores, dtype=float)
                val_validos = val_arr[np.isfinite(val_arr)]
                divergentes = int(np.sum(~np.isfinite(val_arr)))

                media_f = float(np.mean(val_validos)) if len(val_validos) > 0 else np.nan
                mediana_f = float(np.median(val_validos)) if len(val_validos) > 0 else np.nan
                std_f = float(np.std(val_validos)) if len(val_validos) > 0 else np.nan
                min_f = float(np.min(val_validos)) if len(val_validos) > 0 else np.nan
                max_f = float(np.max(val_validos)) if len(val_validos) > 0 else np.nan

                tasa_exito = round((exitos / n_corridas) * 100, 2)
                media_eval_eq = int(round(np.mean(evals_eq_list)))
                costo_exito = round(media_eval_eq / (exitos / n_corridas), 1) if exitos > 0 else None

                if curvas_interp:
                    mediana_curva = np.nanmedian(np.vstack(curvas_interp), axis=0)
                    curvas_convergencia[(nombre_f, dim)][met] = (eje_comun, mediana_curva)

                filas.append({
                    "Funcion": nombre_f,
                    "Dimension": f"{dim}D",
                    "Metodo": met,
                    "Media_f": media_f,
                    "Mediana_f": mediana_f,
                    "Std_f": std_f,
                    "Mejor_f": min_f,
                    "Peor_f": max_f,
                    "Exitos": exitos,
                    "Divergentes": divergentes,
                    "Tasa_Exito_%": tasa_exito,
                    "Media_Evals_f": int(round(np.mean(evals_f_list))),
                    "Media_Evals_grad": int(round(np.mean(evals_grad_list))),
                    "Media_Evals_Equiv": media_eval_eq,
                    "Presupuesto_Equiv": presupuesto_equiv,
                    "Costo_Por_Exito": costo_exito,
                    "Tiempo_Total_s": round(dt, 2),
                })
                print(f"  [{met:3s}] Mediana: {mediana_f:10.3e} | Media: {media_f:10.3e} | "
                      f"Exito: {exitos:2d}/{n_corridas} ({tasa_exito:5.1f}%) | "
                      f"Evals Eq: {media_eval_eq:6d} ({dt:.2f}s)")

    df = pd.DataFrame(filas)
    csv_path = DIR_RESULTADOS / "resumen_comparativo_parte1.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    json_path = DIR_RESULTADOS / "resumen_comparativo_parte1.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(filas, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 78)
    print("Resultados guardados en:")
    print(f"  - CSV : {csv_path}")
    print(f"  - JSON: {json_path}")
    print("=" * 78)
    return df, curvas_convergencia


def generar_gif_2d(nombre_f: str = "Rosenbrock", metodo: str = "GD", fps: int = 15):
    """Genera animacion GIF 2D del proceso de optimizacion sobre curvas de nivel."""
    f_info = FUNCIONES_INFO[nombre_f]
    func = f_info["func"]
    lb, ub = f_info["limites"]

    print(f"Generando GIF 2D para {metodo} en {nombre_f}...")
    cfg = cargar_config(parte="parte_1")
    res = ejecutar_metodo(metodo, f_info, dim=2, semilla=42, cfg=cfg)

    hist_x = np.array(res["historial_x"])
    if len(hist_x) == 0:
        return

    # Malla para curvas de nivel
    x_grid = np.linspace(lb, ub, 150)
    y_grid = np.linspace(lb, ub, 150)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))

    fig, ax = plt.subplots(figsize=(8, 7))
    if nombre_f == "Rosenbrock":
        levels = np.logspace(-1, 3.5, 30)
        cs = ax.contour(X, Y, Z, levels=levels, cmap="viridis", alpha=0.7)
    else:
        levels = 30
        cs = ax.contour(X, Y, Z, levels=levels, cmap="plasma", alpha=0.7)

    # Optimo conocido
    x_opt = f_info["optimo_x"](2)
    ax.scatter([x_opt[0]], [x_opt[1]], color="red", marker="*", s=150, zorder=5, label="Minimo Global")

    linea, = ax.plot([], [], color="cyan" if metodo == "GD" else "lime",
                     marker="o", markersize=4, linewidth=1.5, label=f"Trayectoria {metodo}")
    ax.set_title(f"Optimizacion {metodo} en {nombre_f} 2D", fontsize=12, fontweight="bold")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(loc="upper left")
    fig.tight_layout()

    # Reducir fotogramas si la trayectoria es muy larga
    n_frames = min(len(hist_x), 120)
    indices = np.linspace(0, len(hist_x) - 1, n_frames, dtype=int)

    def update(frame):
        k = indices[frame]
        sub_x = hist_x[:k + 1]
        linea.set_data(sub_x[:, 0], sub_x[:, 1])
        return (linea,)

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=True)
    out_file = DIR_RESULTADOS / f"animacion_{metodo.lower()}_{nombre_f.lower()}_2d.gif"
    anim.save(str(out_file), writer=PillowWriter(fps=fps), dpi=100)
    plt.close(fig)
    print(f"GIF guardado: {out_file}")


def generar_grafica_convergencia_parte1(cfg: dict, curvas_convergencia: dict | None = None):
    """Genera grafica comparativa de convergencia vs E_equiv para Parte 1 (2D y 3D)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    metodos = ["GD", "PSO", "EA", "DE"]
    colores = {"GD": "#4472C4", "PSO": "#ED7D31", "EA": "#70AD47", "DE": "#A5A5A5"}
    combinaciones = [("Rastrigin", 2), ("Rastrigin", 3), ("Rosenbrock", 2), ("Rosenbrock", 3)]

    for ax, (nombre_f, dim) in zip(axes.flat, combinaciones):
        f_info = FUNCIONES_INFO[nombre_f]
        if curvas_convergencia and (nombre_f, dim) in curvas_convergencia:
            for met in metodos:
                if met in curvas_convergencia[(nombre_f, dim)]:
                    eje_eq, med_curva = curvas_convergencia[(nombre_f, dim)][met]
                    ax.plot(eje_eq, np.clip(med_curva, 1e-16, None),
                            label=f"{met} (med. final: {med_curva[-1]:.2e})",
                            color=colores[met], linewidth=2.1)
        else:
            for met in metodos:
                res = ejecutar_metodo(met, f_info, dim=dim, semilla=42, cfg=cfg,
                                      nombre_f=nombre_f, respetar_presupuesto=True)
                hist = np.asarray(res["historial_f"], dtype=float)
                he = np.asarray(res["evals_equiv_hist"], dtype=float)
                if len(hist) > 0:
                    ax.plot(he, np.clip(np.minimum.accumulate(hist), 1e-16, None),
                            label=f"{met} (final: {res['f_val']:.2e})",
                            color=colores[met], linewidth=2.0)

        ax.set_title(f"{nombre_f} - {dim}D", fontsize=11, fontweight="bold")
        ax.set_xlabel("Evaluaciones equivalentes ($E_{\\mathrm{equiv}}$)", fontsize=10)
        ax.set_ylabel("Mediana de $f(x)$ (escala log)", fontsize=10)
        ax.set_yscale("log")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(fontsize=8.5, loc="upper right")

    presupuesto = int(cfg.get("presupuesto_evaluaciones_equiv", 15000))
    fig.suptitle(
        f"Convergencia Comparativa en Igualdad de Esfuerzo ($E_{{\\mathrm{{equiv}}}} \\leq {presupuesto:,}$)",
        fontsize=13, fontweight="bold", y=0.98
    )
    fig.tight_layout()
    salida = DIR_RESULTADOS / "convergencia_comparativa_parte1.png"
    fig.savefig(salida, dpi=130)
    plt.close(fig)
    print(f"[Parte 1] Grafica comparativa guardada: {salida}")


def ejecutar_metodo_individual(metodo: str, funcion_nom: str, dim: int, cfg: dict, n_corridas: int = 1):
    """Ejecuta un metodo especifico de la Parte 1 de forma aislada."""
    f_info = FUNCIONES_INFO[funcion_nom]
    umbral = float(cfg.get("umbral_exito", f_info["umbral_exito"]))
    print("\n" + "=" * 75)
    print(f"PARTE 1: EJECUCION INDIVIDUAL - METODO [{metodo.upper()}]")
    print(f"Funcion: {funcion_nom} | Dimension: {dim}D | Corridas: {n_corridas}")
    print("=" * 75)

    if n_corridas == 1:
        semilla = cfg.get("semilla_base", 2026)
        res = ejecutar_metodo(
            metodo.upper(), f_info, dim, semilla, cfg,
            nombre_f=funcion_nom, respetar_presupuesto=False
        )
        x_fin = res["historial_x"][-1] if len(res["historial_x"]) > 0 else "N/A"
        print(f"\n[RESULTADO]")
        print(f"  Metodo                 : {metodo.upper()}")
        print(f"  Punto final x*         : {np.round(x_fin, 6) if isinstance(x_fin, np.ndarray) else x_fin}")
        print(f"  Valor optimo f(x*)     : {res['f_val']:.6e}")
        print(f"  Evaluaciones f(x)      : {res['evals_f']:,}")
        print(f"  Evaluaciones grad_f(x) : {res['evals_grad']:,}")
        print(f"  Evaluaciones equiv     : {res['evals_equiv']:,}")
        exito = abs(res['f_val'] - f_info['optimo_f']) <= umbral
        print(f"  Criterio de exito      : {'EXITOSO (<= ' + f'{umbral:.0e})' if exito else 'NO ALCANZO UMBRAL'}")
        print("=" * 75)
        return [res]
    else:
        valores, evals_eq, exitos = [], [], 0
        t0 = time.time()
        for c in range(n_corridas):
            sem = cfg.get("semilla_base", 2026) + c * 37
            r = ejecutar_metodo(
                metodo.upper(), f_info, dim, sem, cfg,
                nombre_f=funcion_nom, respetar_presupuesto=True
            )
            valores.append(r["f_val"])
            evals_eq.append(r["evals_equiv"])
            if np.isfinite(r["f_val"]) and abs(r["f_val"] - f_info["optimo_f"]) <= umbral:
                exitos += 1
        dt = time.time() - t0
        vals_fin = np.array([v for v in valores if np.isfinite(v)])
        print(f"\n[RESUMEN {n_corridas} CORRIDAS]")
        print(f"  Mejor f(x*)            : {np.min(vals_fin):.6e}" if len(vals_fin) else "  Mejor f(x*): NaN")
        print(f"  Media f(x*)            : {np.mean(vals_fin):.6e}" if len(vals_fin) else "  Media f(x*): NaN")
        print(f"  Mediana f(x*)          : {np.median(vals_fin):.6e}" if len(vals_fin) else "  Mediana: NaN")
        print(f"  Desv. Estandar         : {np.std(vals_fin):.6e}" if len(vals_fin) else "  Std: NaN")
        print(f"  Tasa de Exito          : {(exitos / n_corridas) * 100:.1f}% ({exitos}/{n_corridas})")
        print(f"  Evals Equiv Promedio   : {int(np.mean(evals_eq)):,}")
        print(f"  Tiempo Total           : {dt:.2f} s")
        print("=" * 75)
        return valores


def main():
    parser = argparse.ArgumentParser(description="Runner Parte 1: Optimizacion Numerica")
    parser.add_argument("--demo", action="store_true", help="Corrida rapida (<15 s)")
    parser.add_argument("--metodo", choices=["gd", "pso", "ea", "de", "todos"], default="todos",
                        help="Algoritmo especifico a ejecutar (gd, pso, ea, de o todos)")
    parser.add_argument("--funcion", choices=["Rosenbrock", "Rastrigin", "ambas"], default="ambas",
                        help="Funcion de prueba a optimizar")
    parser.add_argument("--dim", choices=[2, 3, 0], type=int, default=0,
                        help="Dimension del problema (2, 3 o 0 para ambas)")
    parser.add_argument("--corridas", type=int, default=None, help="Numero de corridas independientes")
    parser.add_argument("--gif", action="store_true", help="Genera animaciones GIF de GD y PSO en Rosenbrock")
    args = parser.parse_args()

    cfg = cargar_config(parte="parte_1", demo=args.demo)
    n_corridas = args.corridas or (3 if args.demo else cfg.get("n_corridas", 30))

    if args.metodo != "todos":
        funcs = [args.funcion] if args.funcion != "ambas" else ["Rosenbrock", "Rastrigin"]
        punto_configurado = cfg.get("gradiente_descendente", {}).get("punto_inicial")
        if args.metodo == "gd" and punto_configurado is not None:
            punto_configurado = np.asarray(punto_configurado, dtype=float)
            if punto_configurado.ndim != 1 or punto_configurado.size not in (2, 3):
                parser.error("El punto inicial de GD debe tener 2 o 3 coordenadas.")
            if not np.all(np.isfinite(punto_configurado)):
                parser.error("Las coordenadas del punto inicial de GD deben ser finitas.")
            dimension_punto = punto_configurado.size
            if args.dim in (2, 3) and args.dim != dimension_punto:
                parser.error(
                    f"--dim debe ser {dimension_punto} para el punto inicial configurado."
                )
            dims = [dimension_punto] if args.dim == 0 else [args.dim]
        else:
            dims = [args.dim] if args.dim in (2, 3) else [2, 3]
        for f in funcs:
            for d in dims:
                ejecutar_metodo_individual(args.metodo, f, d, cfg, n_corridas=n_corridas)
        return

    # Ejecutar estudio comparativo completo
    _, curvas_conv = ejecutar_estudio_estadistico(cfg, n_corridas=n_corridas)

    # Generar grafica comparativa estatica de convergencia
    generar_grafica_convergencia_parte1(cfg, curvas_convergencia=curvas_conv)

    # Generar animaciones GIF
    if args.gif or args.demo:
        generar_gif_2d("Rosenbrock", "GD", fps=12)
        generar_gif_2d("Rosenbrock", "PSO", fps=12)


if __name__ == "__main__":
    main()
