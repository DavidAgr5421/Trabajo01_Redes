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
        "alpha": 0.00085,
        "umbral_exito": 1e-2,
    },
    "Rastrigin": {
        "func": rastrigin,
        "grad": grad_rastrigin,
        "limites": (-5.12, 5.12),
        "optimo_f": 0.0,
        "optimo_x": lambda d: np.zeros(d),
        "alpha": 0.001,
        "umbral_exito": 1e-2,
    },
}


def ejecutar_metodo(metodo: str, f_info: dict, dim: int, semilla: int, cfg: dict) -> dict:
    """Ejecuta una corrida individual del metodo seleccionado."""
    func = f_info["func"]
    grad = f_info["grad"]
    lb, ub = f_info["limites"]
    rng = np.random.default_rng(semilla)

    if metodo == "GD":
        c_gd = cfg.get("gradiente_descendente", {})
        x0 = rng.uniform(lb, ub, size=dim)
        res = gradient_descent(
            func, grad, x0,
            alpha=f_info["alpha"],
            max_iter=c_gd.get("max_iter", 5000),
            tolerancia=c_gd.get("tolerancia", 1e-4),
        )
        evals_f = res["evaluaciones_f"]
        evals_grad = res["evaluaciones_gradiente"]
        f_val = res["f_final"]
        historial_f = res["historial_f"]
        historial_x = res["historial_x"]

    elif metodo == "PSO":
        c_pso = cfg.get("pso", {})
        res = pso(
            func, dim,
            n_particulas=c_pso.get("n_particulas", 30),
            max_iter=c_pso.get("max_iter", 300),
            limites=(lb, ub),
            w=c_pso.get("w", 0.7),
            c1=c_pso.get("c1", 1.5),
            c2=c_pso.get("c2", 1.5),
            semilla=semilla,
        )
        evals_f = res["evaluaciones_f"]
        evals_grad = 0
        f_val = res["f_final"]
        historial_f = res["historial_f"]
        historial_x = res["historial_x"]

    elif metodo == "EA":
        c_ea = cfg.get("algoritmo_evolutivo", {})
        res = optimizar_algoritmo_evolutivo(
            func, dim, lb, ub,
            n_pop=c_ea.get("poblacion", 40),
            max_generaciones=c_ea.get("generaciones", 80),
            frac_elitismo=c_ea.get("elitismo", 0.15),
            frac_mutacion=c_ea.get("mutacion", 0.2),
            frac_cruzamiento=c_ea.get("cruzamiento", 0.65),
            semilla=semilla,
        )
        evals_f = res["evaluaciones"]
        evals_grad = 0
        f_val = res["mejor_f"]
        historial_f = res["historial_mejor_f"]
        historial_x = [res["mejor_x"]]

    elif metodo == "DE":
        c_de = cfg.get("evolucion_diferencial", {})
        res = optimizar_evolucion_diferencial(
            func, dim, lb, ub,
            n_pop=c_de.get("poblacion", 35),
            max_generaciones=c_de.get("generaciones", 80),
            F=c_de.get("F", 0.7),
            CR=c_de.get("CR", 0.8),
            semilla=semilla,
        )
        evals_f = res["evaluaciones"]
        evals_grad = 0
        f_val = res["mejor_f"]
        historial_f = res["historial_mejor_f"]
        historial_x = [res["mejor_x"]]
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
        "historial_x": historial_x,
    }


def ejecutar_estudio_estadistico(cfg: dict, n_corridas: int = 30) -> pd.DataFrame:
    """Ejecuta las 30 corridas independientes por metodo/funcion/dimension."""
    metodos = ["GD", "PSO", "EA", "DE"]
    filas = []

    print("\n" + "=" * 78)
    print(f"PARTE 1: PROTOCOLO DE COMPARACION JUSTA ({n_corridas} CORRIDAS INDEPENDIENTES)")
    print("=" * 78)

    for nombre_f in cfg.get("funciones_seleccionadas", ["Rosenbrock", "Rastrigin"]):
        f_info = FUNCIONES_INFO[nombre_f]
        for dim in cfg.get("dimensiones", [2, 3]):
            print(f"\n>>> Evaluando {nombre_f} en {dim}D...")
            for met in metodos:
                valores = []
                evals_f_list = []
                evals_grad_list = []
                evals_eq_list = []
                exitos = 0
                t0 = time.time()

                for corrida in range(n_corridas):
                    semilla = cfg.get("semilla_base", 2026) + corrida * 37
                    res = ejecutar_metodo(met, f_info, dim, semilla, cfg)
                    val = res["f_val"]
                    valores.append(val)
                    evals_f_list.append(res["evals_f"])
                    evals_grad_list.append(res["evals_grad"])
                    evals_eq_list.append(res["evals_equiv"])

                    # Criterio de exito
                    if np.isfinite(val) and abs(val - f_info["optimo_f"]) <= f_info["umbral_exito"]:
                        exitos += 1

                dt = time.time() - t0
                val_arr = np.array(valores, dtype=float)
                val_validos = val_arr[np.isfinite(val_arr)]
                media_f = float(np.mean(val_validos)) if len(val_validos) > 0 else np.nan
                std_f = float(np.std(val_validos)) if len(val_validos) > 0 else np.nan
                min_f = float(np.min(val_validos)) if len(val_validos) > 0 else np.nan
                max_f = float(np.max(val_validos)) if len(val_validos) > 0 else np.nan

                tasa_exito = round((exitos / n_corridas) * 100, 2)
                media_eval_eq = int(np.mean(evals_eq_list))

                filas.append({
                    "Funcion": nombre_f,
                    "Dimension": f"{dim}D",
                    "Metodo": met,
                    "Media_f": media_f,
                    "Std_f": std_f,
                    "Mejor_f": min_f,
                    "Peor_f": max_f,
                    "Tasa_Exito_%": tasa_exito,
                    "Media_Evals_f": int(np.mean(evals_f_list)),
                    "Media_Evals_grad": int(np.mean(evals_grad_list)),
                    "Media_Evals_Equiv": media_eval_eq,
                    "Tiempo_Total_s": round(dt, 2),
                })
                print(f"  [{met:3s}] Mejor: {min_f:10.3e} | Media: {media_f:10.3e} | "
                      f"Exito: {tasa_exito:5.1f}% | Evals Eq: {media_eval_eq:6d} ({dt:.2f}s)")

    df = pd.DataFrame(filas)
    csv_path = DIR_RESULTADOS / "resumen_comparativo_parte1.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    json_path = DIR_RESULTADOS / "resumen_comparativo_parte1.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(filas, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 78)
    print(f"Resultados guardados en:")
    print(f"  - CSV : {csv_path}")
    print(f"  - JSON: {json_path}")
    print("=" * 78)
    return df


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


def generar_grafica_convergencia_parte1(cfg: dict):
    """Genera grafica comparativa de convergencia para Parte 1 (Rosenbrock y Rastrigin)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    metodos = ["GD", "PSO", "EA", "DE"]
    colores = {"GD": "#e74c3c", "PSO": "#2ecc71", "EA": "#f39c12", "DE": "#3498db"}

    for idx, nombre_f in enumerate(["Rosenbrock", "Rastrigin"]):
        ax = axes[idx]
        f_info = FUNCIONES_INFO[nombre_f]
        for met in metodos:
            res = ejecutar_metodo(met, f_info, dim=2, semilla=42, cfg=cfg)
            hist = np.asarray(res["historial_f"], dtype=float)
            if len(hist) > 0:
                n_pts = min(len(hist), 300)
                sub_idx = np.linspace(0, len(hist) - 1, n_pts, dtype=int)
                ax.plot(sub_idx + 1, hist[sub_idx], label=f"{met} (final: {res['f_val']:.2e})",
                        color=colores[met], linewidth=2.0)

        ax.set_title(f"Convergencia en {nombre_f} 2D", fontsize=11, fontweight="bold")
        ax.set_xlabel("Iteraciones / Generaciones", fontsize=10)
        ax.set_ylabel("f(x)", fontsize=10)
        if nombre_f == "Rosenbrock":
            ax.set_yscale("log")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(fontsize=9, loc="upper right")

    fig.tight_layout()
    salida = DIR_RESULTADOS / "convergencia_comparativa_parte1.png"
    fig.savefig(salida, dpi=130)
    plt.close(fig)
    print(f"[Parte 1] Grafica comparativa guardada: {salida}")


def main():
    parser = argparse.ArgumentParser(description="Runner Parte 1: Optimizacion Numerica")
    parser.add_argument("--demo", action="store_true", help="Corrida rapida (<15 s)")
    parser.add_argument("--corridas", type=int, default=None, help="Numero de corridas independientes")
    parser.add_argument("--gif", action="store_true", help="Genera animaciones GIF de GD y PSO en Rosenbrock")
    args = parser.parse_args()

    cfg = cargar_config(parte="parte_1", demo=args.demo)
    n_corridas = args.corridas or cfg.get("n_corridas", 30)

    # Ejecutar estudio comparativo
    ejecutar_estudio_estadistico(cfg, n_corridas=n_corridas)

    # Generar grafica comparativa estatica de convergencia
    generar_grafica_convergencia_parte1(cfg)

    # Generar animaciones GIF
    if args.gif or args.demo:
        generar_gif_2d("Rosenbrock", "GD", fps=12)
        generar_gif_2d("Rosenbrock", "PSO", fps=12)


if __name__ == "__main__":
    main()
