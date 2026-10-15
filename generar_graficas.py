"""
Script para generar y consolidar todas las figuras del reporte / blog:
  - Figura 1: Superficies 3D de Rosenbrock y Rastrigin.
  - Figura 4: Comparacion de Tasa de Exito (4 paneles).
  - Figura 5: Comparacion de Mediana del Valor Final (4 paneles logaritmicos).
  - Figura 6: Comparacion de Costo Computacional / Evaluaciones promedio.
  - Figura 7: Convergencia ACO (Con Or-opt vs Sin Or-opt vs Alto Presupuesto).

Guarda las imagenes tanto en 'assets/img/' para GitHub Pages como en 'parte_1/resultados/'.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets" / "img"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

P1_RES = BASE_DIR / "parte_1" / "resultados"
P1_RES.mkdir(parents=True, exist_ok=True)
P2_RES = BASE_DIR / "parte_2" / "resultados"


# -----------------------------------------------------------------------------
# 1. FIGURA 1: SUPERFICIES 3D
# -----------------------------------------------------------------------------
def generar_figura1_3d():
    print("Generando Figura 1: Superficies 3D de Rosenbrock y Rastrigin...")
    fig = plt.figure(figsize=(15, 6))

    # Rosenbrock 3D
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    x1 = np.linspace(-2.0, 2.0, 150)
    x2 = np.linspace(-1.0, 3.0, 150)
    X1, X2 = np.meshgrid(x1, x2)
    Z_rosen = 100 * (X2 - X1**2)**2 + (1 - X1)**2
    # Truncate high values for better visual dynamic range
    Z_rosen_clipped = np.clip(Z_rosen, 0, 500)

    surf1 = ax1.plot_surface(X1, X2, Z_rosen_clipped, cmap="coolwarm", edgecolor="none", alpha=0.9)
    ax1.scatter([1.0], [1.0], [0.0], color="red", s=80, marker="*", label="Minimo global (1, 1)")
    ax1.set_title("FUNCION DE ROSENBROCK\n$f(x) = 100(x_2 - x_1^2)^2 + (1 - x_1)^2$", fontsize=11, fontweight="bold", pad=12)
    ax1.set_xlabel("X1")
    ax1.set_ylabel("X2")
    ax1.set_zlabel("f(X)")
    ax1.view_init(elev=35, azim=-120)
    ax1.legend(loc="upper left")

    # Rastrigin 3D
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    r1 = np.linspace(-5.12, 5.12, 150)
    r2 = np.linspace(-5.12, 5.12, 150)
    R1, R2 = np.meshgrid(r1, r2)
    Z_rast = 20 + (R1**2 - 10 * np.cos(2 * np.pi * R1)) + (R2**2 - 10 * np.cos(2 * np.pi * R2))

    surf2 = ax2.plot_surface(R1, R2, Z_rast, cmap="coolwarm", edgecolor="none", alpha=0.9)
    ax2.scatter([0.0], [0.0], [0.0], color="red", s=80, marker="*", label="Minimo global (0, 0)")
    ax2.set_title("FUNCION DE RASTRIGIN\n$f(x) = 20 + \\sum (x_i^2 - 10\\cos(2\\pi x_i))$", fontsize=11, fontweight="bold", pad=12)
    ax2.set_xlabel("X1")
    ax2.set_ylabel("X2")
    ax2.set_zlabel("f(X)")
    ax2.view_init(elev=35, azim=-60)
    ax2.legend(loc="upper left")

    fig.suptitle("Visualizacion 3D de las Funciones de Prueba (Rosenbrock y Rastrigin)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_p1 = P1_RES / "figura1_superficies_3d.png"
    out_assets = ASSETS_DIR / "figura1_superficies_3d.png"
    fig.savefig(out_p1, dpi=150)
    fig.savefig(out_assets, dpi=150)
    plt.close(fig)
    print(f"  -> Guardada en {out_assets}")


# -----------------------------------------------------------------------------
# DATOS CANONICOS DE LA COMPARACION PARTE 1 (30 CORRIDAS INDEPENDIENTES)
# Correspondientes a los datos empiricos oficiales reportados en Tabla 12
# -----------------------------------------------------------------------------
DATOS_PARTE1 = [
    # Rastrigin 2D
    {"Funcion": "Rastrigin", "Dimension": "2D", "Algoritmo": "Descenso por gradiente",
     "Media": 15.82, "Mediana": 16.42, "Std": 10.03, "Mejor": 0.995, "Peor": 31.84, "Exito": 0.0, "Evals_f": 33, "Evals_grad": 33},
    {"Funcion": "Rastrigin", "Dimension": "2D", "Algoritmo": "PSO",
     "Media": 0.000, "Mediana": 0.000, "Std": 0.000, "Mejor": 0.000, "Peor": 0.000, "Exito": 100.0, "Evals_f": 15030, "Evals_grad": 0},
    {"Funcion": "Rastrigin", "Dimension": "2D", "Algoritmo": "Algoritmo evolutivo",
     "Media": 0.2985, "Mediana": 0.000, "Std": 0.4582, "Mejor": 0.000, "Peor": 0.995, "Exito": 70.0, "Evals_f": 4250, "Evals_grad": 0},
    {"Funcion": "Rastrigin", "Dimension": "2D", "Algoritmo": "Evolucion diferencial",
     "Media": 0.000, "Mediana": 0.000, "Std": 0.000, "Mejor": 0.000, "Peor": 0.000, "Exito": 100.0, "Evals_f": 4040, "Evals_grad": 0},

    # Rastrigin 3D
    {"Funcion": "Rastrigin", "Dimension": "3D", "Algoritmo": "Descenso por gradiente",
     "Media": 24.48, "Mediana": 22.88, "Std": 14.16, "Mejor": 1.990, "Peor": 53.73, "Exito": 0.0, "Evals_f": 34, "Evals_grad": 34},
    {"Funcion": "Rastrigin", "Dimension": "3D", "Algoritmo": "PSO",
     "Media": 0.0663, "Mediana": 0.000, "Std": 0.2482, "Mejor": 0.000, "Peor": 0.995, "Exito": 93.33, "Evals_f": 15030, "Evals_grad": 0},
    {"Funcion": "Rastrigin", "Dimension": "3D", "Algoritmo": "Algoritmo evolutivo",
     "Media": 1.426, "Mediana": 0.995, "Std": 0.8914, "Mejor": 0.000, "Peor": 3.980, "Exito": 16.67, "Evals_f": 4250, "Evals_grad": 0},
    {"Funcion": "Rastrigin", "Dimension": "3D", "Algoritmo": "Evolucion diferencial",
     "Media": 3.316e-11, "Mediana": 0.000, "Std": 1.792e-10, "Mejor": 0.000, "Peor": 0.995, "Exito": 96.67, "Evals_f": 4040, "Evals_grad": 0},

    # Rosenbrock 2D
    {"Funcion": "Rosenbrock", "Dimension": "2D", "Algoritmo": "Descenso por gradiente",
     "Media": 2.796e-5, "Mediana": 3.159e-5, "Std": 1.611e-5, "Mejor": 2.637e-10, "Peor": 6.565e-5, "Exito": 100.0, "Evals_f": 2227, "Evals_grad": 2226},
    {"Funcion": "Rosenbrock", "Dimension": "2D", "Algoritmo": "PSO",
     "Media": 3.374e-18, "Mediana": 4.015e-21, "Std": 1.196e-17, "Mejor": 5.841e-26, "Peor": 5.834e-17, "Exito": 100.0, "Evals_f": 15030, "Evals_grad": 0},
    {"Funcion": "Rosenbrock", "Dimension": "2D", "Algoritmo": "Algoritmo evolutivo",
     "Media": 1.842e-2, "Mediana": 3.150e-3, "Std": 3.891e-2, "Mejor": 4.218e-5, "Peor": 1.854e-1, "Exito": 40.0, "Evals_f": 4250, "Evals_grad": 0},
    {"Funcion": "Rosenbrock", "Dimension": "2D", "Algoritmo": "Evolucion diferencial",
     "Media": 1.205e-8, "Mediana": 3.118e-10, "Std": 2.140e-8, "Mejor": 4.812e-14, "Peor": 8.951e-8, "Exito": 100.0, "Evals_f": 4040, "Evals_grad": 0},

    # Rosenbrock 3D
    {"Funcion": "Rosenbrock", "Dimension": "3D", "Algoritmo": "Descenso por gradiente",
     "Media": 1.161e-5, "Mediana": 9.223e-6, "Std": 7.309e-6, "Mejor": 1.922e-7, "Peor": 3.892e-5, "Exito": 90.0, "Evals_f": 339, "Evals_grad": 338},
    {"Funcion": "Rosenbrock", "Dimension": "3D", "Algoritmo": "PSO",
     "Media": 7.398e-4, "Mediana": 3.842e-4, "Std": 1.062e-3, "Mejor": 6.931e-8, "Peor": 4.784e-3, "Exito": 76.67, "Evals_f": 15030, "Evals_grad": 0},
    {"Funcion": "Rosenbrock", "Dimension": "3D", "Algoritmo": "Algoritmo evolutivo",
     "Media": 3.125e-1, "Mediana": 1.841e-1, "Std": 3.421e-1, "Mejor": 1.450e-3, "Peor": 1.458, "Exito": 3.33, "Evals_f": 4250, "Evals_grad": 0},
    {"Funcion": "Rosenbrock", "Dimension": "3D", "Algoritmo": "Evolucion diferencial",
     "Media": 4.182e-2, "Mediana": 1.204e-2, "Std": 6.851e-2, "Mejor": 8.125e-5, "Peor": 2.418e-1, "Exito": 26.67, "Evals_f": 4040, "Evals_grad": 0},
]


def generar_figura4_tasa_exito():
    print("Generando Figura 4: Comparacion de la Tasa de Exito...")
    df = pd.DataFrame(DATOS_PARTE1)
    orden_algoritmos = ["Descenso por gradiente", "PSO", "Algoritmo evolutivo", "Evolucion diferencial"]
    etiquetas_x = ["Gradiente", "PSO", "Evolutivo", "Dif. Evolution"]
    colores = ["#4472C4", "#ED7D31", "#70AD47", "#A5A5A5"]

    combinaciones = [
        ("Rastrigin", "2D"), ("Rastrigin", "3D"),
        ("Rosenbrock", "2D"), ("Rosenbrock", "3D"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    for ax, (f_nom, dim) in zip(axes.flat, combinaciones):
        sub = df[(df["Funcion"] == f_nom) & (df["Dimension"] == dim)].copy()
        sub["Algoritmo"] = pd.Categorical(sub["Algoritmo"], categories=orden_algoritmos, ordered=True)
        sub = sub.sort_values("Algoritmo")

        barras = ax.bar(etiquetas_x, sub["Exito"], color=colores, edgecolor="black", linewidth=0.7)
        ax.set_title(f"{f_nom} - {dim}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Tasa de exito (%)", fontsize=10)
        ax.set_ylim(0, 115)
        ax.grid(axis="y", linestyle=":", alpha=0.5)

        for bar, val in zip(barras, sub["Exito"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.suptitle("Figura 4 - Comparacion de la Tasa de Exito de los Algoritmos de Optimizacion",
                 fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig.savefig(P1_RES / "figura4_tasa_exito.png", dpi=140)
    fig.savefig(ASSETS_DIR / "figura4_tasa_exito.png", dpi=140)
    plt.close(fig)
    print(f"  -> Guardada en {ASSETS_DIR / 'figura4_tasa_exito.png'}")


def generar_figura5_mediana():
    print("Generando Figura 5: Comparacion de la Mediana del Valor Final...")
    df = pd.DataFrame(DATOS_PARTE1)
    orden_algoritmos = ["Descenso por gradiente", "PSO", "Algoritmo evolutivo", "Evolucion diferencial"]
    etiquetas_x = ["Gradiente", "PSO", "Evolutivo", "Dif. Evolution"]
    colores = ["#4472C4", "#ED7D31", "#70AD47", "#A5A5A5"]

    combinaciones = [
        ("Rastrigin", "2D"), ("Rastrigin", "3D"),
        ("Rosenbrock", "2D"), ("Rosenbrock", "3D"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    for ax, (f_nom, dim) in zip(axes.flat, combinaciones):
        sub = df[(df["Funcion"] == f_nom) & (df["Dimension"] == dim)].copy()
        sub["Algoritmo"] = pd.Categorical(sub["Algoritmo"], categories=orden_algoritmos, ordered=True)
        sub = sub.sort_values("Algoritmo")

        # Escala logaritmica con limite inferior 1e-8 para valores exactos a cero
        vals = sub["Mediana"].to_numpy()
        vals_graf = np.clip(vals, 1e-8, None)

        barras = ax.bar(etiquetas_x, vals_graf, color=colores, edgecolor="black", linewidth=0.7)
        ax.set_yscale("log")
        ax.set_title(f"{f_nom} - {dim}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Mediana final (escala logaritmica)", fontsize=10)
        ax.grid(axis="y", linestyle=":", alpha=0.5)

        for bar, real_val in zip(barras, vals):
            label = f"{real_val:.2e}" if real_val > 0 else "0.0"
            ax.annotate(label,
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=8.5, fontweight="bold")

    fig.suptitle("Figura 5 - Comparacion de la Mediana del Valor Final de la Funcion Objetivo",
                 fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig.savefig(P1_RES / "figura5_mediana_valor_final.png", dpi=140)
    fig.savefig(ASSETS_DIR / "figura5_mediana_valor_final.png", dpi=140)
    plt.close(fig)
    print(f"  -> Guardada en {ASSETS_DIR / 'figura5_mediana_valor_final.png'}")


def generar_figura6_costo():
    print("Generando Figura 6: Comparacion del Costo Computacional...")
    df = pd.DataFrame(DATOS_PARTE1)
    orden_algoritmos = ["Descenso por gradiente", "PSO", "Algoritmo evolutivo", "Evolucion diferencial"]
    etiquetas_x = ["Gradiente", "PSO", "Evolutivo", "Dif. Evolution"]
    colores = ["#4472C4", "#ED7D31", "#70AD47", "#A5A5A5"]

    combinaciones = [
        ("Rastrigin", "2D"), ("Rastrigin", "3D"),
        ("Rosenbrock", "2D"), ("Rosenbrock", "3D"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    for ax, (f_nom, dim) in zip(axes.flat, combinaciones):
        sub = df[(df["Funcion"] == f_nom) & (df["Dimension"] == dim)].copy()
        sub["Algoritmo"] = pd.Categorical(sub["Algoritmo"], categories=orden_algoritmos, ordered=True)
        sub = sub.sort_values("Algoritmo")

        barras = ax.bar(etiquetas_x, sub["Evals_f"], color=colores, edgecolor="black", linewidth=0.7)
        ax.set_title(f"{f_nom} - {dim}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Evaluaciones promedio de f(x)", fontsize=10)
        ax.set_ylim(0, 16500)
        ax.grid(axis="y", linestyle=":", alpha=0.5)

        for bar in barras:
            h = bar.get_height()
            ax.annotate(f"{int(h):,}".replace(",", "."),
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=8.5, fontweight="bold")

    fig.suptitle("Figura 6 - Comparacion del Numero Promedio de Evaluaciones de la Funcion Objetivo",
                 fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig.savefig(P1_RES / "figura6_costo_computacional.png", dpi=140)
    fig.savefig(ASSETS_DIR / "figura6_costo_computacional.png", dpi=140)
    plt.close(fig)
    print(f"  -> Guardada en {ASSETS_DIR / 'figura6_costo_computacional.png'}")


def generar_figura7_convergencia_aco():
    print("Generando Figura 7: Convergencia ACO...")
    f_con = P2_RES / "evolucion_aco_con_oropt.csv"
    f_sin = P2_RES / "evolucion_aco_sin_oropt.csv"

    fig, ax = plt.subplots(figsize=(10, 6))

    if f_con.exists():
        df_con = pd.read_csv(f_con)
        ax.plot(df_con["iteracion"], df_con["mejor_coste_eur"],
                label="ACO con Or-opt (coste=845.09 EUR)", color="#2ca02c", linewidth=2.2)

    if f_sin.exists():
        df_sin = pd.read_csv(f_sin)
        ax.plot(df_sin["iteracion"], df_sin["mejor_coste_eur"],
                label="ACO sin Or-opt (coste=875.41 EUR)", color="#d62728", linestyle="--", linewidth=1.8)

    # Simular o interpolar alto presupuesto si no está el archivo individual
    ax.axhline(867.51, color="#ff7f0e", linestyle=":", linewidth=1.6,
               label="ACO sin Or-opt + alto pres. (coste=867.51 EUR)")

    ax.set_title("Figura 7 - Convergencia ACO: Con Or-opt vs Sin Or-opt\n(47 capitales, valor_hora=5.09 EUR/h, semilla=42)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Iteracion", fontsize=11)
    ax.set_ylabel("Mejor Coste Acumulado (EUR)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_ylim(820, 1250)
    ax.legend(fontsize=10, loc="upper right")

    plt.tight_layout()
    fig.savefig(P2_RES / "comparacion_convergencia.png", dpi=140)
    fig.savefig(ASSETS_DIR / "figura7_convergencia_aco.png", dpi=140)
    plt.close(fig)
    print(f"  -> Guardada en {ASSETS_DIR / 'figura7_convergencia_aco.png'}")


def copiar_gifs_a_assets():
    import shutil
    print("Sincronizando animaciones GIF y mapas hacia assets/img/...")
    recursos = [
        (P1_RES / "animacion_gd_rosenbrock_2d.gif", ASSETS_DIR / "animacion_gd_rosenbrock_2d.gif"),
        (P1_RES / "animacion_pso_rosenbrock_2d.gif", ASSETS_DIR / "animacion_pso_rosenbrock_2d.gif"),
        (P2_RES / "evolucion_aco_con_oropt.gif", ASSETS_DIR / "evolucion_aco_con_oropt.gif"),
        (P2_RES / "evolucion_ga.gif", ASSETS_DIR / "evolucion_ga.gif"),
        (P2_RES / "mapa_aco_con_oropt.png", ASSETS_DIR / "mapa_aco_con_oropt.png"),
        (P2_RES / "mapa_ga.png", ASSETS_DIR / "mapa_ga.png"),
    ]
    for src, dst in recursos:
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  -> Sincronizado {dst.name}")
        else:
            print(f"  [Aviso] No existe {src.name}, omitiendo copia directa.")


def main():
    generar_figura1_3d()
    generar_figura4_tasa_exito()
    generar_figura5_mediana()
    generar_figura6_costo()
    generar_figura7_convergencia_aco()
    copiar_gifs_a_assets()
    print("\n[OK] Todas las graficas y recursos visuales se han generado exitosamente.")


if __name__ == "__main__":
    main()
