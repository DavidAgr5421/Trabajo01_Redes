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
# CARGA DINAMICA DE LA COMPARACION PARTE 1 (IGUALDAD DE ESFUERZO, 30 CORRIDAS)
# Fuente unica: parte_1/resultados/resumen_comparativo_parte1.json
# -----------------------------------------------------------------------------
MAPA_NOMBRES_P1 = {
    "GD": "Descenso por gradiente",
    "PSO": "PSO",
    "EA": "Algoritmo evolutivo",
    "DE": "Evolucion diferencial",
}


def cargar_datos_parte1() -> pd.DataFrame:
    ruta_json = P1_RES / "resumen_comparativo_parte1.json"
    if not ruta_json.exists():
        raise FileNotFoundError(
            f"No existe {ruta_json}. Ejecute primero 'python main.py -e parte1'."
        )
    with open(ruta_json, encoding="utf-8") as f:
        filas = json.load(f)
    df = pd.DataFrame(filas)
    df["Algoritmo"] = df["Metodo"].map(MAPA_NOMBRES_P1)
    df["Exito"] = df["Tasa_Exito_%"].astype(float)
    df["Mediana"] = df["Mediana_f"].astype(float)
    df["Evals_f"] = df["Media_Evals_f"].astype(int)
    df["Evals_grad"] = df["Media_Evals_grad"].astype(int)
    df["Evals_Equiv"] = df["Media_Evals_Equiv"].astype(int)
    df["Presupuesto"] = df["Presupuesto_Equiv"].astype(int) if "Presupuesto_Equiv" in df else 15000
    return df


def generar_figura4_tasa_exito():
    print("Generando Figura 4: Comparacion de la Tasa de Exito...")
    df = cargar_datos_parte1()
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

    presupuesto = int(df["Presupuesto"].iloc[0])
    fig.suptitle(
        f"Figura 4 - Tasa de Exito en Igualdad de Esfuerzo ($E_{{\\mathrm{{equiv}}}} \\leq {presupuesto:,}$)",
        fontsize=13, fontweight="bold", y=0.98,
    )
    plt.tight_layout()
    fig.savefig(P1_RES / "figura4_tasa_exito.png", dpi=140)
    fig.savefig(ASSETS_DIR / "figura4_tasa_exito.png", dpi=140)
    plt.close(fig)
    print(f"  -> Guardada en {ASSETS_DIR / 'figura4_tasa_exito.png'}")


def generar_figura5_mediana():
    print("Generando Figura 5: Comparacion de la Mediana del Valor Final...")
    df = cargar_datos_parte1()
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

        # Escala logaritmica con limite inferior 1e-26 para valores muy pequenos o cero
        vals = sub["Mediana"].to_numpy()
        vals_graf = np.clip(np.where(np.isfinite(vals), vals, 1e-26), 1e-26, None)

        barras = ax.bar(etiquetas_x, vals_graf, color=colores, edgecolor="black", linewidth=0.7)
        ax.set_yscale("log")
        ax.set_ylim(1e-27, 1e4)
        ax.set_title(f"{f_nom} - {dim}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Mediana final (escala logaritmica)", fontsize=10)
        ax.grid(axis="y", linestyle=":", alpha=0.5)

        for bar, real_val in zip(barras, vals):
            if not np.isfinite(real_val):
                label = "div."
            elif real_val == 0:
                label = "0.0"
            else:
                label = f"{real_val:.2e}"
            ax.annotate(label,
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=8.5, fontweight="bold")

    presupuesto = int(df["Presupuesto"].iloc[0])
    fig.suptitle(
        f"Figura 5 - Mediana del Valor Final en Igualdad de Esfuerzo ($E_{{\\mathrm{{equiv}}}} \\leq {presupuesto:,}$)",
        fontsize=13, fontweight="bold", y=0.98,
    )
    plt.tight_layout()
    fig.savefig(P1_RES / "figura5_mediana_valor_final.png", dpi=140)
    fig.savefig(ASSETS_DIR / "figura5_mediana_valor_final.png", dpi=140)
    plt.close(fig)
    print(f"  -> Guardada en {ASSETS_DIR / 'figura5_mediana_valor_final.png'}")


def generar_figura6_costo():
    print("Generando Figura 6: Comparacion del Costo Computacional Equivalente...")
    df = cargar_datos_parte1()
    orden_algoritmos = ["Descenso por gradiente", "PSO", "Algoritmo evolutivo", "Evolucion diferencial"]
    etiquetas_x = ["Gradiente", "PSO", "Evolutivo", "Dif. Evolution"]
    colores = ["#4472C4", "#ED7D31", "#70AD47", "#A5A5A5"]

    combinaciones = [
        ("Rastrigin", "2D"), ("Rastrigin", "3D"),
        ("Rosenbrock", "2D"), ("Rosenbrock", "3D"),
    ]

    presupuesto = int(df["Presupuesto"].iloc[0])
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    for ax, (f_nom, dim) in zip(axes.flat, combinaciones):
        sub = df[(df["Funcion"] == f_nom) & (df["Dimension"] == dim)].copy()
        sub["Algoritmo"] = pd.Categorical(sub["Algoritmo"], categories=orden_algoritmos, ordered=True)
        sub = sub.sort_values("Algoritmo")

        barras = ax.bar(etiquetas_x, sub["Evals_Equiv"], color=colores, edgecolor="black", linewidth=0.7)
        ax.axhline(presupuesto, color="#c0392b", linestyle="--", linewidth=1.4,
                   label=f"Tope comun ({presupuesto:,} E_equiv)")
        ax.set_title(f"{f_nom} - {dim}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Evals equivalentes promedio ($E_{equiv}$)", fontsize=10)
        ax.set_ylim(0, presupuesto * 1.15)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.legend(fontsize=8.5, loc="upper left")

        for bar in barras:
            h = bar.get_height()
            ax.annotate(f"{int(h):,}".replace(",", "."),
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=8.5, fontweight="bold")

    fig.suptitle(
        "Figura 6 - Evaluaciones Equivalentes ($E_{\\mathrm{equiv}} = E_f + 2n\\,E_{\\nabla}$) bajo Presupuesto Comun",
        fontsize=13, fontweight="bold", y=0.98,
    )
    plt.tight_layout()
    fig.savefig(P1_RES / "figura6_costo_computacional.png", dpi=140)
    fig.savefig(ASSETS_DIR / "figura6_costo_computacional.png", dpi=140)
    plt.close(fig)
    print(f"  -> Guardada en {ASSETS_DIR / 'figura6_costo_computacional.png'}")


def generar_figura7_convergencia_aco():
    print("Generando Figura 7: Convergencia ACO...")
    f_con = P2_RES / "evolucion_aco_con_oropt.csv"
    f_sin = P2_RES / "evolucion_aco_sin_oropt.csv"
    f_comp = P2_RES / "comparacion_oropt.json"

    comp = {}
    if f_comp.exists():
        with open(f_comp, encoding="utf-8") as f:
            comp = json.load(f)

    vh = float(comp.get("valor_hora_eur", 5.09))
    semilla = int(comp.get("semilla", 42))

    fig, ax = plt.subplots(figsize=(10, 6))
    y_vals = []

    if f_con.exists():
        df_con = pd.read_csv(f_con)
        coste_con = comp.get("con_or_opt", {}).get("coste_eur")
        if coste_con is None and len(df_con) > 0:
            coste_con = float(df_con["mejor_coste_eur"].iloc[-1])
        lbl_con = (
            f"ACO con Or-opt (coste={float(coste_con):.2f} EUR)"
            if coste_con is not None else "ACO con Or-opt"
        )
        ax.plot(df_con["iteracion"], df_con["mejor_coste_eur"],
                label=lbl_con, color="#2ca02c", linewidth=2.2)
        y_vals.extend(df_con["mejor_coste_eur"].astype(float).tolist())

    if f_sin.exists():
        df_sin = pd.read_csv(f_sin)
        coste_sin = comp.get("sin_or_opt", {}).get("coste_eur")
        if coste_sin is None and len(df_sin) > 0:
            coste_sin = float(df_sin["mejor_coste_eur"].iloc[-1])
        lbl_sin = (
            f"ACO sin Or-opt (coste={float(coste_sin):.2f} EUR)"
            if coste_sin is not None else "ACO sin Or-opt"
        )
        ax.plot(df_sin["iteracion"], df_sin["mejor_coste_eur"],
                label=lbl_sin, color="#d62728", linestyle="--", linewidth=1.8)
        y_vals.extend(df_sin["mejor_coste_eur"].astype(float).tolist())

    alto_pres = comp.get("sin_or_opt_alto_presupuesto")
    if isinstance(alto_pres, dict) and "coste_eur" in alto_pres:
        coste_alto = float(alto_pres["coste_eur"])
        ax.axhline(coste_alto, color="#ff7f0e", linestyle=":", linewidth=1.6,
                   label=f"ACO sin Or-opt + alto pres. (coste={coste_alto:.2f} EUR)")
        y_vals.append(coste_alto)

    ax.set_title(
        f"Figura 7 - Convergencia ACO: Con Or-opt vs Sin Or-opt\n"
        f"(47 capitales, valor_hora={vh:.2f} EUR/h, semilla={semilla})",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Iteracion", fontsize=11)
    ax.set_ylabel("Mejor Coste Acumulado (EUR)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    if y_vals:
        y_min, y_max = min(y_vals), max(y_vals)
        pad = max(20.0, 0.06 * (y_max - y_min))
        ax.set_ylim(y_min - pad, y_max + pad)
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
        (P1_RES / "convergencia_comparativa_parte1.png", ASSETS_DIR / "convergencia_comparativa_parte1.png"),
        (P2_RES / "evolucion_aco_con_oropt.gif", ASSETS_DIR / "evolucion_aco_con_oropt.gif"),
        (P2_RES / "evolucion_ga.gif", ASSETS_DIR / "evolucion_ga.gif"),
        (P2_RES / "mapa_aco_con_oropt.png", ASSETS_DIR / "mapa_aco_con_oropt.png"),
        (P2_RES / "mapa_ga.png", ASSETS_DIR / "mapa_ga.png"),
        (P2_RES / "comparacion_convergencia_aco_ga.png", ASSETS_DIR / "comparacion_convergencia_aco_ga.png"),
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
