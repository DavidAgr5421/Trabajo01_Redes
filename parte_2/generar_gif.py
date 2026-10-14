"""
GIF animado con la evolucion de la mejor solucion del GA sobre el mapa.

Ejecuta el algoritmo genetico registrando la mejor ruta cada cierto numero
de generaciones y dibuja el recorrido sobre un contorno simplificado de la
peninsula (embebido en el propio archivo; no se descarga nada en tiempo de
ejecucion).

Salida por defecto: resultados/evolucion_ga.gif

Uso:
  python3 generar_gif.py
  python3 generar_gif.py --valor-hora 20 --cada 4 --fps 8
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from costes_comunes import BASE_DIR, VALOR_HORA_POR_DEFECTO, cargar_costes

DIR_RESULTADOS = BASE_DIR / "resultados"

# Contorno simplificado de la peninsula iberica (Espana peninsular), en
# (longitud, latitud). Fuente: geo.json publico de fronteras (johan/world.geo.json,
# ESP), simplificado a 51 vertices. Embebido para no depender de descargas.
CONTORNO = np.array([
    [-9.0348, 41.8806], [-8.9844, 42.5928], [-9.3929, 43.0266], [-7.9782, 43.7483],
    [-6.7545, 43.5679], [-5.4119, 43.5742], [-4.3478, 43.4034], [-3.5175, 43.4559],
    [-1.9014, 43.4228], [-1.5028, 43.0340], [0.3380, 42.5795], [0.7016, 42.7957],
    [1.8268, 42.3434], [2.9860, 42.4730], [3.0395, 41.8921], [2.0918, 41.2261],
    [0.8105, 41.0147], [0.7213, 40.6783], [0.1067, 40.1239], [-0.2787, 39.3100],
    [0.1113, 38.7385], [-0.4671, 38.2924], [-0.6834, 37.6424], [-1.4384, 37.4431],
    [-2.1465, 36.6741], [-3.4158, 36.6589], [-4.3689, 36.6778], [-4.9952, 36.3247],
    [-5.3772, 35.9468], [-5.8664, 36.0298], [-6.2367, 36.3677], [-6.5202, 36.9429],
    [-7.4537, 37.0978], [-7.5371, 37.4289], [-7.1665, 37.8039], [-7.0293, 38.0758],
    [-7.3741, 38.3731], [-7.0980, 39.0301], [-7.4986, 39.6296], [-7.0666, 39.7119],
    [-7.0264, 40.1845], [-6.8640, 40.3309], [-6.8511, 41.1111], [-6.3891, 41.3818],
    [-6.6686, 41.8834], [-7.2513, 41.9183], [-7.4225, 41.7921], [-8.0132, 41.7909],
    [-8.2639, 42.2805], [-8.6719, 42.1347], [-9.0348, 41.8806],
])


def construir_animacion(datos, valor_hora, *, algoritmo="aco",
                        semilla=42, cada=2, fps=8,
                        hormigas=20, iteraciones=150,
                        alfa=1.0, beta=3.0, evaporacion=0.2,
                        usar_or_opt=True):
    M = datos.matriz_coste(valor_hora)
    registro = []
    from colonia_hormigas import ejecutar_aco
    resultado = ejecutar_aco(
        M, hormigas=hormigas, iteraciones=iteraciones,
        alfa=alfa, beta=beta, evaporacion=evaporacion,
        semilla=semilla, usar_or_opt=usar_or_opt,
        registro=registro, grabar_cada=cada,
    )
    total = iteraciones
    etiqueta = "ACO (con Or-opt)" if usar_or_opt else "ACO (sin Or-opt)"

    lons = datos.coords.loc[datos.ciudades, "lon"].to_numpy(dtype=float)
    lats = datos.coords.loc[datos.ciudades, "lat"].to_numpy(dtype=float)
    inicio = int(resultado["tour"][0])

    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    ax.fill(CONTORNO[:, 0], CONTORNO[:, 1], color="#f0efe9",
            edgecolor="#7d7d7d", linewidth=1.2, zorder=1)
    ax.scatter(lons, lats, s=16, color="#1f3b73", zorder=4,
               label="Capitales (47)")
    ax.scatter([lons[inicio]], [lats[inicio]], s=120, color="#0a7d32",
               marker="*", zorder=6, label="Ciudad de salida")
    ax.annotate(datos.ciudades[inicio], (lons[inicio], lats[inicio]),
                textcoords="offset points", xytext=(6, 4), fontsize=9,
                color="#0a7d32", zorder=7)

    (linea,) = ax.plot([], [], color="#c0392b", linewidth=2.0,
                       alpha=0.9, zorder=5, label="Mejor recorrido")
    titulo = ax.set_title("", fontsize=12)
    ax.set_xlim(CONTORNO[:, 0].min() - 0.6, CONTORNO[:, 0].max() + 0.6)
    ax.set_ylim(CONTORNO[:, 1].min() - 0.6, CONTORNO[:, 1].max() + 0.6)
    ax.set_aspect("equal")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()

    def actualizar(k):
        gen, coste, tour = registro[k]
        lons_tour = lons[tour]
        lats_tour = lats[tour]
        xs = np.concatenate([lons_tour, lons_tour[:1]])
        ys = np.concatenate([lats_tour, lats_tour[:1]])
        linea.set_data(xs, ys)
        titulo.set_text(
            f"Mejor solucion {etiqueta}  |  iter {gen}/{total}  |  "
            f"coste {coste:,.2f} EUR  (valor hora {valor_hora:.0f} EUR/h)"
        )
        return linea, titulo

    animacion = FuncAnimation(fig, actualizar, frames=len(registro),
                             interval=1000 / fps, blit=False)
    return fig, animacion, registro, resultado


def _main():
    parser = argparse.ArgumentParser(description="GIF y mapa de la evolucion de Colonia de Hormigas (ACO)")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--valor-hora", type=float, default=None)
    parser.add_argument("--hormigas", type=int, default=None)
    parser.add_argument("--iteraciones", type=int, default=None)
    parser.add_argument("--alfa", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--evaporacion", type=float, default=None)
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--sin-or-opt", action="store_true",
                        help="Desactiva la busqueda local Or-opt")
    parser.add_argument("--cada", type=int, default=None,
                        help="fotogramas: 1 por cada N iteraciones")
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--salida", type=Path, default=None)
    args = parser.parse_args()

    from config import cargar_config
    cfg = cargar_config(args.config, demo=args.demo)
    aco_c, gif_c = cfg.get("aco", {}), cfg.get("gif", {})
    vh = (args.valor_hora if args.valor_hora is not None
          else cfg.get("valor_hora", VALOR_HORA_POR_DEFECTO))
    cada = args.cada or gif_c.get("cada", 2)
    fps = args.fps or gif_c.get("fps", 8)
    usar_or_opt = aco_c.get("usar_or_opt", True) and not args.sin_or_opt
    semilla = args.semilla or aco_c.get("semilla", 42)

    sufijo = "con_oropt" if usar_or_opt else "sin_oropt"
    if args.salida:
        salida = args.salida
        png = salida.with_suffix(".png")
    else:
        salida = DIR_RESULTADOS / f"evolucion_aco_{sufijo}.gif"
        png = DIR_RESULTADOS / f"mapa_aco_{sufijo}.png"

    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    datos = cargar_costes()
    print(f"Ejecutando ACO ({'con Or-opt' if usar_or_opt else 'sin Or-opt'}) "
          f"({datos.n} ciudades, valor_hora={vh:.2f} EUR/h, semilla={semilla})...")

    fig, animacion, registro, resultado = construir_animacion(
        datos, vh, algoritmo="aco",
        semilla=semilla,
        cada=cada, fps=fps,
        hormigas=args.hormigas or aco_c.get("hormigas", 20),
        iteraciones=args.iteraciones or aco_c.get("iteraciones", 150),
        alfa=args.alfa or aco_c.get("alfa", 1.0),
        beta=args.beta or aco_c.get("beta", 3.0),
        evaporacion=(args.evaporacion if args.evaporacion is not None
                     else aco_c.get("evaporacion", 0.2)),
        usar_or_opt=usar_or_opt,
    )

    print(f"Fotogramas: {len(registro)}")
    animacion.save(str(salida), writer=PillowWriter(fps=fps),
                   dpi=110)
    fig.savefig(png, dpi=140)
    plt.close(fig)
    print(f"Mejor coste final: {resultado['coste']:.2f} EUR")
    print(f"GIF: {salida}")


if __name__ == "__main__":
    _main()
