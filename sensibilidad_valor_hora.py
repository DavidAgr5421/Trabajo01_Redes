"""
Sensibilidad de la ruta optima al valor de la hora del vendedor.

El coste de cada arco es afin en el parametro v (EUR/h):

    C_ij(v) = v * tiempo_ij + (tax_ij + gas_ij) = v * T_ij + B_ij

Por tanto el coste de un ciclo tambien es afin: coste(tour) = v*T + B, con
T la suma de tiempos y B la suma de peajes+combustible del ciclo. La ruta
optima es, por tanto, una funcion escalonada de v cuyos puntos de cambio se
localizan exactamente como intersecciones entre los tours candidatos.

Estrategia:
  1. Barrido de v (rejilla) re-ejecutando el GA con warm start (cada
     ejecucion parte de la mejor ruta del valor anterior).
  2. Se recogen todos los tours distintos encontrados y sus pares (T, B).
  3. Se calcula la envolvente inferior exacta de las rectas coste(v)=T*v+B
     sobre [v_min, v_max] (convex hull trick), obteniendo los tramos y los
     valores exactos de v donde cambia la ruta optima.

Salidas en resultados/:
  sensibilidad_valor_hora.csv   coste y metricas en cada v muestreado
  tramos_optimos.csv            intervalos [v_inf,v_sup] y ruta optima
  rutas_optimas.csv             secuencia de cada ruta optima distinta
  sensibilidad_valor_hora.png   grafica coste vs valor_hora
  resumen_sensibilidad.json     cambios detectados y totales

Uso:
  python3 sensibilidad_valor_hora.py
  python3 sensibilidad_valor_hora.py --desde 0 --hasta 100 --paso 1
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from costes_comunes import BASE_DIR, cargar_costes
from algoritmo_genetico import ejecutar_ga, or_opt, vecino_mas_cercano

DIR_RESULTADOS = BASE_DIR / "resultados"
EPS = 1e-7


def barrido(datos, valores, *, tam_poblacion, generaciones, semilla, verbose=True):
    """Ejecuta el GA con warm start en cada valor y guarda los candidatos."""
    candidatos = []
    registros = []
    mejor_previo = None
    for v in valores:
        M = datos.matriz_coste(float(v))
        iniciales = [mejor_previo] if mejor_previo is not None else None
        resultado = ejecutar_ga(
            M,
            tam_poblacion=tam_poblacion,
            generaciones=generaciones,
            semilla=semilla,
            tours_iniciales=iniciales,
        )
        tour = resultado["tour"]
        firma = datos.firma(tour)
        T, B = datos.componentes_tour(tour)
        desglose = datos.desglose_tour(tour, float(v))
        candidatos.append((T, B, firma))
        registros.append({
            "valor_hora": float(v),
            "coste_eur": desglose["coste_eur"],
            "distancia_km": desglose["distancia_km"],
            "tiempo_h": desglose["tiempo_h"],
            "tax_eur": desglose["tax_eur"],
            "gas_eur": desglose["gas_eur"],
            "firma": firma,
        })
        mejor_previo = tour
        if verbose and (len(registros) % 10 == 0):
            print(f"  v={v:.1f}: coste={desglose['coste_eur']:.2f}")
    return candidatos, registros, mejor_previo


def _cruce_x(l1, l2):
    """Abscisa donde l1 y l2 se cruzan (l1.T > l2.T, l1 activa a la izquierda)."""
    return (l1[1] - l2[1]) / (l2[0] - l1[0])


def _mejor_tour_en(v, tramos):
    for x_inf, x_sup, linea in tramos:
        if x_inf - EPS <= v <= x_sup + EPS:
            return np.asarray(linea[2], dtype=np.int64)
    return None


def refinar(datos, candidatos, tramos, puntos, *, tam_poblacion, generaciones,
            semillas):
    """Re-ejecuta un GA fuerte en los puntos criticos para hallar tours que
    dominen la envolvente actual. Devuelve el numero de candidatos nuevos."""
    conocidas = {c[2] for c in candidatos}
    nuevos = 0
    for v in puntos:
        M = datos.matriz_coste(float(v))
        iniciales = [_mejor_tour_en(v, tramos)]
        for semilla in semillas:
            resultado = ejecutar_ga(
                M, tam_poblacion=tam_poblacion, generaciones=generaciones,
                semilla=semilla, tours_iniciales=iniciales,
            )
            T, B = datos.componentes_tour(resultado["tour"])
            firma = datos.firma(resultado["tour"])
            if firma not in conocidas:
                conocidas.add(firma)
                candidatos.append((T, B, firma))
                nuevos += 1
    return nuevos


def envolvente_inferior(lineas, x_min, x_max):
    """Envolvente inferior de rectas y=T*x+B con T,B finitos.

    'lineas' es una lista de (T, B, firma). Devuelve
    (tramos, puntos_cambio) donde tramos es una lista de
    (x_inf, x_sup, (T, B, firma)) y puntos_cambio los v exactos.
    """
    unicas = {}
    for T, B, firma in lineas:
        clave = round(T, 9)
        if clave not in unicas or B < unicas[clave][1]:
            unicas[clave] = (T, B, firma)
    ordenadas = sorted(unicas.values(), key=lambda l: (-l[0], l[1]))

    pila = []
    for linea in ordenadas:
        while len(pila) >= 2 and _cruce_x(pila[-2], pila[-1]) >= _cruce_x(
            pila[-1], linea
        ):
            pila.pop()
        pila.append(linea)

    cruces = [-np.inf]
    for k in range(1, len(pila)):
        cruces.append(_cruce_x(pila[k - 1], pila[k]))
    cruces.append(np.inf)

    tramos = []
    puntos_cambio = []
    for k, linea in enumerate(pila):
        x_inf = max(cruces[k], x_min)
        x_sup = min(cruces[k + 1], x_max)
        if x_inf > x_sup + EPS:
            continue
        tramos.append((x_inf, x_sup, linea))
        if k + 1 < len(pila) and x_min < cruces[k + 1] < x_max:
            puntos_cambio.append(cruces[k + 1])
    return tramos, puntos_cambio


def _evaluar_linea(v, linea):
    return linea[0] * v + linea[1]


def guardar_salidas(datos, registros, candidatos, tramos, puntos_cambio,
                    directorio=DIR_RESULTADOS):
    directorio = Path(directorio)
    directorio.mkdir(parents=True, exist_ok=True)

    firmas = {}
    for reg in registros:
        firmas.setdefault(reg["firma"], len(firmas) + 1)
    firmas_optimas = {}
    for _, _, linea in tramos:
        firmas_optimas[linea[2]] = linea
        firmas.setdefault(linea[2], len(firmas) + 1)

    with open(directorio / "sensibilidad_valor_hora.csv", "w",
              encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["valor_hora", "coste_eur", "distancia_km", "tiempo_h",
                    "tax_eur", "gas_eur", "firma_id"])
        for reg in registros:
            w.writerow([f"{reg['valor_hora']:.4f}",
                        f"{reg['coste_eur']:.4f}",
                        f"{reg['distancia_km']:.3f}",
                        f"{reg['tiempo_h']:.6f}",
                        f"{reg['tax_eur']:.2f}",
                        f"{reg['gas_eur']:.2f}", firmas[reg["firma"]]])

    with open(directorio / "tramos_optimos.csv", "w",
              encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tramo", "valor_hora_inf", "valor_hora_sup",
                    "firma_id", "suma_tiempo_h", "suma_base_eur",
                    "coste_inf_eur", "coste_sup_eur"])
        for k, (x_inf, x_sup, linea) in enumerate(tramos):
            w.writerow([k + 1, f"{x_inf:.4f}",
                        "inf" if np.isinf(x_sup) else f"{x_sup:.4f}",
                        firmas.get(linea[2], "?"),
                        f"{linea[0]:.6f}", f"{linea[1]:.2f}",
                        f"{_evaluar_linea(x_inf, linea):.4f}",
                        "inf" if np.isinf(x_sup)
                        else f"{_evaluar_linea(x_sup, linea):.4f}"])

    with open(directorio / "rutas_optimas.csv", "w",
              encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["firma_id", "paso", "ciudad", "lat", "lon"])
        for firma, linea in firmas_optimas.items():
            fid = firmas.get(firma, "?")
            tour = datos.firma_a_tour(firma)
            for paso, idx in enumerate(tour, start=1):
                ciudad = datos.ciudades[idx]
                lat, lon = datos.coords.loc[ciudad, ["lat", "lon"]]
                w.writerow([fid, paso, ciudad, f"{lat:.6f}", f"{lon:.6f}"])

    _graficar(datos, registros, candidatos, tramos, puntos_cambio, directorio)

    resumen = {
        "n_valores_muestreados": len(registros),
        "n_rutas_optimas_distintas": len(firmas_optimas),
        "valores_cambio_eur_h": [round(x, 4) for x in puntos_cambio],
        "tramos": [
            {
                "tramo": k + 1,
                "valor_hora_inf": round(x_inf, 4),
                "valor_hora_sup": (None if np.isinf(x_sup)
                                   else round(x_sup, 4)),
                "firma_id": firmas.get(linea[2], "?"),
                "suma_tiempo_h": round(linea[0], 6),
                "suma_base_eur": round(linea[1], 4),
            }
            for k, (x_inf, x_sup, linea) in enumerate(tramos)
        ],
        "nota": "Tramos exactos de la envolvente inferior de los tours "
                "candidatos; los valores de cambio son intersecciones.",
    }
    with open(directorio / "resumen_sensibilidad.json", "w",
              encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)
    return resumen, firmas


def _graficar(datos, registros, candidatos, tramos, puntos_cambio, directorio):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    v_vals = np.array([r["valor_hora"] for r in registros])
    c_vals = np.array([r["coste_eur"] for r in registros])
    ax.scatter(v_vals, c_vals, s=14, color="tab:blue", zorder=3,
               label="Optimo GA en cada valor")

    x_min = float(v_vals.min())
    x_max = float(v_vals.max())
    x_linea = np.linspace(x_min, x_max, 400)
    lineas_unicas = {}
    for T, B, firma in candidatos:
        lineas_unicas[(round(T, 9), round(B, 4))] = (T, B)
    for T, B in lineas_unicas.values():
        ax.plot(x_linea, T * x_linea + B, color="gray", lw=0.6, alpha=0.35)

    for x_inf, x_sup, linea in tramos:
        xs = np.linspace(x_inf, x_sup, 200)
        ax.plot(xs, linea[0] * xs + linea[1], color="tab:red", lw=2.4,
                zorder=4, label="Envolvente inferior" if x_inf == tramos[0][0]
                else None)

    for xc in puntos_cambio:
        ax.axvline(xc, color="tab:green", ls="--", lw=1.2, alpha=0.8)
        ax.annotate(f"{xc:.2f}", (xc, ax.get_ylim()[0]),
                    textcoords="offset points", xytext=(3, 6),
                    color="tab:green", fontsize=9, rotation=90)

    ax.set_xlabel("Valor de la hora del vendedor (EUR/h)")
    ax.set_ylabel("Coste del recorrido (EUR)")
    ax.set_title("Sensibilidad de la ruta optima al valor de la hora")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(directorio / "sensibilidad_valor_hora.png", dpi=140)
    plt.close(fig)


def _main():
    parser = argparse.ArgumentParser(
        description="Sensibilidad de la ruta optima al valor de la hora"
    )
    parser.add_argument("--desde", type=float, default=0.0)
    parser.add_argument("--hasta", type=float, default=100.0)
    parser.add_argument("--paso", type=float, default=1.0)
    parser.add_argument("--poblacion", type=int, default=40)
    parser.add_argument("--generaciones", type=int, default=40)
    parser.add_argument("--semilla", type=int, default=7)
    parser.add_argument("--refuerzo", type=int, default=2)
    parser.add_argument("--refuerzo-poblacion", type=int, default=100)
    parser.add_argument("--refuerzo-generaciones", type=int, default=150)
    args = parser.parse_args()

    datos = cargar_costes()
    valores = np.arange(args.desde, args.hasta + 1e-9, args.paso)
    v_min, v_max = float(valores[0]), float(valores[-1])
    print(f"Barrido de {len(valores)} valores de v en "
          f"[{v_min:g}, {v_max:g}] EUR/h")

    candidatos, registros, _ = barrido(
        datos, valores,
        tam_poblacion=args.poblacion,
        generaciones=args.generaciones,
        semilla=args.semilla,
    )

    for v_extremo in (v_min, v_max):
        M_ext = datos.matriz_coste(v_extremo)
        for inicio in range(datos.n):
            tour_ext = or_opt(vecino_mas_cercano(M_ext, inicio), M_ext)
            T, B = datos.componentes_tour(tour_ext)
            candidatos.append((T, B, datos.firma(tour_ext)))

    tramos, puntos_cambio = envolvente_inferior(candidatos, v_min, v_max)

    for ronda in range(args.refuerzo or 0):
        puntos = sorted({v_min, v_max, *puntos_cambio})
        print(f"Refinamiento {ronda + 1}/{args.refuerzo}: "
              f"GA fuerte en {len(puntos)} puntos criticos")
        nuevos = refinar(
            datos, candidatos, tramos, puntos,
            tam_poblacion=args.refuerzo_poblacion,
            generaciones=args.refuerzo_generaciones,
            semillas=[args.semilla, args.semilla + 1],
        )
        tramos, puntos_cambio = envolvente_inferior(
            candidatos, v_min, v_max
        )
        print(f"  candidatos nuevos: {nuevos}")
        if nuevos == 0:
            break

    resumen, firmas = guardar_salidas(
        datos, registros, candidatos, tramos, puntos_cambio
    )

    print(f"\nRutas optimas distintas: {resumen['n_rutas_optimas_distintas']}")
    print(f"Tramos de valor de la hora: {len(tramos)}")
    cambios = resumen["valores_cambio_eur_h"]
    if cambios:
        print(f"La ruta optima cambia a partir de v = "
              f"{cambios[0]:.2f} EUR/h (y en {len(cambios)} umbrales mas).")
        print(f"Umbrales: {', '.join(f'{c:.2f}' for c in cambios)}")
    else:
        print("La ruta optima no cambia en el rango muestreado.")
    print("Salidas en resultados/: sensibilidad_valor_hora.csv, "
          "tramos_optimos.csv, rutas_optimas.csv, "
          "sensibilidad_valor_hora.png")


if __name__ == "__main__":
    _main()
