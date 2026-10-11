"""
Algoritmo genetico (GA) para el TSP cerrado de las 47 capitales.

Minimiza el coste dirigido C = valor_hora*tiempo + tax + gas sobre un ciclo
que visita las 47 ciudades exactamente una vez y vuelve al origen.

Componentes:
  - Representacion por permutacion (el ciclo se evalua cerrando el ultimo
    tramo contra el primero).
  - Inicializacion mixta: vecino mas cercano desde varios origenes + azar.
  - Cruce OX (order crossover), mutacion por inversion/swap.
  - Seleccion por torneo, elitismo.
  - Busqueda local Or-opt (reubicacion de segmentos de longitud 1-3), valida
    para matrices asimetricas y por tanto aplicable a este problema.

Salidas (por defecto en resultados/):
  mejor_ruta_ga.csv   secuencia y desglose por tramo de la mejor ruta
  evolucion_ga.csv    mejor coste y coste medio por generacion
  resumen_ga.json     parametros y totales

Uso:
  python3 algoritmo_genetico.py --valor-hora 20
  python3 algoritmo_genetico.py --valor-hora 35 --poblacion 120 --generaciones 400
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from costes_comunes import (
    BASE_DIR,
    VALOR_HORA_POR_DEFECTO,
    cargar_costes,
)

DIR_RESULTADOS = BASE_DIR / "resultados"
EPS = 1e-9


def coste_tour(M: np.ndarray, tour) -> float:
    idx = np.asarray(tour, dtype=np.int64)
    return float(M[idx, np.roll(idx, -1)].sum())


def vecino_mas_cercano(M: np.ndarray, inicio: int = 0) -> np.ndarray:
    """Heuristica voraz: desde 'inicio' salta siempre al no visitado mas barato."""
    n = M.shape[0]
    visitados = np.zeros(n, dtype=bool)
    tour = np.empty(n, dtype=np.int64)
    tour[0] = inicio
    visitados[inicio] = True
    actual = inicio
    for k in range(1, n):
        fila = M[actual].astype(float).copy()
        fila[visitados] = np.inf
        siguiente = int(np.argmin(fila))
        tour[k] = siguiente
        visitados[siguiente] = True
        actual = siguiente
    return tour


def or_opt(tour, M: np.ndarray, max_seg: int = 3) -> np.ndarray:
    """Busqueda local Or-opt: reubica segmentos de longitud 1..max_seg.

    El orden interno del segmento no cambia, por lo que el delta de coste
    solo afecta a cuatro aristas y es correcto aunque M sea asimetrica.
    Mantiene fija la ciudad de la posicion 0 para evitar ambiguedad ciclica.
    Los deltas de todas las posiciones destino se calculan de forma
    vectorizada para que cada pasada sea rapida.
    """
    n = len(tour)
    t = np.asarray(tour, dtype=np.int64).copy()
    while True:
        mejorado = False
        d_arr = np.roll(t, -1)
        for L in range(1, max_seg + 1):
            i = 1
            while i + L - 1 < n:
                seg_ini = t[i]
                seg_fin = t[i + L - 1]
                p = t[i - 1]
                q = t[(i + L) % n]
                rem = M[p, q] - M[p, seg_ini] - M[seg_fin, q]
                delta = (
                    rem + M[t, seg_ini] + M[seg_fin, d_arr] - M[t, d_arr]
                )
                delta[i - 1] = np.inf
                delta[i:i + L] = np.inf
                j = int(np.argmin(delta))
                if delta[j] < -EPS:
                    seg = t[i:i + L].copy()
                    resto = np.concatenate([t[:i], t[i + L:]])
                    j2 = j if j < i else j - L
                    t = np.concatenate([resto[:j2 + 1], seg, resto[j2 + 1:]])
                    mejorado = True
                    break
                i += 1
            if mejorado:
                break
        if not mejorado:
            break
    return t


def cruce_ox(p1: np.ndarray, p2: np.ndarray, rng) -> np.ndarray:
    """Order crossover (OX): conserva un segmento de p1 y el orden de p2."""
    n = len(p1)
    a, b = sorted(rng.choice(n, size=2, replace=False))
    hijo = np.full(n, -1, dtype=np.int64)
    hijo[a:b + 1] = p1[a:b + 1]
    usados = set(int(x) for x in p1[a:b + 1])
    relleno = [int(x) for x in p2 if int(x) not in usados]
    pos = (b + 1) % n
    k = 0
    for _ in range(n):
        if hijo[pos] == -1:
            hijo[pos] = relleno[k]
            k += 1
        pos = (pos + 1) % n
    return hijo


def mutar(tour: np.ndarray, prob: float, rng) -> np.ndarray:
    t = tour.copy()
    if rng.random() >= prob:
        return t
    n = len(t)
    if rng.random() < 0.5:
        a, b = sorted(rng.choice(n, size=2, replace=False))
        t[a:b + 1] = t[a:b + 1][::-1]
    else:
        a, b = rng.choice(n, size=2, replace=False)
        t[a], t[b] = t[b], t[a]
    return t


def _torneo(poblacion, costes, tam, rng) -> np.ndarray:
    idx = rng.integers(0, len(poblacion), size=tam)
    mejor = idx[int(np.argmin(costes[idx]))]
    return poblacion[mejor].copy()


def ejecutar_ga(
    M: np.ndarray,
    *,
    tam_poblacion: int = 100,
    generaciones: int = 300,
    prob_cruce: float = 0.9,
    prob_mutacion: float = 0.3,
    tam_torneo: int = 3,
    elite: int = 2,
    semilla: int = 42,
    tours_iniciales=None,
    verbose: bool = False,
    registro: list | None = None,
    grabar_cada: int = 1,
    **_,
) -> dict:
    """Ejecuta el GA y devuelve la mejor ruta, su coste y la historia."""
    rng = np.random.default_rng(semilla)
    n = M.shape[0]

    def evaluar(pop):
        return np.array([coste_tour(M, ind) for ind in pop], dtype=float)

    poblacion = []
    if tours_iniciales:
        for t in tours_iniciales:
            poblacion.append(np.array(t, dtype=np.int64))
    n_voraces = min(n, max(2, tam_poblacion // 8))
    for inicio in rng.choice(n, size=n_voraces, replace=False):
        poblacion.append(or_opt(vecino_mas_cercano(M, int(inicio)), M))
    while len(poblacion) < tam_poblacion:
        poblacion.append(rng.permutation(n).astype(np.int64))
    poblacion = poblacion[:tam_poblacion]
    costes = evaluar(poblacion)

    historia_mejor = []
    historia_media = []

    orden = np.argsort(costes)
    poblacion = [poblacion[k] for k in orden]
    costes = costes[orden]
    if registro is not None:
        registro.append((0, float(costes[0]), poblacion[0].copy()))

    for gen in range(generaciones):
        orden = np.argsort(costes)
        poblacion = [poblacion[k] for k in orden]
        costes = costes[orden]

        for e in range(min(elite, tam_poblacion)):
            pulido = or_opt(poblacion[e], M)
            c = coste_tour(M, pulido)
            if c < costes[e] - EPS:
                poblacion[e] = pulido
                costes[e] = c

        historia_mejor.append(float(costes[0]))
        historia_media.append(float(costes.mean()))

        if registro is not None and (
            (gen + 1) % grabar_cada == 0 or gen == generaciones - 1
        ):
            registro.append((gen + 1, float(costes[0]), poblacion[0].copy()))

        nueva = [poblacion[e].copy() for e in range(min(elite, tam_poblacion))]
        while len(nueva) < tam_poblacion:
            p1 = _torneo(poblacion, costes, tam_torneo, rng)
            p2 = _torneo(poblacion, costes, tam_torneo, rng)
            if rng.random() < prob_cruce:
                hijo = cruce_ox(p1, p2, rng)
            else:
                hijo = p1.copy()
            hijo = mutar(hijo, prob_mutacion, rng)
            if rng.random() < 0.2:
                hijo = or_opt(hijo, M)
            nueva.append(hijo)
        poblacion = nueva
        costes = evaluar(poblacion)

        if verbose and (gen % 50 == 0 or gen == generaciones - 1):
            print(f"  gen {gen:4d}: mejor {costes.min():.2f} "
                  f"medio {costes.mean():.2f}")

    orden = np.argsort(costes)
    mejor = or_opt(poblacion[orden[0]], M)
    mejor_coste = coste_tour(M, mejor)
    if registro is not None:
        registro.append((generaciones, mejor_coste, mejor.copy()))

    return {
        "tour": mejor,
        "coste": mejor_coste,
        "historia_mejor": historia_mejor,
        "historia_media": historia_media,
        "tam_poblacion": tam_poblacion,
        "generaciones": generaciones,
        "semilla": semilla,
    }


def guardar_resultados(datos, tour, valor_hora, historia, directorio=DIR_RESULTADOS):
    directorio = Path(directorio)
    directorio.mkdir(parents=True, exist_ok=True)
    idx = np.asarray(tour, dtype=np.int64)
    nombres = datos.tour_a_nombres(tour)
    n = len(idx)
    M = datos.matriz_coste(valor_hora)

    import csv

    ruta_csv = directorio / "mejor_ruta_ga.csv"
    with open(ruta_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["paso", "origen", "destino", "lat_origen", "lon_origen",
                    "distancia_km", "tiempo_h", "tax_eur", "gas_eur",
                    "coste_eur"])
        for k in range(n):
            a, b = idx[k], idx[(k + 1) % n]
            lat, lon = datos.coords.loc[datos.ciudades[a], ["lat", "lon"]]
            w.writerow([
                k + 1, datos.ciudades[a], datos.ciudades[b],
                f"{lat:.6f}", f"{lon:.6f}",
                f"{datos.dist_km[a, b]:.3f}", f"{datos.time_h[a, b]:.6f}",
                f"{datos.tax_eur[a, b]:.2f}", f"{datos.gas_eur[a, b]:.2f}",
                f"{M[a, b]:.2f}",
            ])

    evo_csv = directorio / "evolucion_ga.csv"
    with open(evo_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generacion", "mejor_coste_eur", "coste_medio_eur"])
        for gen, (mejor, medio) in enumerate(
            zip(historia["historia_mejor"], historia["historia_media"])
        ):
            w.writerow([gen, f"{mejor:.4f}", f"{medio:.4f}"])

    desglose = datos.desglose_tour(tour, valor_hora)
    resumen = {
        "valor_hora_eur": valor_hora,
        "ciudades": n,
        "parametros_ga": {
            "tam_poblacion": historia["tam_poblacion"],
            "generaciones": historia["generaciones"],
            "semilla": historia["semilla"],
        },
        "coste_total_eur": round(desglose["coste_eur"], 4),
        "distancia_total_km": round(desglose["distancia_km"], 3),
        "tiempo_total_h": round(desglose["tiempo_h"], 6),
        "peajes_total_eur": round(desglose["tax_eur"], 4),
        "combustible_total_eur": round(desglose["gas_eur"], 4),
        "orden_ciudades": nombres,
        "fuente_datos": "datos_rutas/rutas_costes.csv + ciudades.csv",
    }
    with open(directorio / "resumen_ga.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)

    return ruta_csv, evo_csv, resumen


def _main():
    parser = argparse.ArgumentParser(description="GA para el TSP de 47 capitales")
    parser.add_argument("--valor-hora", type=float,
                        default=VALOR_HORA_POR_DEFECTO)
    parser.add_argument("--poblacion", type=int, default=100)
    parser.add_argument("--generaciones", type=int, default=300)
    parser.add_argument("--prob-mutacion", type=float, default=0.3)
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    datos = cargar_costes()
    M = datos.matriz_coste(args.valor_hora)
    print(f"GA: {datos.n} ciudades, valor_hora={args.valor_hora:.2f} EUR/h, "
          f"poblacion={args.poblacion}, generaciones={args.generaciones}")

    resultado = ejecutar_ga(
        M,
        tam_poblacion=args.poblacion,
        generaciones=args.generaciones,
        prob_mutacion=args.prob_mutacion,
        semilla=args.semilla,
        verbose=args.verbose,
    )
    ruta_csv, evo_csv, resumen = guardar_resultados(
        datos, resultado["tour"], args.valor_hora, resultado
    )
    print(f"Mejor coste: {resultado['coste']:.2f} EUR")
    print(f"Distancia: {resumen['distancia_total_km']:.1f} km | "
          f"Tiempo: {resumen['tiempo_total_h']:.2f} h | "
          f"Peajes: {resumen['peajes_total_eur']:.2f} EUR")
    print(f"Ruta: {ruta_csv}")
    print(f"Evolucion: {evo_csv}")


if __name__ == "__main__":
    _main()
