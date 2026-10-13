"""
Colonia de hormigas (ACO) para el TSP cerrado de las 47 capitales.

Minimiza el mismo coste dirigido que el GA:
    C(valor_hora) = valor_hora * tiempo_h + tax_eur + gas_eur

Metodo (Ant System + elitismo):
  - Cada hormiga construye un tour eligiendo la proxima ciudad con
    probabilidad proporcional a (feromona^alfa * heuristica^beta),
    donde heuristica = 1 / (coste + eps).
  - Evaporacion: feromona *= (1 - evaporacion) en cada iteracion.
  - Deposito: cada hormiga aporta Q/coste sobre sus aristas; la mejor
    global aporta un refuerzo elitista extra.
  - Busqueda local Or-opt sobre la mejor de cada iteracion (misma que el GA,
    valida para matrices asimetricas).

Salidas (por defecto en resultados/):
  mejor_ruta_aco.csv, evolucion_aco.csv, resumen_aco.json

Uso:
  python3 colonia_hormigas.py --config configuracion.json
  python3 colonia_hormigas.py --valor-hora 20 --hormigas 30 --iteraciones 120
"""
from __future__ import annotations

import argparse
import csv
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
    """Coste del ciclo cerrado suma de M[a]->M[b] con retorno al origen."""
    idx = np.asarray(tour, dtype=np.int64)
    return float(M[idx, np.roll(idx, -1)].sum())


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


def construir_tour(M: np.ndarray, feromona: np.ndarray, heuristica: np.ndarray,
                   alfa: float, beta: float, rng) -> np.ndarray:
    n = M.shape[0]
    inicio = int(rng.integers(0, n))
    tour = np.empty(n, dtype=np.int64)
    tour[0] = inicio
    visitados = np.zeros(n, dtype=bool)
    visitados[inicio] = True
    actual = inicio
    for k in range(1, n):
        f = np.power(feromona[actual], alfa)
        h = np.power(heuristica[actual], beta)
        prob = f * h
        prob[visitados] = 0.0
        s = prob.sum()
        if s <= 0 or not np.isfinite(s):
            candidatos = np.flatnonzero(~visitados)
            actual = int(rng.choice(candidatos))
        else:
            prob /= s
            actual = int(rng.choice(n, p=prob))
        tour[k] = actual
        visitados[actual] = True
    return tour


def ejecutar_aco(
    M: np.ndarray,
    *,
    hormigas: int = 30,
    iteraciones: int = 120,
    alfa: float = 1.0,
    beta: float = 3.0,
    evaporacion: float = 0.2,
    semilla: int = 42,
    usar_or_opt: bool = True,
    verbose: bool = False,
    registro: list | None = None,
    grabar_cada: int = 1,
    **_,
) -> dict:
    """Ejecuta el ACO y devuelve mejor tour, coste e historia."""
    rng = np.random.default_rng(semilla)
    n = M.shape[0]
    heuristica = 1.0 / (M + 1e-6)
    np.fill_diagonal(heuristica, 0.0)
    feromona = np.ones((n, n), dtype=float)
    np.fill_diagonal(feromona, 0.0)

    mejor_global = None
    coste_global = float("inf")
    historia_mejor: list[float] = []
    historia_media: list[float] = []

    if registro is not None:
        registro.append((0, float("inf"), None))

    for it in range(iteraciones):
        tours = [construir_tour(M, feromona, heuristica, alfa, beta, rng)
                 for _ in range(hormigas)]
        costes = np.array([coste_tour(M, t) for t in tours], dtype=float)

        idx_best = int(np.argmin(costes))
        if usar_or_opt:
            candidato = or_opt(tours[idx_best], M)
        else:
            candidato = np.asarray(tours[idx_best], dtype=np.int64)
        coste_candidato = coste_tour(M, candidato)
        if coste_candidato < coste_global - EPS:
            mejor_global = candidato
            coste_global = coste_candidato

        # Evaporacion + deposito proporcional (Q = coste medio para escala).
        feromona *= (1.0 - evaporacion)
        Q = float(costes.mean()) if costes.mean() > 0 else 1.0
        for t, c in zip(tours, costes):
            aporte = Q / max(c, EPS)
            a = np.asarray(t, dtype=np.int64)
            b = np.roll(a, -1)
            feromona[a, b] += aporte
        # Refuerzo elitista de la mejor global.
        if mejor_global is not None:
            a = np.asarray(mejor_global, dtype=np.int64)
            b = np.roll(a, -1)
            feromona[a, b] += Q / max(coste_global, EPS)
        np.fill_diagonal(feromona, 0.0)

        historia_mejor.append(float(coste_global))
        historia_media.append(float(costes.mean()))
        if registro is not None and (
            (it + 1) % grabar_cada == 0 or it == iteraciones - 1
        ):
            registro.append((it + 1, float(coste_global),
                             np.asarray(mejor_global).copy()))
        if verbose and (it % 20 == 0 or it == iteraciones - 1):
            print(f"  it {it:4d}: mejor {coste_global:.2f} "
                  f"media {costes.mean():.2f}")

    if registro is not None and registro[0][2] is None:
        registro.pop(0)
    return {
        "tour": np.asarray(mejor_global, dtype=np.int64),
        "coste": float(coste_global),
        "historia_mejor": historia_mejor,
        "historia_media": historia_media,
        "hormigas": hormigas,
        "iteraciones": iteraciones,
        "semilla": semilla,
        "usar_or_opt": bool(usar_or_opt),
    }


def guardar_resultados(datos, tour, valor_hora, historia,
                       directorio=DIR_RESULTADOS):
    directorio = Path(directorio)
    directorio.mkdir(parents=True, exist_ok=True)
    idx = np.asarray(tour, dtype=np.int64)
    nombres = datos.tour_a_nombres(tour)
    n = len(idx)
    M = datos.matriz_coste(valor_hora)

    ruta_csv = directorio / "mejor_ruta_aco.csv"
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

    with open(directorio / "evolucion_aco.csv", "w",
              encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iteracion", "mejor_coste_eur", "coste_medio_eur"])
        for it, (mejor, medio) in enumerate(
            zip(historia["historia_mejor"], historia["historia_media"])
        ):
            w.writerow([it, f"{mejor:.4f}", f"{medio:.4f}"])

    desglose = datos.desglose_tour(tour, valor_hora)
    resumen = {
        "algoritmo": "aco",
        "valor_hora_eur": valor_hora,
        "ciudades": n,
        "parametros_aco": {
            "hormigas": historia["hormigas"],
            "iteraciones": historia["iteraciones"],
            "semilla": historia["semilla"],
            "usar_or_opt": historia.get("usar_or_opt", True),
        },
        "coste_total_eur": round(desglose["coste_eur"], 4),
        "distancia_total_km": round(desglose["distancia_km"], 3),
        "tiempo_total_h": round(desglose["tiempo_h"], 6),
        "peajes_total_eur": round(desglose["tax_eur"], 4),
        "combustible_total_eur": round(desglose["gas_eur"], 4),
        "orden_ciudades": nombres,
        "fuente_datos": "datos_rutas/rutas_costes.csv + ciudades.csv",
    }
    with open(directorio / "resumen_aco.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)
    return ruta_csv, resumen


def _main():
    parser = argparse.ArgumentParser(description="ACO para el TSP de 47 capitales")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--valor-hora", type=float, default=None)
    parser.add_argument("--hormigas", type=int, default=None)
    parser.add_argument("--iteraciones", type=int, default=None)
    parser.add_argument("--alfa", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--evaporacion", type=float, default=None)
    parser.add_argument("--semilla", type=int, default=None)
    parser.add_argument("--sin-or-opt", action="store_true",
                        help="Desactiva la busqueda local Or-opt")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    from config import cargar_config
    cfg = cargar_config(args.config)
    aco = cfg.get("aco", {})
    vh = args.valor_hora if args.valor_hora is not None else cfg.get(
        "valor_hora", VALOR_HORA_POR_DEFECTO)
    hormigas = args.hormigas or aco.get("hormigas", 30)
    iters = args.iteraciones or aco.get("iteraciones", 120)

    datos = cargar_costes()
    M = datos.matriz_coste(vh)
    usar_or_opt = aco.get("usar_or_opt", True) and not args.sin_or_opt
    print(f"ACO: {datos.n} ciudades, valor_hora={vh:.2f} EUR/h, "
          f"hormigas={hormigas}, iteraciones={iters}, "
          f"or_opt={'si' if usar_or_opt else 'no'}")
    res = ejecutar_aco(
        M, hormigas=hormigas, iteraciones=iters,
        alfa=args.alfa or aco.get("alfa", 1.0),
        beta=args.beta or aco.get("beta", 3.0),
        evaporacion=(args.evaporacion if args.evaporacion is not None
                     else aco.get("evaporacion", 0.2)),
        semilla=args.semilla or aco.get("semilla", 42),
        usar_or_opt=usar_or_opt,
        verbose=args.verbose,
    )
    ruta_csv, resumen = guardar_resultados(datos, res["tour"], vh, res)
    print(f"Mejor coste: {res['coste']:.2f} EUR | Ruta: {ruta_csv}")


if __name__ == "__main__":
    _main()
