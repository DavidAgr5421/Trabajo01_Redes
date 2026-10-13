"""
Barrido de configuraciones del ACO para el TSP de las 47 capitales.

Objetivo: encontrar la configuracion de parametros (hormigas, iteraciones,
alfa, beta, evaporacion, semilla) que da el mejor coste dirigido
C = valor_hora * tiempo_h + tax_eur + gas_eur, midiendo tambien el
rendimiento (tiempo de ejecucion, iteracion de convergencia).

Estrategia (no se edita codigo para probar: todo sale del grid de abajo):
  Fase A  reproducibilidad de la config base sobre varias semillas
  Fase B  barrido OFAT (un factor por vez) alrededor de la base
  Fase C  combinacion de los mejores valores por factor, varias semillas
  Fase D  presupuesto alto (mas hormigas e iteraciones)

Salidas en resultados/configuracion_aco/:
  barrido_aco.csv                 cada corrida con parametros y metricas
  ranking_aco.json                corridas ordenadas por coste y tiempo
  agregados_por_fase.json         medias por fase y candidatas de robustez
  mejor_configuracion_aco.json    config ganadora + corrida y agregado
  mejor_ruta_aco.csv              arista a arista del mejor tour
  evolucion_aco.csv               convergencia de la mejor config
  resumen_aco.json                desglose de coste (formato comun)
  resumen_mejor_aco.json          desglose + corrida representativa

Uso:
  python3 experimentos_aco.py                 # estudio completo
  python3 experimentos_aco.py --fase base     # solo una fase
  python3 experimentos_aco.py --valor-hora 20
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from colonia_hormigas import ejecutar_aco, guardar_resultados
from costes_comunes import BASE_DIR, cargar_costes

DIR_SALIDA = BASE_DIR / "resultados" / "configuracion_aco"
EPS = 1e-9

# Configuracion de referencia (la de configuracion.json).
BASE = {
    "hormigas": 30,
    "iteraciones": 120,
    "alfa": 1.0,
    "beta": 3.0,
    "evaporacion": 0.2,
}

# Valores a explorar por factor (barrido OFAT).
GRID = {
    "hormigas": [10, 20, 30, 50, 80],
    "iteraciones": [30, 60, 120, 200, 300],
    "alfa": [0.5, 1.0, 1.5, 2.0, 2.5],
    "beta": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    "evaporacion": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7],
}

SEEDS_BASE = [1, 2, 3, 4, 5, 42]
SEEDS_COMBINADA = [1, 2, 3, 4, 5, 7, 11, 42]

# Candidatas a la mejor config: baratas vs. mas hormigas, mismo resto base.
# Se evaluan sobre las mismas semillas para comparar robustez (no una sola).
CANDIDATOS_ROBUSTEZ = [
    {"hogar": "h10_i120", "hormigas": 10, "iteraciones": 120},
    {"hogar": "h20_i120", "hormigas": 20, "iteraciones": 120},
    {"hogar": "h30_i120", "hormigas": 30, "iteraciones": 120},
    {"hogar": "h50_i120", "hormigas": 50, "iteraciones": 120},
    {"hogar": "h10_i200", "hormigas": 10, "iteraciones": 200},
]
SEEDS_ROBUSTEZ = [1, 2, 3, 4, 5, 6, 7, 8]


@dataclass
class Corrida:
    fase: str
    etiqueta: str
    hormigas: int
    iteraciones: int
    alfa: float
    beta: float
    evaporacion: float
    semilla: int
    coste_eur: float
    distancia_km: float
    tiempo_total_h: float
    coste_medio_final_eur: float
    iteracion_convergencia: int
    tiempo_ejecucion_s: float
    usar_or_opt: bool = True


def correr(M, datos, valor_hora, fase, etiqueta, parametros):
    """Ejecuta una configuracion y devuelve una Corrida + el resultado crudo."""
    t0 = time.perf_counter()
    res = ejecutar_aco(M, **parametros, verbose=False)
    dt = time.perf_counter() - t0

    desglose = datos.desglose_tour(res["tour"], valor_hora)
    hist = np.asarray(res["historia_mejor"], dtype=float)
    convergencia = int(np.argmax(hist <= hist[-1] + EPS))

    corrida = Corrida(
        fase=fase,
        etiqueta=etiqueta,
        hormigas=parametros["hormigas"],
        iteraciones=parametros["iteraciones"],
        alfa=parametros["alfa"],
        beta=parametros["beta"],
        evaporacion=parametros["evaporacion"],
        semilla=parametros["semilla"],
        usar_or_opt=parametros.get("usar_or_opt", True),
        coste_eur=round(desglose["coste_eur"], 4),
        distancia_km=round(desglose["distancia_km"], 3),
        tiempo_total_h=round(desglose["tiempo_h"], 6),
        coste_medio_final_eur=round(float(res["historia_media"][-1]), 4),
        iteracion_convergencia=convergencia,
        tiempo_ejecucion_s=round(dt, 3),
    )
    print(f"  [{fase:9s}] {etiqueta:32s} "
          f"coste={corrida.coste_eur:9.2f} EUR "
          f"conv_it={convergencia:3d} t={dt:5.2f}s")
    return corrida, res


def con_semilla(cfg, semilla):
    return {**BASE, **cfg, "semilla": semilla}


def fase_base(M, datos, vh, corridas):
    print("\n== Fase A: reproducibilidad de la config base ==")
    for s in SEEDS_BASE:
        p = con_semilla({}, s)
        corridas.append(correr(M, datos, vh, "base", f"base_seed{s}", p)[0])


def fase_ofat(M, datos, vh, corridas):
    print("\n== Fase B: barrido OFAT ==")
    for factor, valores in GRID.items():
        for valor in valores:
            cfg = {factor: valor}
            p = con_semilla(cfg, 42)
            corridas.append(
                correr(M, datos, vh, "ofat", f"{factor}={valor}", p)[0]
            )


def mejores_por_factor(corridas):
    """Mejor valor de cada factor segun el coste medio de las corridas OFAT."""
    elegidos = {}
    for factor, valores in GRID.items():
        candidatas = [
            c for c in corridas
            if c.fase == "ofat" and c.etiqueta.startswith(f"{factor}=")
        ]
        if candidatas:
            mejor = min(candidatas, key=lambda c: c.coste_eur)
            elegidos[factor] = getattr(mejor, factor)
    return elegidos


def agregar(corridas, etiqueta):
    costes = np.array([c.coste_eur for c in corridas], dtype=float)
    return {
        "etiqueta": etiqueta,
        "n_semillas": len(corridas),
        "coste_mejor_eur": round(float(costes.min()), 4),
        "coste_medio_eur": round(float(costes.mean()), 4),
        "coste_peor_eur": round(float(costes.max()), 4),
        "coste_desv_std_eur": round(float(costes.std()), 4),
        "tiempo_medio_s": round(float(np.mean(
            [c.tiempo_ejecucion_s for c in corridas])), 3),
        "iteracion_convergencia_media": round(float(np.mean(
            [c.iteracion_convergencia for c in corridas])), 2),
    }


def fase_combinada(M, datos, vh, corridas, elegidos):
    print("\n== Fase C: combinacion de mejores valores ==")
    print(f"  elegidos: {elegidos}")
    deltas = []
    for s in SEEDS_COMBINADA:
        p = con_semilla(elegidos, s)
        c, _ = correr(M, datos, vh, "combinada", f"comb_seed{s}", p)
        corridas.append(c)
        deltas.append(c)
    return agregar(deltas, "combinada")


def fase_alto_presupuesto(M, datos, vh, corridas):
    print("\n== Fase D: presupuesto alto ==")
    cfg = {"hormigas": 80, "iteraciones": 300}
    deltas = []
    for s in [1, 42]:
        p = con_semilla(cfg, s)
        c, _ = correr(M, datos, vh, "alto", f"alto_seed{s}", p)
        corridas.append(c)
        deltas.append(c)
    return agregar(deltas, "alto")


def fase_robustez(M, datos, vh, corridas):
    """Evalua cada candidata sobre las mismas semillas y agrega por config."""
    print("\n== Fase E: robustez de candidatas (mismas semillas) ==")
    agregados_cand = []
    for cand in CANDIDATOS_ROBUSTEZ:
        cfg = {k: v for k, v in cand.items() if k != "hogar"}
        deltas = []
        for s in SEEDS_ROBUSTEZ:
            p = con_semilla(cfg, s)
            c, _ = correr(M, datos, vh, "robustez", f"{cand['hogar']}_s{s}", p)
            corridas.append(c)
            deltas.append(c)
        agg = agregar(deltas, cand["hogar"])
        agg["configuracion"] = cfg
        agregados_cand.append(agg)
        print(f"    -> {agg['etiqueta']}: medio={agg['coste_medio_eur']:.2f} "
              f"mejor={agg['coste_mejor_eur']:.2f} "
              f"std={agg['coste_desv_std_eur']:.2f} "
              f"t_medio={agg['tiempo_medio_s']:.2f}s")
    return agregados_cand


def comparar_or_opt(M, datos, vh):
    """Enfrenta la mejor config con y sin Or-opt sobre las mismas semillas."""
    with open(DIR_SALIDA / "mejor_configuracion_aco.json",
              encoding="utf-8") as f:
        ganadora = json.load(f)["configuracion_aco"]
    base_cfg = {k: v for k, v in ganadora.items() if k != "semilla"}
    print(f"\n== Comparacion Or-opt | config {base_cfg} ==")
    filas: list[Corrida] = []
    resumen = {}
    for usar in (True, False):
        etiqueta = "con_oropt" if usar else "sin_oropt"
        deltas = []
        for s in SEEDS_ROBUSTEZ:
            p = {**base_cfg, "semilla": s, "usar_or_opt": usar}
            c, _ = correr(M, datos, vh, "oropt", f"{etiqueta}_s{s}", p)
            filas.append(c)
            deltas.append(c)
        agg = agregar(deltas, etiqueta)
        agg["usar_or_opt"] = usar
        resumen[etiqueta] = agg
        print(f"    -> {etiqueta}: medio={agg['coste_medio_eur']:.2f} "
              f"mejor={agg['coste_mejor_eur']:.2f} "
              f"std={agg['coste_desv_std_eur']:.2f} "
              f"t_medio={agg['tiempo_medio_s']:.2f}s")
    with open(DIR_SALIDA / "comparacion_oropt.csv", "w",
              encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(filas[0]).keys()))
        w.writeheader()
        for c in filas:
            w.writerow(asdict(c))
    with open(DIR_SALIDA / "comparacion_oropt.json", "w",
              encoding="utf-8") as f:
        json.dump({"valor_hora_eur": vh, "configuracion": base_cfg,
                   "semillas": SEEDS_ROBUSTEZ, "resumen": resumen},
                  f, ensure_ascii=False, indent=2)
    print(f"Salidas: comparacion_oropt.csv/json en {DIR_SALIDA}")
    return resumen


def guardar_barrido(corridas):
    ruta = DIR_SALIDA / "barrido_aco.csv"
    with open(ruta, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(corridas[0]).keys()))
        w.writeheader()
        for c in corridas:
            w.writerow(asdict(c))
    return ruta


def guardar_resumen_y_mejor(M, datos, vh, corridas, res_corrida, mejor,
                            config_aco, criterio, agregado=None):
    """Guarda ranking, mejor configuracion y artefactos del mejor tour."""
    # Ranking por coste (empates: menos tiempo de ejecucion gana).
    ranking = sorted(corridas, key=lambda c: (c.coste_eur,
                                              c.tiempo_ejecucion_s))
    with open(DIR_SALIDA / "ranking_aco.json", "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in ranking], f, ensure_ascii=False,
                  indent=2)

    mejor_json = {
        "criterio": criterio,
        "valor_hora_eur": vh,
        "configuracion_aco": config_aco,
        "mejor_corrida_representativa": asdict(mejor),
        "agregado_ganador": agregado,
    }
    with open(DIR_SALIDA / "mejor_configuracion_aco.json", "w",
              encoding="utf-8") as f:
        json.dump(mejor_json, f, ensure_ascii=False, indent=2)

    # Artefactos del mejor tour (ruta, evolucion, desglose).
    _, resumen = guardar_resultados(datos, res_corrida["tour"], vh,
                                    res_corrida, directorio=DIR_SALIDA)
    with open(DIR_SALIDA / "resumen_mejor_aco.json", "w",
              encoding="utf-8") as f:
        json.dump({"mejor_corrida": asdict(mejor),
                   "configuracion_aco": config_aco, "tour": resumen},
                  f, ensure_ascii=False, indent=2)
    return mejor, mejor_json


def main():
    p = argparse.ArgumentParser(description="Barrido de configuraciones ACO")
    p.add_argument("--fase", choices=["base", "ofat", "combinada", "alto",
                                      "robustez", "todo"], default="todo")
    p.add_argument("--valor-hora", type=float, default=9.36)
    p.add_argument("--comparar-oropt", action="store_true",
                   help="Solo enfrenta la mejor config con/sin Or-opt")
    args = p.parse_args()

    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    datos = cargar_costes()
    vh = args.valor_hora
    M = datos.matriz_coste(vh)
    print(f"ACO barrido | {datos.n} ciudades | valor_hora={vh:.2f} EUR/h")

    if args.comparar_oropt:
        t0 = time.perf_counter()
        comparar_or_opt(M, datos, vh)
        print(f"TOTAL {time.perf_counter()-t0:.1f}s")
        return

    t0 = time.perf_counter()
    corridas: list[Corrida] = []
    agregados: list[dict] = []
    agregados_cand: list[dict] = []

    if args.fase in ("base", "todo"):
        fase_base(M, datos, vh, corridas)
        agregados.append(agregar([c for c in corridas if c.fase == "base"],
                                 "base"))
    if args.fase in ("ofat", "todo"):
        fase_ofat(M, datos, vh, corridas)
    elegidos = mejores_por_factor(corridas)
    if args.fase in ("combinada", "todo"):
        agregados.append(fase_combinada(M, datos, vh, corridas, elegidos))
    if args.fase in ("alto", "todo"):
        agregados.append(fase_alto_presupuesto(M, datos, vh, corridas))
    if args.fase in ("robustez", "todo"):
        agregados_cand = fase_robustez(M, datos, vh, corridas)

    if not corridas:
        print("Sin corridas.")
        return

    # Eleccion del ganador. Si hubo fase de robustez, decide el coste MEDIO
    # (desempate: tiempo medio); si no, la corrida mas barata.
    agregado_ganador = None
    if agregados_cand:
        ganadora = min(agregados_cand,
                       key=lambda a: (a["coste_medio_eur"],
                                      a["tiempo_medio_s"]))
        cfg_ganadora = ganadora["configuracion"]
        mejor = min(
            (c for c in corridas
             if c.fase == "robustez"
             and c.etiqueta.startswith(ganadora["etiqueta"] + "_")),
            key=lambda c: (c.coste_eur, c.tiempo_ejecucion_s))
        agregado_ganador = ganadora
        criterio = ("menor coste_medio_eur sobre 8 semillas "
                    "(desempate: tiempo medio de ejecucion)")
        print(f"\nGanadora por coste medio: {ganadora['etiqueta']} "
              f"({ganadora['coste_medio_eur']:.2f} EUR medio, "
              f"{ganadora['coste_mejor_eur']:.2f} EUR mejor)")
        p_mejor = {**BASE, **cfg_ganadora, "semilla": mejor.semilla}
    else:
        mejor = min(corridas,
                    key=lambda c: (c.coste_eur, c.tiempo_ejecucion_s))
        criterio = "menor coste_eur (desempate: menor tiempo_ejecucion_s)"
        p_mejor = {
            "hormigas": mejor.hormigas, "iteraciones": mejor.iteraciones,
            "alfa": mejor.alfa, "beta": mejor.beta,
            "evaporacion": mejor.evaporacion, "semilla": mejor.semilla,
        }

    print(f"Mejor config: {p_mejor} -> {mejor.coste_eur:.2f} EUR")
    res_mejor = ejecutar_aco(M, **p_mejor, verbose=False)

    barrido_csv = guardar_barrido(corridas)
    mejor_obj, _ = guardar_resumen_y_mejor(
        M, datos, vh, corridas, res_mejor, mejor, p_mejor, criterio,
        agregado_ganador)

    with open(DIR_SALIDA / "agregados_por_fase.json", "w",
              encoding="utf-8") as f:
        json.dump({"valor_hora_eur": vh, "mejores_por_factor": elegidos,
                   "fases": agregados, "candidatas_robustez": agregados_cand},
                  f, ensure_ascii=False, indent=2)

    print(f"\nTOTAL {time.perf_counter()-t0:.1f}s | {len(corridas)} corridas")
    print(f"Salidas en {DIR_SALIDA}")
    print(f"Mejor: {mejor_obj.etiqueta} = {mejor_obj.coste_eur:.2f} EUR "
          f"({mejor_obj.tiempo_ejecucion_s:.2f}s)")


if __name__ == "__main__":
    main()
