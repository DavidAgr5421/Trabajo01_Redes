"""
Barrido de configuraciones del ACO para el TSP de las 47 capitales peninsulares.

Objetivo:
  Encontrar la configuracion optima de la Colonia de Hormigas (ACO) manteniendo la
  semilla fija en 42, con el valor de la hora del vendedor fijado en 5.09 EUR/h
  segun el SMI 2026 (RD 126/2026, 40.70 EUR/dia / 8 h continuas).

Comparacion central:
  ACO con busqueda local Or-opt vs ACO puro sin Or-opt.

Estrategia (semilla = 42 fija en todas las configuraciones):
  Fase A: Configuracion base de referencia
  Fase B: Barrido OFAT (un factor por vez: hormigas, iteraciones, alfa, beta, evaporacion)
  Fase C: Combinacion de mejores factores individuales
  Fase D: Exploracion de presupuesto computacional (hormigas x iteraciones)
  Fase E: Comparativa exhaustiva CON Or-opt vs SIN Or-opt

Salidas en resultados/ y resultados/configuracion_aco/:
  - barrido_aco.csv
  - ranking_aco.json
  - mejor_configuracion_aco.json
  - comparacion_oropt.json / comparacion_oropt.csv
  - mejor_ruta_aco_con_oropt.csv / mejor_ruta_aco_sin_oropt.csv
  - evolucion_aco_con_oropt.csv / evolucion_aco_sin_oropt.csv
  - comparacion_convergencia.png
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from colonia_hormigas import ejecutar_aco, guardar_resultados
from costes_comunes import BASE_DIR, VALOR_HORA_POR_DEFECTO, cargar_costes

DIR_RESULTADOS = BASE_DIR / "resultados"
DIR_SALIDA = DIR_RESULTADOS
EPS = 1e-9
SEMILLA_FIJA = 42

# Configuracion de referencia
BASE = {
    "hormigas": 20,
    "iteraciones": 150,
    "alfa": 1.0,
    "beta": 3.0,
    "evaporacion": 0.2,
    "semilla": SEMILLA_FIJA,
}

# Espacio de exploracion OFAT (manteniendo semilla 42 fija)
GRID = {
    "hormigas": [10, 15, 20, 30, 50, 80],
    "iteraciones": [50, 100, 150, 200, 300],
    "alfa": [0.5, 1.0, 1.5, 2.0, 2.5],
    "beta": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    "evaporacion": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
}

# Combinaciones de presupuesto a evaluar con semilla 42
CANDIDATOS_PRESUPUESTO = [
    {"etiqueta": "h10_i100", "hormigas": 10, "iteraciones": 100},
    {"etiqueta": "h10_i200", "hormigas": 10, "iteraciones": 200},
    {"etiqueta": "h20_i150", "hormigas": 20, "iteraciones": 150},
    {"etiqueta": "h30_i120", "hormigas": 30, "iteraciones": 120},
    {"etiqueta": "h30_i200", "hormigas": 30, "iteraciones": 200},
    {"etiqueta": "h50_i150", "hormigas": 50, "iteraciones": 150},
    {"etiqueta": "h50_i300", "hormigas": 50, "iteraciones": 300},
]


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
    usar_or_opt: bool
    coste_eur: float
    distancia_km: float
    tiempo_total_h: float
    coste_tiempo_eur: float
    peajes_eur: float
    gas_eur: float
    coste_medio_final_eur: float
    iteracion_convergencia: int
    tiempo_ejecucion_s: float


def correr(M, datos, valor_hora, fase, etiqueta, parametros):
    """Ejecuta una corrida del ACO con parametros especificos."""
    parametros = dict(parametros)
    parametros["semilla"] = SEMILLA_FIJA  # Garantiza semilla fija 42

    t0 = time.perf_counter()
    res = ejecutar_aco(M, **parametros, verbose=False)
    dt = time.perf_counter() - t0

    desglose = datos.desglose_tour(res["tour"], valor_hora)
    hist = np.asarray(res["historia_mejor"], dtype=float)
    convergencia = int(np.argmax(hist <= hist[-1] + EPS))
    coste_tiempo = round(valor_hora * desglose["tiempo_h"], 4)

    corrida = Corrida(
        fase=fase,
        etiqueta=etiqueta,
        hormigas=parametros["hormigas"],
        iteraciones=parametros["iteraciones"],
        alfa=parametros["alfa"],
        beta=parametros["beta"],
        evaporacion=parametros["evaporacion"],
        semilla=SEMILLA_FIJA,
        usar_or_opt=parametros.get("usar_or_opt", True),
        coste_eur=round(desglose["coste_eur"], 4),
        distancia_km=round(desglose["distancia_km"], 3),
        tiempo_total_h=round(desglose["tiempo_h"], 6),
        coste_tiempo_eur=coste_tiempo,
        peajes_eur=round(desglose["tax_eur"], 4),
        gas_eur=round(desglose["gas_eur"], 4),
        coste_medio_final_eur=round(float(res["historia_media"][-1]), 4),
        iteracion_convergencia=convergencia,
        tiempo_ejecucion_s=round(dt, 3),
    )
    opt_str = "con Or-opt" if corrida.usar_or_opt else "SIN Or-opt"
    print(f"  [{fase:11s}] {etiqueta:28s} ({opt_str}) -> "
          f"coste={corrida.coste_eur:8.2f} EUR | conv_it={convergencia:3d} | t={dt:5.2f}s")
    return corrida, res


def fase_base(M, datos, vh, corridas):
    print("\n== Fase A: Configuracion Base (semilla=42) ==")
    c, r = correr(M, datos, vh, "base", "config_base", BASE)
    corridas.append(c)
    return c, r


def fase_ofat(M, datos, vh, corridas):
    print("\n== Fase B: Barrido OFAT (semilla=42 fija) ==")
    for factor, valores in GRID.items():
        for val in valores:
            p = {**BASE, factor: val}
            c, _ = correr(M, datos, vh, "ofat", f"{factor}={val}", p)
            corridas.append(c)


def mejores_por_factor(corridas):
    """Obtiene el mejor parametro de cada factor del barrido OFAT."""
    elegidos = {}
    for factor, valores in GRID.items():
        candidatas = [
            c for c in corridas
            if c.fase == "ofat" and c.etiqueta.startswith(f"{factor}=")
        ]
        if candidatas:
            mejor = min(candidatas, key=lambda c: (c.coste_eur, c.tiempo_ejecucion_s))
            elegidos[factor] = getattr(mejor, factor)
    return elegidos


def fase_combinada(M, datos, vh, corridas, elegidos):
    print("\n== Fase C: Combinacion de Mejores Factores OFAT (semilla=42) ==")
    p = {**BASE, **elegidos, "semilla": SEMILLA_FIJA}
    c, r = correr(M, datos, vh, "combinada", "comb_mejores_ofat", p)
    corridas.append(c)
    return c, r


def fase_presupuesto(M, datos, vh, corridas):
    print("\n== Fase D: Exploracion de Presupuesto Computacional (semilla=42) ==")
    for cand in CANDIDATOS_PRESUPUESTO:
        cfg = {**BASE, "hormigas": cand["hormigas"], "iteraciones": cand["iteraciones"]}
        c, _ = correr(M, datos, vh, "presupuesto", cand["etiqueta"], cfg)
        corridas.append(c)


def comparar_con_y_sin_oropt(M, datos, vh, config_ganadora):
    """Realiza la comparativa detallada entre ACO con Or-opt y sin Or-opt."""
    print(f"\n== Fase E: Comparacion CON Or-opt vs SIN Or-opt (semilla=42) ==")
    base_cfg = {k: v for k, v in config_ganadora.items() if k not in ("usar_or_opt",)}
    base_cfg["semilla"] = SEMILLA_FIJA

    # 1. Con Or-opt
    p_con = {**base_cfg, "usar_or_opt": True}
    c_con, res_con = correr(M, datos, vh, "comparativa", "ganador_con_oropt", p_con)

    # 2. Sin Or-opt (misma configuracion)
    p_sin = {**base_cfg, "usar_or_opt": False}
    c_sin, res_sin = correr(M, datos, vh, "comparativa", "mismo_sin_oropt", p_sin)

    # 3. Sin Or-opt con mayor presupuesto para ver si converge mejor
    p_sin_alto = {**base_cfg, "hormigas": max(base_cfg["hormigas"] * 2, 50),
                  "iteraciones": max(base_cfg["iteraciones"] * 2, 300),
                  "usar_or_opt": False}
    c_sin_alto, res_sin_alto = correr(M, datos, vh, "comparativa", "sin_oropt_alto_presupuesto", p_sin_alto)

    comparacion = {
        "valor_hora_eur": vh,
        "semilla": SEMILLA_FIJA,
        "con_or_opt": {
            **asdict(c_con),
            "orden_ciudades": datos.tour_a_nombres(res_con["tour"]),
        },
        "sin_or_opt": {
            **asdict(c_sin),
            "orden_ciudades": datos.tour_a_nombres(res_sin["tour"]),
        },
        "sin_or_opt_alto_presupuesto": {
            **asdict(c_sin_alto),
            "orden_ciudades": datos.tour_a_nombres(res_sin_alto["tour"]),
        },
        "diferencias": {
            "ahorro_coste_eur": round(c_sin.coste_eur - c_con.coste_eur, 4),
            "ahorro_coste_porcentaje": round((c_sin.coste_eur - c_con.coste_eur) / c_sin.coste_eur * 100, 2),
            "diferencia_distancia_km": round(c_sin.distancia_km - c_con.distancia_km, 3),
            "diferencia_tiempo_h": round(c_sin.tiempo_total_h - c_con.tiempo_total_h, 4),
            "diferencia_tiempo_ejecucion_s": round(c_con.tiempo_ejecucion_s - c_sin.tiempo_ejecucion_s, 3),
        }
    }

    # Guardar rutas detalladas para ambas variantes
    guardar_resultados(datos, res_con["tour"], vh, res_con, directorio=DIR_SALIDA)
    guardar_resultados(datos, res_con["tour"], vh, res_con, directorio=DIR_RESULTADOS)

    # Guardar archivos especificos nombrados con y sin or-opt
    _guardar_ruta_csv(DIR_RESULTADOS / "mejor_ruta_aco_con_oropt.csv", datos, res_con["tour"], vh)
    _guardar_ruta_csv(DIR_RESULTADOS / "mejor_ruta_aco_sin_oropt.csv", datos, res_sin["tour"], vh)
    _guardar_evolucion_csv(DIR_RESULTADOS / "evolucion_aco_con_oropt.csv", res_con)
    _guardar_evolucion_csv(DIR_RESULTADOS / "evolucion_aco_sin_oropt.csv", res_sin)

    # Tambien en configuracion_aco para compatibilidad
    _guardar_ruta_csv(DIR_SALIDA / "mejor_ruta_aco_con_oropt.csv", datos, res_con["tour"], vh)
    _guardar_ruta_csv(DIR_SALIDA / "mejor_ruta_aco_sin_oropt.csv", datos, res_sin["tour"], vh)

    # Generar grafica comparativa de convergencia
    generar_grafica_comparativa(res_con, res_sin, res_sin_alto, vh,
                                DIR_RESULTADOS / "comparacion_convergencia.png")
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    generar_grafica_comparativa(res_con, res_sin, res_sin_alto, vh,
                                DIR_SALIDA / "comparacion_convergencia.png")

    # Guardar JSON y CSV comparativo
    with open(DIR_SALIDA / "comparacion_oropt.json", "w", encoding="utf-8") as f:
        json.dump(comparacion, f, ensure_ascii=False, indent=2)
    with open(DIR_RESULTADOS / "comparacion_oropt.json", "w", encoding="utf-8") as f:
        json.dump(comparacion, f, ensure_ascii=False, indent=2)

    filas_csv = [asdict(c_con), asdict(c_sin), asdict(c_sin_alto)]
    for destino in (DIR_SALIDA / "comparacion_oropt.csv", DIR_RESULTADOS / "comparacion_oropt.csv"):
        with open(destino, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(filas_csv[0].keys()))
            w.writeheader()
            for fila in filas_csv:
                w.writerow(fila)

    return comparacion, res_con, res_sin


def _guardar_ruta_csv(ruta_csv: Path, datos, tour, valor_hora):
    idx = np.asarray(tour, dtype=np.int64)
    n = len(idx)
    M = datos.matriz_coste(valor_hora)
    with open(ruta_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["paso", "origen", "destino", "lat_origen", "lon_origen",
                    "distancia_km", "tiempo_h", "tax_eur", "gas_eur", "coste_eur"])
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


def _guardar_evolucion_csv(ruta_csv: Path, res):
    with open(ruta_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iteracion", "mejor_coste_eur", "coste_medio_eur"])
        for it, (mejor, medio) in enumerate(zip(res["historia_mejor"], res["historia_media"])):
            w.writerow([it + 1, f"{mejor:.4f}", f"{medio:.4f}"])


def generar_grafica_comparativa(res_con, res_sin, res_sin_alto, valor_hora, ruta_salida):
    fig, ax = plt.subplots(figsize=(10, 6))

    h_con = res_con["historia_mejor"]
    h_sin = res_sin["historia_mejor"]
    h_sin_alto = res_sin_alto["historia_mejor"]

    ax.plot(range(1, len(h_con) + 1), h_con, label=f"ACO con Or-opt (coste={h_con[-1]:.2f} EUR)",
            color="#27ae60", linewidth=2.2)
    ax.plot(range(1, len(h_sin) + 1), h_sin, label=f"ACO sin Or-opt (coste={h_sin[-1]:.2f} EUR)",
            color="#c0392b", linewidth=2.0, linestyle="--")
    ax.plot(range(1, len(h_sin_alto) + 1), h_sin_alto,
            label=f"ACO sin Or-opt + alto pres. (coste={h_sin_alto[-1]:.2f} EUR)",
            color="#e67e22", linewidth=1.8, linestyle=":")

    ax.set_title(f"Convergencia ACO: Con Or-opt vs Sin Or-opt\n"
                 f"(47 capitales, valor_hora={valor_hora:.2f} EUR/h, semilla=42)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Iteración", fontsize=11)
    ax.set_ylabel("Mejor Coste Acumulado (EUR)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=10, loc="upper right")
    fig.tight_layout()
    fig.savefig(ruta_salida, dpi=130)
    plt.close(fig)


def guardar_resultados_barrido(corridas, mejor_corrida, config_ganadora, vh):
    """Guarda barrido CSV, ranking JSON y resumen de la mejor configuracion."""
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)

    ranking = sorted(corridas, key=lambda c: (c.coste_eur, c.tiempo_ejecucion_s))

    for destino_dir in (DIR_SALIDA, DIR_RESULTADOS):
        # CSV de barrido
        with open(destino_dir / "barrido_aco.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(corridas[0]).keys()))
            w.writeheader()
            for c in corridas:
                w.writerow(asdict(c))

        # Ranking JSON
        with open(destino_dir / "ranking_aco.json", "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in ranking], f, ensure_ascii=False, indent=2)

        # Mejor configuracion JSON
        mejor_json = {
            "criterio": "menor coste_eur con semilla fija 42 (desempate: menor tiempo_ejecucion_s)",
            "valor_hora_eur": vh,
            "semilla": SEMILLA_FIJA,
            "configuracion_ganadora": config_ganadora,
            "metricas_mejor_corrida": asdict(mejor_corrida),
        }
        with open(destino_dir / "mejor_configuracion_aco.json", "w", encoding="utf-8") as f:
            json.dump(mejor_json, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Experimentos y barrido de configuraciones ACO (semilla 42 fija)")
    parser.add_argument("--valor-hora", type=float, default=VALOR_HORA_POR_DEFECTO)
    parser.add_argument("--solo-comparar", action="store_true",
                        help="Ejecuta unicamente la comparativa Con Or-opt vs Sin Or-opt")
    args = parser.parse_args()

    vh = args.valor_hora
    datos = cargar_costes()
    M = datos.matriz_coste(vh)

    print("=" * 70)
    print("EXPERIMENTOS ACO - TSP 47 CAPITALES PENINSULARES DE ESPAÑA")
    print(f"Valor hora del vendedor : {vh:.2f} EUR/h (SMI 2026 / 8h)")
    print(f"Semilla fija            : {SEMILLA_FIJA}")
    print("=" * 70)

    t_inicio = time.perf_counter()

    if args.solo_comparar:
        from config import cargar_config
        cfg = cargar_config().get("aco", dict(BASE))
        comparacion, _, _ = comparar_con_y_sin_oropt(M, datos, vh, cfg)
        print(f"\nComparacion finalizada en {time.perf_counter() - t_inicio:.2f}s")
        return

    corridas: list[Corrida] = []

    # 1. Base
    fase_base(M, datos, vh, corridas)

    # 2. OFAT
    fase_ofat(M, datos, vh, corridas)
    elegidos = mejores_por_factor(corridas)

    # 3. Combinada
    fase_combinada(M, datos, vh, corridas, elegidos)

    # 4. Presupuesto
    fase_presupuesto(M, datos, vh, corridas)

    # Seleccionar la mejor configuracion con Or-opt
    candidatas_con_oropt = [c for c in corridas if c.usar_or_opt]
    mejor_corrida = min(candidatas_con_oropt, key=lambda c: (c.coste_eur, c.tiempo_ejecucion_s))

    config_ganadora = {
        "hormigas": mejor_corrida.hormigas,
        "iteraciones": mejor_corrida.iteraciones,
        "alfa": mejor_corrida.alfa,
        "beta": mejor_corrida.beta,
        "evaporacion": mejor_corrida.evaporacion,
        "usar_or_opt": True,
        "semilla": SEMILLA_FIJA,
    }

    print("\n" + "=" * 70)
    print("CONFIGURACION GANADORA ACO (semilla=42):")
    for k, v in config_ganadora.items():
        print(f"  {k:15s}: {v}")
    print(f"  Coste obtenido : {mejor_corrida.coste_eur:.2f} EUR "
          f"en {mejor_corrida.tiempo_ejecucion_s:.2f} s")
    print("=" * 70)

    # 5. Comparativa con y sin Or-opt
    comparacion, res_con, res_sin = comparar_con_y_sin_oropt(M, datos, vh, config_ganadora)

    # Guardar resultados y ranking
    guardar_resultados_barrido(corridas, mejor_corrida, config_ganadora, vh)

    print(f"\nPROCESO COMPLETO EN {time.perf_counter() - t_inicio:.2f}s")
    print(f"Archivos guardados en {DIR_RESULTADOS} y {DIR_SALIDA}")
    print("=" * 70)


if __name__ == "__main__":
    main()
