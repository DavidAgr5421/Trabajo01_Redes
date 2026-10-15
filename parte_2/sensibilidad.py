"""
Estudio de Sensibilidad del Valor de la Hora del Vendedor en el TSP peninsular.

Responde formalmente a las preguntas:
  - ¿Cambia la ruta optima al variar el valor de la hora del vendedor?
  - ¿A partir de que umbral se modifica el recorrido?
  - ¿Que regimen operativo predomina en cada rango?
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

from colonia_hormigas import ejecutar_aco
from costes_comunes import BASE_DIR, VALOR_HORA_POR_DEFECTO, cargar_costes

DIR_RESULTADOS = BASE_DIR / "resultados"
DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)


def analizar_sensibilidad(valores_hora: list[float] | None = None, demo: bool = False) -> pd.DataFrame:
    if valores_hora is None:
        if demo:
            valores_hora = [0.0, 5.09, 20.0, 50.0]
        else:
            valores_hora = [0.0, 2.0, 5.09, 10.0, 20.0, 35.0, 50.0, 100.0]

    datos = cargar_costes()
    hormigas = 10 if demo else 20
    iteraciones = 30 if demo else 150
    semilla = 42

    print("=" * 78)
    print("ESTUDIO DE SENSIBILIDAD DEL VALOR DE LA HORA DEL VENDEDOR (TSP 47 CAPITALES)")
    print(f"Configuracion: {hormigas} hormigas, {iteraciones} iteraciones, semilla={semilla}")
    print("=" * 78)

    # 1. Obtener primero el tour base de 5.09 EUR/h
    M_base = datos.matriz_coste(VALOR_HORA_POR_DEFECTO)
    res_base = ejecutar_aco(
        M_base,
        hormigas=hormigas,
        iteraciones=iteraciones,
        alfa=1.0,
        beta=3.0,
        evaporacion=0.4,
        semilla=semilla,
        usar_or_opt=True,
        verbose=False,
    )
    firma_base = datos.firma(res_base["tour"])

    filas = []
    for vh in valores_hora:
        t0 = time.time()
        M = datos.matriz_coste(vh)
        res = ejecutar_aco(
            M,
            hormigas=hormigas,
            iteraciones=iteraciones,
            alfa=1.0,
            beta=3.0,
            evaporacion=0.4,
            semilla=semilla,
            usar_or_opt=True,
            verbose=False,
        )
        desglose = datos.desglose_tour(res["tour"], vh)
        firma_actual = datos.firma(res["tour"])
        es_identica = (firma_actual == firma_base)

        if vh == 0.0:
            regimen = "Minima distancia / Combustible puro"
        elif vh < 15.0:
            regimen = "Equilibrio Economico Laboral (Base)"
        elif vh < 40.0:
            regimen = "Reduccion Temporal (Mayor velocidad media)"
        else:
            regimen = "Dominancia Severa del Tiempo Laboral"

        dt = time.time() - t0
        filas.append({
            "Valor_Hora_EUR_h": vh,
            "Coste_Total_EUR": round(desglose["coste_eur"], 2),
            "Distancia_km": round(desglose["distancia_km"], 2),
            "Tiempo_h": round(desglose["tiempo_h"], 2),
            "Coste_Combustible_EUR": round(desglose["gas_eur"], 2),
            "Coste_Tiempo_EUR": round(desglose["coste_eur"] - desglose["gas_eur"] - desglose["tax_eur"], 2),
            "Ruta_Identica_a_Base": "Si" if es_identica else "No",
            "Regimen_Operativo": regimen,
            "Tiempo_Calculo_s": round(dt, 2),
        })
        print(f"  [VH = {vh:6.2f} EUR/h] Coste: {desglose['coste_eur']:8.2f} EUR | "
              f"Dist: {desglose['distancia_km']:7.1f} km | Tiempo: {desglose['tiempo_h']:5.2f} h | "
              f"Identica al base: {'SI' if es_identica else 'NO'}")

    df = pd.DataFrame(filas)
    out_csv = DIR_RESULTADOS / "sensibilidad_valor_hora.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    out_json = DIR_RESULTADOS / "sensibilidad_valor_hora.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(filas, f, ensure_ascii=False, indent=2)

    print("=" * 78)
    print(f"[OK] Resultados de sensibilidad guardados en:\n  - {out_csv}\n  - {out_json}")
    print("=" * 78)
    return df


def main():
    parser = argparse.ArgumentParser(description="Estudio de sensibilidad del valor de hora del vendedor")
    parser.add_argument("--demo", action="store_true", help="Barrido rapido (< 10 s)")
    args = parser.parse_args()
    analizar_sensibilidad(demo=args.demo)


if __name__ == "__main__":
    main()
