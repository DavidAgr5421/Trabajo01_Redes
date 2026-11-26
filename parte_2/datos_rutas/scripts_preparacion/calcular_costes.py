"""
Calculo exacto de peajes oficiales 2026 y combustible para los 2.162 pares
dirigidos entre las 47 capitales peninsulares.

Entradas:
  parte_2/datos_rutas/rutas.csv
  parte_2/datos_rutas/tramos_peaje_rutas.json
  parte_2/datos_rutas/tabla_peajes_2026.csv

Salidas:
  parte_2/datos_rutas/rutas_costes.csv
  parte_2/datos_rutas/metadata_costes.json
  parte_2/datos_rutas/rutas.csv (columnas tax y gas actualizadas)
  parte_2/Optimizacion_Combinatoria/Base_de_Datos/rutas.csv (sincronizado)
  parte_2/Optimizacion_Combinatoria/Base_de_Datos/rutas_costes.csv (sincronizado)

Vehiculo: Toyota Corolla 140H e-CVT 1.8, gasolina 95, 4.5 L/100km (WLTP 4.4-4.7).
Combustible: 1.785 EUR/L, media nacional gasolina 95 E5 MITECO del 09/10/2026.
Peajes: Tramos detectados en Geoapify Routing API / OSRM y tarifas oficiales 2026
        (corte normativo 2026-11-25, incluyendo reversion AP-68 La Rioja/Aragon).
"""
from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent
P2_DIR = DATA_DIR.parent
LEGACY_DB_DIR = P2_DIR / "Optimizacion_Combinatoria" / "Base_de_Datos"

RUTAS_CSV = DATA_DIR / "rutas.csv"
TRAMOS_JSON = DATA_DIR / "tramos_peaje_rutas.json"
TABLA_PEAJES_CSV = DATA_DIR / "tabla_peajes_2026.csv"
SALIDA_CSV = DATA_DIR / "rutas_costes.csv"
METADATA_JSON = DATA_DIR / "metadata_costes.json"

VEHICLE_MODEL = "Toyota Corolla 140H e-CVT 1.8"
FUEL_TYPE = "Gasolina 95 E5"
CONSUMPTION_L_100KM = 4.5
FUEL_PRICE_EUR_L = 1.785
FUEL_DATE = "2026-10-09"
METHOD = "geoapify_osrm_official_2026"

# Tarifas oficiales fijas/combinadas 2026 (EUR, ligeros IVA incl.)
COMBINED_AP6_AP51_EUR = 11.85   # Madrid - Avila (Villalba - Villacastin - Avila)
COMBINED_AP6_AP61_EUR = 8.60    # Madrid - Segovia (Villalba - San Rafael - Segovia)
AP9_CORUNA_SANTIAGO = 8.80
AP9_SANTIAGO_PONTEVEDRA = 7.55
AP9_CORUNA_PONTEVEDRA = 16.35
AP9_CORUNA_VIGO = 21.40
AP15_IRURTZUN_PAMPLONA = 2.65


def cargar_tabla_peajes(ruta: Path) -> dict[str, dict]:
    if not ruta.exists():
        raise FileNotFoundError(f"No existe {ruta}")
    tabla: dict[str, dict] = {}
    with open(ruta, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            via = row["via"].strip()
            tabla[via] = {
                "tarifa_km": float(row["tarifa_eur_km_2026"]),
                "tope": float(row["tope_tramo_completo_eur_2026"]),
                "estado": row["estado_2026"].strip(),
                "concesionaria": row["concesionaria"].strip(),
            }
    return tabla


def calcular_peaje_ruta(
    origen: str,
    destino: str,
    tramos: list[dict],
    liberadas: dict[str, float],
    tabla: dict[str, dict],
) -> tuple[float, str, str, str]:
    """Devuelve (tax_eur, toll_roads, toll_status, toll_notes)."""
    if not tramos:
        if liberadas:
            lib_txt = ", ".join(
                f"{k} ({v:.1f} km liberados)" for k, v in sorted(liberadas.items())
            )
            nota = f"Via liberada sin peaje vigente en 2026: {lib_txt}."
        else:
            nota = "Ruta verificada por red de autovias/carreteras gratuitas."
        return 0.0, "", "no_toll_on_route_verified", nota

    por_via = {t["via"]: t for t in tramos}
    desglose: list[str] = []
    total = 0.0
    usadas: set[str] = set()

    # 1. Reglas combinadas oficiales Castellana (AP-6 + AP-51 / AP-61)
    if "AP-6" in por_via and "AP-51" in por_via:
        km6 = por_via["AP-6"]["km"]
        km51 = por_via["AP-51"]["km"]
        if km6 <= 48.0 and km51 >= 10.0:
            total += COMBINED_AP6_AP51_EUR
            desglose.append(f"AP-6+AP-51 ({km6 + km51:.1f} km -> {COMBINED_AP6_AP51_EUR:.2f} EUR)")
            usadas.update(("AP-6", "AP-51"))

    if "AP-6" in por_via and "AP-61" in por_via and "AP-6" not in usadas:
        km6 = por_via["AP-6"]["km"]
        km61 = por_via["AP-61"]["km"]
        if km6 <= 42.0 and km61 >= 10.0:
            total += COMBINED_AP6_AP61_EUR
            desglose.append(f"AP-6+AP-61 ({km6 + km61:.1f} km -> {COMBINED_AP6_AP61_EUR:.2f} EUR)")
            usadas.update(("AP-6", "AP-61"))

    # 2. Resto de vias detectadas
    par = frozenset((origen, destino))
    for t in tramos:
        via = t["via"]
        if via in usadas:
            continue
        km = float(t["km"])
        if km < 0.05:
            continue
        info = tabla.get(via)
        if not info or info["estado"] != "vigente":
            continue

        tarifa_km = info["tarifa_km"]
        tope = info["tope"]
        c_lat = t["centroid"][0]

        if via in ("TUNEL_CADI", "TUNEL_VALLVIDRERA"):
            coste = tope
        elif via == "AP-6":
            coste = tope if km >= 55.0 else min(round(km * tarifa_km, 2), tope)
        elif via == "AP-51":
            coste = tope if km >= 20.0 else min(round(km * tarifa_km, 2), tope)
        elif via == "AP-61":
            coste = tope if km >= 22.0 else min(round(km * tarifa_km, 2), tope)
        elif via == "AP-66":
            coste = tope if km >= 55.0 else min(round(km * tarifa_km, 2), tope)
        elif via == "AP-71":
            coste = tope if km >= 30.0 else min(round(km * tarifa_km, 2), tope)
        elif via == "AP-53":
            coste = tope if km >= 45.0 else min(round(km * tarifa_km, 2), tope)
        elif via == "AP-9":
            if par == frozenset(("A Coruña", "Pontevedra")):
                coste = AP9_CORUNA_PONTEVEDRA
            elif par == frozenset(("A Coruña", "Vigo")):
                coste = AP9_CORUNA_VIGO
            elif "A Coruña" in par and 60.0 <= km <= 78.0:
                coste = AP9_CORUNA_SANTIAGO
            elif "Pontevedra" in par and 48.0 <= km <= 65.0:
                coste = AP9_SANTIAGO_PONTEVEDRA
            elif "Pontevedra" in par and 120.0 <= km <= 145.0:
                coste = AP9_CORUNA_PONTEVEDRA
            else:
                coste = min(round(km * tarifa_km, 2), tope)
        elif via == "AP-15":
            if c_lat >= 42.80 and 13.0 <= km <= 25.0:
                coste = AP15_IRURTZUN_PAMPLONA
            else:
                coste = min(round(km * tarifa_km, 2), tope)
        else:
            coste = min(round(km * tarifa_km, 2), tope)

        if coste > 0.0:
            total += coste
            desglose.append(f"{via} ({km:.1f} km -> {coste:.2f} EUR)")
            usadas.add(via)

    total = round(total, 2)
    if total <= 0.0:
        return (
            0.0,
            "",
            "no_toll_on_route_verified",
            "Ruta verificada sin peaje vigente en 2026.",
        )

    roads = " + ".join(sorted(usadas))
    notes = "Tarifas oficiales 2026: " + "; ".join(desglose) + "."
    if liberadas:
        lib_txt = ", ".join(f"{k} ({v:.1f} km)" for k, v in sorted(liberadas.items()))
        notes += f" Tramos liberados excluidos: {lib_txt}."
    return total, roads, "valued_official_2026", notes


def main() -> None:
    if not RUTAS_CSV.exists():
        raise FileNotFoundError(f"No existe {RUTAS_CSV}")
    if not TRAMOS_JSON.exists():
        raise FileNotFoundError(f"No existe {TRAMOS_JSON}. Ejecuta primero extraer_peajes_rutas.py")

    tabla = cargar_tabla_peajes(TABLA_PEAJES_CSV)
    with open(TRAMOS_JSON, encoding="utf-8") as f:
        tramos_data = json.load(f)
    rutas_tramos = tramos_data.get("rutas", {})

    with open(RUTAS_CSV, encoding="utf-8-sig", newline="") as f:
        filas_rutas = list(csv.DictReader(f))
        rutas_fields = list(filas_rutas[0].keys()) if filas_rutas else [
            "route", "origin", "destination", "tax", "gas", "distance", "time"
        ]

    salida = []
    rutas_actualizadas = []
    n_valued = 0
    n_zero = 0
    vias_freq: Counter[str] = Counter()

    for row in filas_rutas:
        origen = str(row.get("origin", "")).strip()
        destino = str(row.get("destination", "")).strip()
        if not origen or not destino or origen == destino:
            continue

        distancia = float(str(row["distance"]).replace(",", "."))
        tiempo = float(str(row["time"]).replace(",", "."))

        clave = f"{origen} -> {destino}"
        if clave not in rutas_tramos:
            raise KeyError(f"Falta el par '{clave}' en {TRAMOS_JSON}")

        info_ruta = rutas_tramos[clave]
        tramos = info_ruta.get("tramos", [])
        liberadas = info_ruta.get("liberadas", {})

        tax, roads, status, notes = calcular_peaje_ruta(
            origen, destino, tramos, liberadas, tabla
        )
        if status == "valued_official_2026":
            n_valued += 1
            for t in tramos:
                if t["km"] >= 0.05 and t["via"] in tabla and tabla[t["via"]]["estado"] == "vigente":
                    vias_freq[t["via"]] += 1
        else:
            n_zero += 1

        litros = distancia * CONSUMPTION_L_100KM / 100.0
        gas = round(litros * FUEL_PRICE_EUR_L, 2)

        salida.append(
            {
                "route": row.get("route", ""),
                "origin": origen,
                "destination": destino,
                "distance_km": round(distancia, 3),
                "time_h": round(tiempo, 6),
                "tax_eur": tax,
                "toll_method": METHOD,
                "toll_status": status,
                "toll_roads": roads,
                "toll_source": "tabla_peajes_2026.csv + tramos_peaje_rutas.json + fuentes.csv",
                "toll_notes": notes,
                "fuel_litres": round(litros, 3),
                "fuel_price_eur_l": FUEL_PRICE_EUR_L,
                "gas_eur": gas,
                "vehicle_model": VEHICLE_MODEL,
                "consumption_L_100km": CONSUMPTION_L_100KM,
                "fuel_type": FUEL_TYPE,
                "fuel_date": FUEL_DATE,
            }
        )

        row_upd = dict(row)
        row_upd["tax"] = f"{tax:.2f}"
        row_upd["gas"] = f"{gas:.2f}"
        rutas_actualizadas.append(row_upd)

    campos = [
        "route", "origin", "destination", "distance_km", "time_h",
        "tax_eur", "toll_method", "toll_status", "toll_roads",
        "toll_source", "toll_notes", "fuel_litres", "fuel_price_eur_l",
        "gas_eur", "vehicle_model", "consumption_L_100km", "fuel_type",
        "fuel_date",
    ]
    with open(SALIDA_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        w.writerows(salida)

    # Actualizar columnas tax y gas de rutas.csv para coincidir exactamente
    with open(RUTAS_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rutas_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rutas_actualizadas)

    # Sincronizar hacia parte_2/Optimizacion_Combinatoria/Base_de_Datos/
    LEGACY_DB_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(RUTAS_CSV, LEGACY_DB_DIR / "rutas.csv")
    shutil.copy2(SALIDA_CSV, LEGACY_DB_DIR / "rutas_costes.csv")

    metadata = {
        "fecha_generacion_utc": datetime.now(timezone.utc).isoformat(),
        "metodo": METHOD,
        "vehiculo": VEHICLE_MODEL,
        "combustible": FUEL_TYPE,
        "consumo_L_100km": CONSUMPTION_L_100KM,
        "precio_litro_eur": FUEL_PRICE_EUR_L,
        "precio_fecha": FUEL_DATE,
        "precio_fuente": "MITECO Geoportal (media nacional G95 09/10/2026)",
        "fecha_corte_peajes": "2026-11-25",
        "pares_totales": len(salida),
        "pares_con_peaje_valorado": n_valued,
        "pares_sin_peaje_verificado": n_zero,
        "pares_peaje_posible_no_valorado": 0,
        "frecuencia_vias_peaje": dict(vias_freq.most_common()),
        "formula_gas": "distance_km * 4.5/100 * 1.785",
        "formula_coste": "C = valor_hora*time_h + tax_eur + gas_eur",
        "entradas": [
            "rutas.csv",
            "tramos_peaje_rutas.json",
            "tabla_peajes_2026.csv",
            "fuentes.csv",
        ],
        "salidas": [
            "rutas_costes.csv",
            "rutas.csv",
            "Optimizacion_Combinatoria/Base_de_Datos/rutas.csv",
            "Optimizacion_Combinatoria/Base_de_Datos/rutas_costes.csv",
        ],
    }
    with open(METADATA_JSON, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(
        f"Pares: {len(salida)} | con peaje valorado: {n_valued} | "
        f"sin peaje verificado: {n_zero} | posibles no valorados: 0"
    )
    print(f"Salida: {SALIDA_CSV}")
    print(f"Sincronizado en: {LEGACY_DB_DIR}")
    print(f"Metadatos: {METADATA_JSON}")


if __name__ == "__main__":
    main()
