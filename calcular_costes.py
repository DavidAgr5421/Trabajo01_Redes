"""
Costes de peaje y combustible (opción B: inventario oficial).
Solo usa la librería estándar para no exigir dependencias.

Entradas:
  datos_rutas/rutas.csv        (route, origin, destination, tax, gas, distance, time)
  datos_rutas/tabla_peajes_2026.csv
  datos_rutas/fuentes.csv

Salidas:
  datos_rutas/rutas_costes.csv
  datos_rutas/metadata_costes.json

Criterio de ruta de referencia (documentado):
  La matriz Geoapify es 'balanced + free_flow' sin geometría ni desglose
  con/sin peaje. Para el TSP se fija como referencia la ruta rápida por la
  red principal. Donde la autopista de peaje es el eje principal sin
  alternativa gratuita equivalente (AP-9 A Coruña-Pontevedra, AP-6/AP-51/AP-61
  Madrid-Segovia/Ávila, AP-66 Oviedo-Meseta) se suma el peaje oficial 2026.
  Donde existe autovía gratuita paralela (radiales SEITT, AP-2/A-2, A-3/A-4/A-5,
  A-7, C-32/AP-7, túneles) la referencia es la vía gratuita (tax=0) y el peaje
  queda documentado en tabla_peajes_2026.csv para el análisis de sensibilidad.
  No se presenta ninguna estimación como tarifa exacta puerta a puerta.

Vehículo: Toyota Corolla 140H e-CVT 1.8, gasolina 95, 4.5 L/100km
  (centro del rango WLTP oficial 4.4-4.7; ver fuentes.csv ids 23-24).
Precio: 1.785 EUR/L, media nacional gasolina 95 E5 MITECO del 09/10/2026
  (ver fuentes.csv ids 16-22; aviso RD-ley 25/2026 hasta 31/12/2026).
"""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datos_rutas"
RUTAS_CSV = DATA_DIR / "rutas.csv"
SALIDA_CSV = DATA_DIR / "rutas_costes.csv"
METADATA_JSON = DATA_DIR / "metadata_costes.json"

# ---------- Parámetros fijados ----------
VEHICLE_MODEL = "Toyota Corolla 140H e-CVT 1.8"
FUEL_TYPE = "Gasolina 95 E5"
CONSUMPTION_L_100KM = 4.5
FUEL_PRICE_EUR_L = 1.785
FUEL_DATE = "2026-10-09"
METHOD = "official_inventory_reference_B"

# Peajes fijos oficiales 2026 (fuente: tabla_peajes_2026.csv).
TOLL_MADRID_SEGOVIA = 7.60    # AP-6/AP-61 hora valle
TOLL_MADRID_AVILA = 11.85     # AP-6/AP-51
TOLL_CORUNA_PONTEVEDRA = 16.35  # AP-9
TOLL_AP66 = 16.20             # Campomanes-León

# Oviedo: el Huerna (AP-66) es el paso obligado hacia la Meseta y el sur.
# Hacia el norte (Galicia, Cantábrico) y el este (País Vasco, Rioja, Aragón,
# Cataluña) la referencia usa el corredor cantábrico/pirenaico, sin AP-66.
OVIEDO_SUR_MESETA = {
    "León", "Zamora", "Salamanca", "Valladolid", "Palencia", "Burgos",
    "Soria", "Madrid", "Segovia", "Ávila", "Guadalajara", "Toledo",
    "Ciudad Real", "Cuenca", "Albacete", "Cáceres", "Badajoz", "Huelva",
    "Sevilla", "Cádiz", "Córdoba", "Jaén", "Granada", "Almería", "Málaga",
    "Murcia", "Alicante", "Valencia", "Castellón de la Plana", "Teruel",
}

# Red vasca + AP-68 + AP-71 + AP-53: hay peaje posible para ligeros pero no se
# dispone de tarifa fija verificada hoy; se marca para ampliación manual y NO
# se inventa ningún valor (tax=0 + flag). Ver toll_status.
VASCO_NAVARRO_RIOJA = {
    "Bilbao", "San Sebastián", "Vitoria-Gasteiz", "Pamplona", "Logroño",
    "Santander",
}
AP68_CORREDOR_ESTE = {"Zaragoza", "Huesca", "Lleida", "Barcelona", "Girona", "Tarragona"}
AP53_CORREDOR = {"Ourense", "Lugo", "Pontevedra"}


def par(a, b):
    return frozenset((a, b))


def asignar_peaje(origen, destino):
    """Devuelve (tax, toll_roads, toll_status, toll_notes)."""
    if origen == destino:
        return 0.0, "", "no_toll_expected_official_network", "Misma ciudad."

    p = par(origen, destino)

    if p == par("A Coruña", "Pontevedra"):
        return (TOLL_CORUNA_PONTEVEDRA, "AP-9 A Coruña-Pontevedra",
                "valued_official_2026",
                "Suma de tramos oficiales Audasa 2026: 8.80+7.55.")
    if p == par("Madrid", "Segovia"):
        return (TOLL_MADRID_SEGOVIA, "AP-6/AP-61 Madrid-Segovia",
                "valued_official_2026",
                "Tarifa valle ligeros 2026; punta 10.45 (ver tabla).")
    if p == par("Madrid", "Ávila"):
        return (TOLL_MADRID_AVILA, "AP-6/AP-51 Madrid-Ávila",
                "valued_official_2026",
                "10.10 Villalba-Villacastín + 1.75 Villacastín-Ávila.")
    if "Oviedo" in p:
        otro = destino if origen == "Oviedo" else origen
        if otro in OVIEDO_SUR_MESETA:
            return (TOLL_AP66, "AP-66 Campomanes-León",
                    "valued_official_2026",
                    "Paso del Huerna hacia la Meseta/sur; sin alternativa "
                    "rápida equivalente.")
        return (0.0, "", "no_toll_expected_official_network",
                "Corredor cantábrico/pirenaico sin AP-66 en la referencia.")
    if (origen in VASCO_NAVARRO_RIOJA and destino in VASCO_NAVARRO_RIOJA) or \
       (origen == "Bilbao" and destino in AP68_CORREDOR_ESTE) or \
       (destino == "Bilbao" and origen in AP68_CORREDOR_ESTE):
        return (0.0, "AP-68 / red foral (Bidegi)",
                "toll_possible_not_valued",
                "Peaje posible para ligeros; tarifa no verificada hoy. "
                "Ampliar manual con Bidegi/autopistas.com. tax=0 provisional.")
    if (origen in AP53_CORREDOR and destino in AP53_CORREDOR) or \
       (origen == "León" and destino in AP53_CORREDOR) or \
       (destino == "León" and origen in AP53_CORREDOR):
        return (0.0, "AP-53 / AP-71",
                "toll_possible_not_valued",
                "Peaje posible; tarifa no verificada hoy. tax=0 provisional.")
    return (0.0, "", "no_toll_expected_official_network",
            "Red de referencia gratuita post-liberalizaciones "
            "(AP-1/AP-2/AP-4/AP-7 liberadas); ver tabla_peajes_2026.csv.")


def main():
    if not RUTAS_CSV.exists():
        raise FileNotFoundError(f"No existe {RUTAS_CSV}")

    with open(RUTAS_CSV, encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        filas = list(csv.DictReader(f, dialect=dialect))

    salida = []
    n_valued = n_zero = n_flag = 0
    for row in filas:
        origen = str(row.get("origin", "")).strip()
        destino = str(row.get("destination", "")).strip()
        if not origen or not destino or origen == destino:
            continue
        try:
            distancia = float(str(row.get("distance", "")).replace(",", "."))
        except ValueError:
            raise ValueError(f"Distancia inválida en {origen}->{destino}")
        try:
            tiempo = float(str(row.get("time", "")).replace(",", "."))
        except ValueError:
            raise ValueError(f"Tiempo inválido en {origen}->{destino}")

        tax, roads, status, notes = asignar_peaje(origen, destino)
        if status == "valued_official_2026":
            n_valued += 1
        elif status == "toll_possible_not_valued":
            n_flag += 1
        else:
            n_zero += 1

        litros = distancia * CONSUMPTION_L_100KM / 100.0
        gas = litros * FUEL_PRICE_EUR_L

        salida.append({
            "route": row.get("route", ""),
            "origin": origen,
            "destination": destino,
            "distance_km": round(distancia, 3),
            "time_h": round(tiempo, 6),
            "tax_eur": round(tax, 2),
            "toll_method": METHOD,
            "toll_status": status,
            "toll_roads": roads,
            "toll_source": "tabla_peajes_2026.csv + fuentes.csv (MITMA/BOE/concesionarias)",
            "toll_notes": notes,
            "fuel_litres": round(litros, 3),
            "fuel_price_eur_l": FUEL_PRICE_EUR_L,
            "gas_eur": round(gas, 2),
            "vehicle_model": VEHICLE_MODEL,
            "consumption_L_100km": CONSUMPTION_L_100KM,
            "fuel_type": FUEL_TYPE,
            "fuel_date": FUEL_DATE,
        })

    campos = ["route", "origin", "destination", "distance_km", "time_h",
              "tax_eur", "toll_method", "toll_status", "toll_roads",
              "toll_source", "toll_notes", "fuel_litres", "fuel_price_eur_l",
              "gas_eur", "vehicle_model", "consumption_L_100km", "fuel_type",
              "fuel_date"]
    with open(SALIDA_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        w.writerows(salida)

    metadata = {
        "fecha_generacion_utc": datetime.now(timezone.utc).isoformat(),
        "metodo": METHOD,
        "vehiculo": VEHICLE_MODEL,
        "combustible": FUEL_TYPE,
        "consumo_L_100km": CONSUMPTION_L_100KM,
        "precio_litro_eur": FUEL_PRICE_EUR_L,
        "precio_fecha": FUEL_DATE,
        "precio_fuente": "MITECO Geoportal (media nacional G95 09/10/2026)",
        "pares_totales": len(salida),
        "pares_con_peaje_valorado": n_valued,
        "pares_sin_peaje_esperado": n_zero,
        "pares_peaje_posible_no_valorado": n_flag,
        "formula_gas": "distance_km * 4.5/100 * 1.785",
        "formula_coste": "C = valor_hora*time_h + tax_eur + gas_eur",
        "entradas": ["rutas.csv", "tabla_peajes_2026.csv", "fuentes.csv"],
        "salida": "rutas_costes.csv",
        "advertencia": "tax es la tarifa oficial del eje de referencia, no el "
                       "cargo exacto puerta a puerta (depende de entradas, "
                       "hora, Vía-T y descuentos).",
    }
    with open(METADATA_JSON, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Pares: {len(salida)} | con peaje: {n_valued} | "
          f"sin peaje: {n_zero} | posibles no valorados: {n_flag}")
    print(f"Salida: {SALIDA_CSV}")
    print(f"Metadatos: {METADATA_JSON}")


if __name__ == "__main__":
    main()
