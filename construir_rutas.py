
import os
import json
import time
from datetime import datetime
from pathlib import Path

import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEOAPIFY_API_KEY")

if not API_KEY:
    raise ValueError(
        "No se encontró GEOAPIFY_API_KEY en el archivo .env"
    )

BASE_DIR = Path("datos_rutas")
BASE_DIR.mkdir(exist_ok=True)

# Cambia este valor a 0, 1 y 2 en ejecuciones separadas.
# Cada bloque se guarda para no tener que volver a consultarlo.
BATCH_INDEX = 5

GEOCODE_URL = "https://api.geoapify.com/v1/geocode/search"
MATRIX_URL = "https://api.geoapify.com/v1/routematrix"

CIUDADES = [
    "A Coruña", "Albacete", "Alicante", "Almería", "Ávila",
    "Badajoz", "Barcelona", "Bilbao", "Burgos", "Cáceres",
    "Cádiz", "Castellón de la Plana", "Ciudad Real", "Córdoba",
    "Cuenca", "Girona", "Granada", "Guadalajara", "Huelva",
    "Huesca", "Jaén", "León", "Lleida", "Logroño", "Lugo",
    "Madrid", "Málaga", "Murcia", "Ourense", "Oviedo",
    "Palencia", "Pamplona", "Pontevedra", "Salamanca",
    "San Sebastián", "Santander", "Segovia", "Sevilla", "Soria",
    "Tarragona", "Teruel", "Toledo", "Valencia", "Valladolid",
    "Vitoria-Gasteiz", "Zamora", "Zaragoza"
]

assert len(CIUDADES) == 47
assert len(set(CIUDADES)) == 47


def obtener_coordenadas():
    """Obtiene coordenadas y prueba consultas alternativas si falla."""
    archivo = BASE_DIR / "ciudades.csv"

    if archivo.exists():
        ciudades = pd.read_csv(archivo)
        if len(ciudades) == 47:
            print("Reutilizando ciudades.csv")
            return ciudades

    # Variantes para nombres que pueden dar problemas.
    variantes = {
        "A Coruña": ["A Coruña"],
        "Ourense": ["Ourense", "Orense"],
        "San Sebastián": [
            "Donostia-San Sebastián",
            "San Sebastián"
        ],
        "Pamplona": ["Pamplona", "Iruña"]
    }

    resultados = []

    for i, ciudad in enumerate(CIUDADES, start=1):
        consultas = variantes.get(
            ciudad,
            [f"{ciudad}, España"]
        )

        lugar = None

        for consulta in consultas:
            params = {
                "text": consulta,
                "type": "city",
                "filter": "countrycode:es",
                "limit": 5,
                "format": "json",
                "apiKey": API_KEY
            }       

            response = requests.get(
                GEOCODE_URL,
                params=params,
                timeout=30
            )

            if not response.ok:
                print("Error HTTP:", response.status_code)
                print(response.text[:1000])
                response.raise_for_status()

            data = response.json()
            print("Consulta enviada:", response.url.replace(API_KEY, "***"))
            print("HTTP:", response.status_code)
            print("Resultados recibidos:", len(data.get("results", [])))
            print("Primer resultado:", data.get("results", [])[:1])

            # Evitar aceptar resultados de otro país.
            opciones = [
                r for r in data.get("results", [])
                if r.get("country_code", "").lower() == "es"
            ]

            # Preferir resultados administrativos o ciudades.
            opciones.sort(
                key=lambda r: (
                    0 if r.get("result_type") in
                    ("city", "municipality", "county") else 1
                )
            )

            if opciones:
                lugar = opciones[0]
                break

            print(f"Sin resultados para: {consulta}")

        if lugar is None:
            raise RuntimeError(
                f"No se encontraron coordenadas para {ciudad}. "
                "Revisa la consulta y la respuesta de Geoapify."
            )

        resultados.append({
            "ciudad": ciudad,
            "lat": lugar["lat"],
            "lon": lugar["lon"],
            "resultado_geocodificacion": lugar.get("formatted", ""),
            "fecha_consulta": datetime.now().astimezone().isoformat()
        })

        print(
            f"[{i}/47] {ciudad}: "
            f"{lugar['lat']}, {lugar['lon']} | "
            f"{lugar.get('formatted', '')}"
        )

        time.sleep(0.1)

    df = pd.DataFrame(resultados)
    df.to_csv(
        archivo,
        index=False,
        encoding="utf-8-sig"
    )

    return df


def consultar_bloque(ciudades, batch_index):
    """Calcula todas las rutas desde un bloque hacia las 47 ciudades."""
    archivo = BASE_DIR / f"matrix_batch_{batch_index}.json"

    if archivo.exists():
        print(f"El bloque {batch_index} ya existe. No se consulta.")
        return

    # Bloques de 20, 20 y 7 ciudades de origen.
    inicio = batch_index * 20
    fin = min(inicio + 20, len(ciudades))

    if batch_index not in (0, 1, 2):
        raise ValueError("BATCH_INDEX debe ser 0, 1 o 2")

    origenes = ciudades.iloc[inicio:fin]
    destinos = ciudades

    payload = {
        "mode": "drive",
        "type": "balanced",
        "traffic": "free_flow",
        "sources": [
            {"location": [float(row.lon), float(row.lat)]}
            for row in origenes.itertuples()
        ],
        "targets": [
            {"location": [float(row.lon), float(row.lat)]}
            for row in destinos.itertuples()
        ]
    }

    print(
        f"Consultando bloque {batch_index}: "
        f"{len(origenes)} × {len(destinos)} "
        f"= {len(origenes) * len(destinos)} pares"
    )

    response = requests.post(
        MATRIX_URL,
        params={"apiKey": API_KEY},
        json=payload,
        timeout=180
    )

    if not response.ok:
        print("Error de Geoapify:", response.status_code)
        print(response.text[:2000])
        response.raise_for_status()

    data = response.json()

    if "sources_to_targets" not in data:
        raise RuntimeError(
            "La respuesta no contiene sources_to_targets. "
            "No se guardó el bloque."
        )

    # Guardar la respuesta completa para poder auditarla.
    with open(archivo, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Bloque guardado en {archivo}")


def exportar_resultados(ciudades):
    """Exporta rutas y matrices con nombres de columnas normalizados."""
    rutas = []
    bloques_completos = []

    for batch_index in range(3):
        archivo = BASE_DIR / f"matrix_batch_{batch_index}.json"

        if not archivo.exists():
            continue

        with open(archivo, encoding="utf-8") as f:
            data = json.load(f)

        inicio = batch_index * 20
        matriz = data["sources_to_targets"]

        for i, fila in enumerate(matriz):
            idx_origen = inicio + i
            origin = ciudades.iloc[idx_origen]["ciudad"]

            for j, item in enumerate(fila):
                destination = ciudades.iloc[j]["ciudad"]

                distance = item.get("distance")
                time_seconds = item.get("time")

                # Una ciudad no requiere viajar hacia sí misma.
                if idx_origen == j:
                    distance = 0
                    time_seconds = 0

                rutas.append({
                    "route": f"R{idx_origen + 1:02d}_{j + 1:02d}",
                    "origin": origin,
                    "destination": destination,
                    "tax": 0.0,  # Provisional: aún no calculado
                    "gas": 0.0,  # Provisional: aún no calculado
                    "distance": (
                        distance / 1000
                        if distance is not None else None
                    ),
                    "time": (
                        time_seconds / 3600
                        if time_seconds is not None else None
                    )
                })

        bloques_completos.append(batch_index)

    if not rutas:
        print("Todavía no hay bloques para exportar.")
        return

    df = pd.DataFrame(
        rutas,
        columns=[
            "route", "origin", "destination",
            "tax", "gas", "distance", "time"
        ]
    )

    df.to_csv(
        BASE_DIR / "rutas.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Las matrices se mantienen separadas para ACO y GA.
    for columna, nombre_archivo in [
        ("distance", "matriz_distancias.csv"),
        ("time", "matriz_tiempos.csv")
    ]:
        matriz = df.pivot(
            index="origin",
            columns="destination",
            values=columna
        )

        matriz = matriz.reindex(
            index=CIUDADES,
            columns=CIUDADES
        )

        matriz.to_csv(
            BASE_DIR / nombre_archivo,
            encoding="utf-8-sig"
        )

    metadata = {
        "fecha_exportacion": datetime.now().astimezone().isoformat(),
        "ciudades": len(ciudades),
        "rutas_exportadas": len(df),
        "pares_esperados": len(ciudades) ** 2,
        "bloques_disponibles": bloques_completos,
        "bloques_pendientes": [
            i for i in range(3)
            if i not in bloques_completos
        ],
        "columnas": [
            "route", "origin", "destination",
            "tax", "gas", "distance", "time"
        ],
        "unidades": {
            "tax": "EUR",
            "gas": "EUR",
            "distance": "km",
            "time": "hours"
        },
        "tax_status": "provisional_not_calculated",
        "gas_status": "provisional_not_calculated",
        "mode": "drive",
        "traffic": "free_flow",
        "source": "Geoapify Route Matrix API"
    }

    with open(
        BASE_DIR / "metadata.json", "w", encoding="utf-8"
    ) as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\nExportación actualizada.")
    print(f"Rutas exportadas: {len(df)}")
    print(f"Columnas: {list(df.columns)}")
    print(f"Bloques completados: {bloques_completos}")
    print(f"Bloques pendientes: {metadata['bloques_pendientes']}")
    print(f"Archivos guardados en: {BASE_DIR.resolve()}")


if __name__ == "__main__":
    ciudades_df = obtener_coordenadas()

    if BATCH_INDEX in (0, 1, 2):
        consultar_bloque(ciudades_df, BATCH_INDEX)

    exportar_resultados(ciudades_df)