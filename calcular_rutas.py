
import os
import json
import time
from datetime import datetime, timezone

import pandas as pd
import requests

# ---------- CONFIGURACIÓN ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "datos_rutas")

CIUDADES_FILE = os.path.join(DATA_DIR, "ciudades.csv")
CHECKPOINT_FILE = os.path.join(DATA_DIR, "checkpoint_matriz.json")
DISTANCIAS_FILE = os.path.join(DATA_DIR, "matriz_distancias.csv")
TIEMPOS_FILE = os.path.join(DATA_DIR, "matriz_tiempos.csv")
RUTAS_FILE = os.path.join(DATA_DIR, "rutas.csv")
METADATA_FILE = os.path.join(DATA_DIR, "metadata.json")

API_URL = "https://api.geoapify.com/v1/routematrix"
API_KEY = os.getenv("GEOAPIFY_API_KEY")

TAM_BLOQUE = 5
PAUSA_SEGUNDOS = 1

# ---------- VALIDACIONES ----------
if not API_KEY:
    raise RuntimeError(
        "No se encontró GEOAPIFY_API_KEY. "
        "Configúrala en la terminal antes de ejecutar."
    )

if not os.path.exists(CIUDADES_FILE):
    raise FileNotFoundError(
        f"No existe {CIUDADES_FILE}. "
        "Guarda primero las coordenadas en datos_rutas/ciudades.csv."
    )

os.makedirs(DATA_DIR, exist_ok=True)

ciudades_df = pd.read_csv(CIUDADES_FILE)

columnas_requeridas = {"ciudad", "lat", "lon"}
if not columnas_requeridas.issubset(ciudades_df.columns):
    raise ValueError(
        f"El CSV debe contener estas columnas: {columnas_requeridas}"
    )

if ciudades_df["ciudad"].duplicated().any():
    raise ValueError("Hay nombres de ciudades duplicados.")

if ciudades_df[["lat", "lon"]].isnull().any().any():
    raise ValueError("Hay coordenadas vacías.")

ciudades = ciudades_df["ciudad"].astype(str).tolist()

# ---------- CHECKPOINT ----------
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        checkpoint = json.load(f)
    print("Checkpoint encontrado. Se reutilizarán los datos guardados.")
else:
    checkpoint = {}

def guardar_checkpoint():
    temporal = CHECKPOINT_FILE + ".tmp"
    with open(temporal, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False)
    os.replace(temporal, CHECKPOINT_FILE)

# ---------- API ----------
def consultar_bloque(origenes, destinos, profundidad=0):
    """Consulta un bloque; si falla, lo divide para reducir su tamaño."""

    payload = {
        "mode": "drive",
        "type": "balanced",
        "traffic": "free_flow",
        "sources": [
            {"location": [float(c["lon"]), float(c["lat"])]}
            for c in origenes
        ],
        "targets": [
            {"location": [float(c["lon"]), float(c["lat"])]}
            for c in destinos
        ],
    }

    try:
        response = requests.post(
            API_URL,
            params={"apiKey": API_KEY},
            json=payload,
            timeout=120,
        )

        if not response.ok:
            # No imprimir la URL completa: contiene la clave API.
            mensaje = response.text[:1000]
            raise RuntimeError(
                f"HTTP {response.status_code}: {mensaje}"
            )

        data = response.json()
        matriz = data.get("sources_to_targets")

        if not isinstance(matriz, list):
            raise RuntimeError(
                "La respuesta no contiene sources_to_targets."
            )

        if len(matriz) != len(origenes):
            raise RuntimeError("Número inesperado de filas en la matriz.")

        for i, origen in enumerate(origenes):
            if len(matriz[i]) != len(destinos):
                raise RuntimeError(
                    "Número inesperado de destinos en una fila."
                )

            nombre_origen = str(origen["ciudad"])
            checkpoint.setdefault(nombre_origen, {})

            for j, destino in enumerate(destinos):
                celda = matriz[i][j]
                nombre_destino = str(destino["ciudad"])

                checkpoint[nombre_origen][nombre_destino] = {
                    "distance": celda.get("distance"),
                    "time": celda.get("time"),
                }

        guardar_checkpoint()
        time.sleep(PAUSA_SEGUNDOS)

    except Exception as error:
        if len(origenes) > 1:
            mitad = max(1, len(origenes) // 2)
            print(
                f"Bloque de {len(origenes)} orígenes falló. "
                f"Se divide en bloques de {mitad} y "
                f"{len(origenes) - mitad}."
            )
            consultar_bloque(origenes[:mitad], destinos, profundidad + 1)
            consultar_bloque(origenes[mitad:], destinos, profundidad + 1)
            return

        # Un único origen: reintentar solo errores transitorios.
        if profundidad < 3:
            print(
                f"Error para {origenes[0]['ciudad']}: {error}. "
                f"Reintento {profundidad + 1}/3."
            )
            time.sleep(2 ** (profundidad + 1))
            consultar_bloque(origenes, destinos, profundidad + 1)
            return

        raise RuntimeError(
            f"No se pudo consultar el origen "
            f"{origenes[0]['ciudad']}. "
            f"El checkpoint conserva el progreso anterior."
        ) from error

# ---------- CONSULTA DE DATOS FALTANTES ----------
registros = ciudades_df.to_dict(orient="records")

for inicio in range(0, len(registros), TAM_BLOQUE):
    bloque = registros[inicio:inicio + TAM_BLOQUE]

    pendientes = [
        origen for origen in bloque
        if not (
            str(origen["ciudad"]) in checkpoint
            and all(
                destino in checkpoint[str(origen["ciudad"])]
                for destino in ciudades
            )
        )
    ]

    if not pendientes:
        print(f"Bloque {inicio // TAM_BLOQUE + 1}: ya completado.")
        continue

    print(
        f"Consultando bloque {inicio // TAM_BLOQUE + 1}: "
        f"{len(pendientes)} orígenes × {len(ciudades)} destinos"
    )
    consultar_bloque(pendientes, registros)

# ---------- CONSTRUIR MATRICES ----------
distancias = pd.DataFrame(index=ciudades, columns=ciudades, dtype=float)
tiempos = pd.DataFrame(index=ciudades, columns=ciudades, dtype=float)

for origen in ciudades:
    for destino in ciudades:
        celda = checkpoint.get(origen, {}).get(destino)

        if celda is None:
            raise RuntimeError(f"Falta el trayecto {origen} → {destino}.")

        distancia = celda.get("distance")
        tiempo = celda.get("time")

        distancias.loc[origen, destino] = (
            float(distancia) / 1000 if distancia is not None else float("nan")
        )
        tiempos.loc[origen, destino] = (
            float(tiempo) / 3600 if tiempo is not None else float("nan")
        )

distancias.index.name = "ciudad"
tiempos.index.name = "ciudad"

distancias.to_csv(DISTANCIAS_FILE, encoding="utf-8-sig")
tiempos.to_csv(TIEMPOS_FILE, encoding="utf-8-sig")

# ---------- CSV DE RUTAS ----------
rutas = []
numero = 1

for origen in ciudades:
    for destino in ciudades:
        if origen == destino:
            continue

        rutas.append({
            "route": f"R{numero:04d}",
            "origin": origen,
            "destination": destino,
            # Vacíos: los costes todavía no se han investigado.
            "tax": None,
            "gas": None,
            "distance": distancias.loc[origen, destino],
            "time": tiempos.loc[origen, destino],
        })
        numero += 1

rutas_df = pd.DataFrame(
    rutas,
    columns=[
        "route", "origin", "destination",
        "tax", "gas", "distance", "time"
    ],
)
rutas_df.to_csv(RUTAS_FILE, index=False, encoding="utf-8-sig")

# ---------- METADATOS ----------
metadata = {
    "fuente_distancias_tiempos": "Geoapify Route Matrix API",
    "fecha_generacion_utc": datetime.now(timezone.utc).isoformat(),
    "numero_ciudades": len(ciudades),
    "numero_rutas_dirigidas": len(rutas_df),
    "unidad_distancia": "km",
    "unidad_tiempo": "horas",
    "modo": "drive",
    "tipo_ruta": "balanced",
    "trafico": "free_flow",
    "costes_peajes_y_combustible": "pendientes de investigar",
}

with open(METADATA_FILE, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print("\nProceso completado.")
print(f"Ciudades: {len(ciudades)}")
print(f"Rutas dirigidas: {len(rutas_df)}")
print(f"Distancias: {DISTANCIAS_FILE}")
print(f"Tiempos: {TIEMPOS_FILE}")
print(f"Rutas: {RUTAS_FILE}")
print(f"Metadatos: {METADATA_FILE}")