"""
Extrae los tramos reales de autopistas de peaje para los 2.162 pares dirigidos
entre las 47 capitales peninsulares usando Geoapify Routing API (Free Tier) con
fallback automatico a OSRM publico.

Caracteristicas clave:
  - Lee parte_2/datos_rutas/ciudades.csv y parte_2/datos_rutas/rutas.csv.
  - Agrupa los 2.162 pares dirigidos en 1.081 pares no dirigidos {A, B}:
      * Si abs(dist(A->B) - dist(B->A)) <= 8.0 km: 1 sola consulta y reutiliza.
      * Si difieren en > 8.0 km: consulta ambas direcciones por separado.
  - Concurrencia controlada (ThreadPoolExecutor max_workers=4) con limitador
    de tasa global (~4.2 req/s para Geoapify; 3.0 req/s para fallback OSRM).
  - Reanudable: guarda progreso incremental cada 50 consultas en
    parte_2/datos_rutas/tramos_peaje_rutas.json (escritura atomica).
  - NUNCA imprime GEOAPIFY_API_KEY.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parent.parent
P2_DIR = DATA_DIR.parent
REPO_DIR = P2_DIR.parent

CIUDADES_CSV = DATA_DIR / "ciudades.csv"
RUTAS_CSV = DATA_DIR / "rutas.csv"
SALIDA_JSON = DATA_DIR / "tramos_peaje_rutas.json"

GEOAPIFY_URL = "https://api.geoapify.com/v1/routing"
OSRM_URL_TMPL = (
    "http://router.project-osrm.org/route/v1/driving/"
    "{lon1},{lat1};{lon2},{lat2}"
)

SYMMETRY_THRESHOLD_KM = 8.0
GEOAPIFY_RPS = 4.2
OSRM_RPS = 3.0
MAX_WORKERS = 4
SAVE_EVERY = 50

# Orden de prioridad estricto: evita colisiones (ej. AP-1, AP-8 -> AP-8;
# AP-68 antes que AP-6; AP-51 antes que A-5; AP-15 antes que A-15).
ROAD_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("AP-68", re.compile(r"\bAP-68\b", re.IGNORECASE)),
    ("AP-66", re.compile(r"\bAP-66\b", re.IGNORECASE)),
    ("AP-61", re.compile(r"\bAP-61\b", re.IGNORECASE)),
    ("AP-51", re.compile(r"\bAP-51\b", re.IGNORECASE)),
    ("AP-71", re.compile(r"\bAP-71\b", re.IGNORECASE)),
    ("AP-53", re.compile(r"\bAP-53\b", re.IGNORECASE)),
    ("AP-36", re.compile(r"\bAP-36\b", re.IGNORECASE)),
    ("AP-41", re.compile(r"\bAP-41\b", re.IGNORECASE)),
    ("AP-46", re.compile(r"\bAP-46\b", re.IGNORECASE)),
    ("AP-15", re.compile(r"\bAP-15\b", re.IGNORECASE)),
    ("AG-55", re.compile(r"\bAG-55\b", re.IGNORECASE)),
    ("AG-57", re.compile(r"\bAG-57\b", re.IGNORECASE)),
    ("M-12", re.compile(r"\bM-12\b", re.IGNORECASE)),
    ("R-2", re.compile(r"\bR-2\b", re.IGNORECASE)),
    ("R-3", re.compile(r"\bR-3\b", re.IGNORECASE)),
    ("R-4", re.compile(r"\bR-4\b", re.IGNORECASE)),
    ("R-5", re.compile(r"\bR-5\b", re.IGNORECASE)),
    ("C-16", re.compile(r"\bC-16\b", re.IGNORECASE)),
    ("C-32", re.compile(r"\bC-32\b", re.IGNORECASE)),
    ("AP-8", re.compile(r"\bAP-8\b", re.IGNORECASE)),
    ("AP-9", re.compile(r"\bAP-9\b", re.IGNORECASE)),
    ("AP-7", re.compile(r"\bAP-7\b", re.IGNORECASE)),
    ("AP-6", re.compile(r"\bAP-6\b", re.IGNORECASE)),
    ("AP-1", re.compile(r"\bAP-1\b", re.IGNORECASE)),
    ("AP-8", re.compile(r"(Kantauriko autobidea|Bizkaiko Golkoko Autobidea|\bDO\b)", re.IGNORECASE)),
    ("TUNEL_VALLVIDRERA", re.compile(r"Vallvidrera", re.IGNORECASE)),
    ("TUNEL_CADI", re.compile(r"(Cad[ií]|Tunel del Cadi)", re.IGNORECASE)),
]


def cargar_env() -> None:
    """Carga variables de .env sin sobreescribir el entorno ni imprimir claves."""
    for ruta in (REPO_DIR / ".env", P2_DIR / ".env", DATA_DIR / ".env"):
        if not ruta.exists():
            continue
        with open(ruta, encoding="utf-8") as f:
            for linea in f:
                linea = linea.strip()
                if not linea or linea.startswith("#") or "=" not in linea:
                    continue
                clave, valor = linea.split("=", 1)
                os.environ.setdefault(
                    clave.strip(), valor.strip().strip('"').strip("'")
                )


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def identificar_via_base(name: str, ref: str = "") -> str | None:
    texto = f"{ref or ''} {name or ''}".strip()
    if not texto:
        return None
    for via, pat in ROAD_PATTERNS:
        if pat.search(texto):
            return via
    return None


def clasificar_concesion(
    via_base: str, lat: float, lon: float, texto_step: str = ""
) -> tuple[str | None, str | None]:
    """Devuelve (via_activa, via_liberada) segun coordenadas geograficas."""
    if via_base == "AP-68":
        if lat >= 42.65 and lon <= -2.82:
            return "AP-68_FORAL", None
        return None, "AP-68_LIBERADA_ESTATAL"

    if via_base == "AP-1":
        if lat >= 42.83 and lon > -2.78:
            return "AP-1_PV", None
        return None, "AP-1_LIBERADA_BURGOS"

    if via_base == "AP-8":
        if 42.95 <= lat <= 43.45 and -3.35 <= lon <= -1.70:
            return "AP-8", None
        return None, None

    if via_base == "AP-7":
        # 1. Ausur Alicante-Cartagena
        if 37.63 <= lat <= 38.28 and -1.01 <= lon <= -0.50:
            return "AP-7_AUSUR", None
        # 2. SEITT Cartagena-Vera
        if 37.18 <= lat <= 37.65 and -1.92 <= lon < -1.00:
            return "AP-7_SEITT", None
        # 3. Ausol Malaga-Guadiaro
        if 36.25 <= lat <= 36.68 and -5.36 <= lon <= -4.45:
            return "AP-7_AUSOL", None
        return None, "AP-7_LIBERADA"

    if via_base == "C-16":
        if lat >= 42.25:
            return "TUNEL_CADI", None
        if (41.40 <= lat <= 41.45 and 2.04 <= lon <= 2.13) or (
            "vallvidrera" in texto_step.lower()
        ):
            return "TUNEL_VALLVIDRERA", None
        if 41.44 < lat < 42.25:
            return "C-16", None
        return None, None

    if via_base == "C-32":
        if lat <= 41.42:
            return "C-32_SUR", None
        return None, "C-32_NORTE_LIBERADA"

    return via_base, None


class RateLimiter:
    """Limitador de tasa thread-safe basado en intervalo minimo."""

    def __init__(self, rps: float):
        self._lock = threading.Lock()
        self._min_interval = 1.0 / rps
        self._next_ts = 0.0

    def set_rps(self, rps: float) -> None:
        with self._lock:
            self._min_interval = 1.0 / rps

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if now < self._next_ts:
                sleep_s = self._next_ts - now
                self._next_ts += self._min_interval
            else:
                sleep_s = 0.0
                self._next_ts = now + self._min_interval
        if sleep_s > 0:
            time.sleep(sleep_s)


def _acumular_tramo(
    buckets: dict[str, dict], via: str, dist_m: float, lat: float, lon: float
) -> None:
    b = buckets.setdefault(
        via,
        {
            "m": 0.0,
            "lat_w": 0.0,
            "lon_w": 0.0,
            "min_lat": lat,
            "max_lat": lat,
            "min_lon": lon,
            "max_lon": lon,
        },
    )
    w = max(dist_m, 1.0)
    b["m"] += dist_m
    b["lat_w"] += lat * w
    b["lon_w"] += lon * w
    b["min_lat"] = min(b["min_lat"], lat)
    b["max_lat"] = max(b["max_lat"], lat)
    b["min_lon"] = min(b["min_lon"], lon)
    b["max_lon"] = max(b["max_lon"], lon)


def _compactar_buckets(buckets: dict[str, dict]) -> list[dict]:
    res = []
    for via, b in sorted(buckets.items()):
        km = round(b["m"] / 1000.0, 3)
        if km <= 0.0:
            continue
        w_tot = max(b["m"], 1.0)
        res.append(
            {
                "via": via,
                "km": km,
                "centroid": [
                    round(b["lat_w"] / w_tot, 4),
                    round(b["lon_w"] / w_tot, 4),
                ],
                "bbox": [
                    round(b["min_lat"], 4),
                    round(b["min_lon"], 4),
                    round(b["max_lat"], 4),
                    round(b["max_lon"], 4),
                ],
            }
        )
    return res


def parsear_geoapify(data: dict) -> tuple[float, float, list[dict], dict[str, float]]:
    feat = data["features"][0]
    props = feat["properties"]
    geom = feat["geometry"]
    if geom["type"] == "MultiLineString":
        coords = geom["coordinates"][0]
    elif geom["type"] == "LineString":
        coords = geom["coordinates"]
    else:
        coords = []

    activas: dict[str, dict] = {}
    liberadas: dict[str, float] = defaultdict(float)

    for leg in props.get("legs", []):
        for step in leg.get("steps", []):
            if not step.get("toll"):
                continue
            name = step.get("name") or ""
            ref = step.get("ref") or ""
            via_base = identificar_via_base(name, ref)
            if not via_base:
                continue

            dist_m = float(step.get("distance") or 0.0)
            if dist_m <= 0.0:
                continue

            fi = step.get("from_index")
            ti = step.get("to_index")
            if (
                coords
                and fi is not None
                and ti is not None
                and 0 <= fi <= ti < len(coords)
            ):
                if fi == ti:
                    lon, lat = coords[fi][0], coords[fi][1]
                    act, lib = clasificar_concesion(
                        via_base, lat, lon, f"{ref} {name}"
                    )
                    if act:
                        _acumular_tramo(activas, act, dist_m, lat, lon)
                    elif lib:
                        liberadas[lib] += dist_m / 1000.0
                else:
                    segs = []
                    h_sum = 0.0
                    for idx in range(fi, ti):
                        lon1, lat1 = coords[idx][0], coords[idx][1]
                        lon2, lat2 = coords[idx + 1][0], coords[idx + 1][1]
                        hm = haversine_m(lat1, lon1, lat2, lon2)
                        segs.append((hm, 0.5 * (lat1 + lat2), 0.5 * (lon1 + lon2)))
                        h_sum += hm
                    scale = (dist_m / h_sum) if h_sum > 0 else (dist_m / len(segs))
                    for hm, mlat, mlon in segs:
                        seg_m = hm * scale if h_sum > 0 else scale
                        act, lib = clasificar_concesion(
                            via_base, mlat, mlon, f"{ref} {name}"
                        )
                        if act:
                            _acumular_tramo(activas, act, seg_m, mlat, mlon)
                        elif lib:
                            liberadas[lib] += seg_m / 1000.0

    dist_km = round(float(props.get("distance", 0.0)) / 1000.0, 3)
    time_h = round(float(props.get("time", 0.0)) / 3600.0, 6)
    lib_out = {k: round(v, 3) for k, v in sorted(liberadas.items()) if v > 0}
    return dist_km, time_h, _compactar_tramo_list(activas), lib_out


def _compactar_tramo_list(activas: dict[str, dict]) -> list[dict]:
    return _compactar_buckets(activas)


def parsear_osrm(data: dict) -> tuple[float, float, list[dict], dict[str, float]]:
    route = data["routes"][0]
    activas: dict[str, dict] = {}
    liberadas: dict[str, float] = defaultdict(float)

    for leg in route.get("legs", []):
        for step in leg.get("steps", []):
            name = step.get("name") or ""
            ref = step.get("ref") or ""
            via_base = identificar_via_base(name, ref)
            if not via_base:
                continue
            dist_m = float(step.get("distance") or 0.0)
            if dist_m <= 0.0:
                continue
            intersections = step.get("intersections") or []
            if intersections:
                pts = [it["location"] for it in intersections if "location" in it]
            else:
                pts = []
            if not pts and "maneuver" in step and "location" in step["maneuver"]:
                pts = [step["maneuver"]["location"]]
            if not pts:
                continue
            share_m = dist_m / len(pts)
            for lon, lat in pts:
                act, lib = clasificar_concesion(
                    via_base, lat, lon, f"{ref} {name}"
                )
                if act:
                    _acumular_tramo(activas, act, share_m, lat, lon)
                elif lib:
                    liberadas[lib] += share_m / 1000.0

    dist_km = round(float(route.get("distance", 0.0)) / 1000.0, 3)
    time_h = round(float(route.get("duration", 0.0)) / 3600.0, 6)
    lib_out = {k: round(v, 3) for k, v in sorted(liberadas.items()) if v > 0}
    return dist_km, time_h, _compactar_buckets(activas), lib_out


def consultar_par(
    session: requests.Session,
    api_key: str | None,
    limiter: RateLimiter,
    osrm_flag: threading.Event,
    quota_lock: threading.Lock,
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> tuple[str, float, float, list[dict], dict[str, float]]:
    if not api_key:
        osrm_flag.set()

    if not osrm_flag.is_set():
        for intento in range(3):
            limiter.wait()
            try:
                resp = session.get(
                    GEOAPIFY_URL,
                    params={
                        "waypoints": f"{lat1},{lon1}|{lat2},{lon2}",
                        "mode": "drive",
                        "type": "balanced",
                        "traffic": "free_flow",
                        "details": "route_details,tolls",
                        "apiKey": api_key,
                    },
                    timeout=35,
                )
                if resp.status_code == 200:
                    dist_km, time_h, tramos, lib = parsear_geoapify(resp.json())
                    return "geoapify", dist_km, time_h, tramos, lib
                if resp.status_code in (402, 429):
                    if resp.status_code == 429 and intento < 2:
                        time.sleep(1.5 * (intento + 1))
                        continue
                    with quota_lock:
                        if not osrm_flag.is_set():
                            print(
                                f"\n[AVISO] Geoapify HTTP {resp.status_code}; "
                                "cambiando automaticamente a fallback OSRM publico."
                            )
                            osrm_flag.set()
                            limiter.set_rps(OSRM_RPS)
                    break
                if resp.status_code >= 500 and intento < 2:
                    time.sleep(1.0 * (intento + 1))
                    continue
                break
            except requests.RequestException:
                if intento < 2:
                    time.sleep(1.0 * (intento + 1))
                    continue
                break

    # Fallback OSRM publico
    url = OSRM_URL_TMPL.format(lon1=lon1, lat1=lat1, lon2=lon2, lat2=lat2)
    for intento in range(4):
        limiter.wait()
        try:
            resp = session.get(
                url,
                params={"overview": "false", "steps": "true"},
                timeout=35,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("code") == "Ok" and data.get("routes"):
                    dist_km, time_h, tramos, lib = parsear_osrm(data)
                    return "osrm_fallback", dist_km, time_h, tramos, lib
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(1.5 * (intento + 1))
                continue
        except requests.RequestException:
            time.sleep(1.5 * (intento + 1))
    raise RuntimeError(f"Fallo definitivo consultando ({lat1},{lon1})->({lat2},{lon2})")


def guardar_json(salida_path: Path, payload: dict) -> None:
    tmp = salida_path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, salida_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extraer tramos de peaje 2026")
    parser.add_argument("--limit", type=int, default=0, help="Limitar N consultas")
    parser.add_argument("--reset", action="store_true", help="Ignorar JSON previo")
    args = parser.parse_args()

    cargar_env()
    api_key = os.getenv("GEOAPIFY_API_KEY")

    ciudades: dict[str, tuple[float, float]] = {}
    with open(CIUDADES_CSV, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            ciudades[r["ciudad"].strip()] = (float(r["lat"]), float(r["lon"]))

    dist_csv: dict[tuple[str, str], float] = {}
    with open(RUTAS_CSV, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            o, d = r["origin"].strip(), r["destination"].strip()
            if o and d and o != d:
                dist_csv[(o, d)] = float(str(r["distance"]).replace(",", "."))

    # Cargar progreso previo si existe
    rutas_guardadas: dict[str, dict] = {}
    if SALIDA_JSON.exists() and not args.reset:
        try:
            with open(SALIDA_JSON, encoding="utf-8") as f:
                prev = json.load(f)
            rutas_guardadas = prev.get("rutas", {})
            print(f"Progreso previo cargado: {len(rutas_guardadas)} pares dirigidos.")
        except Exception:
            rutas_guardadas = {}

    # Agrupar en 1081 pares no dirigidos
    pares_no_dirigidos = sorted({tuple(sorted((o, d))) for (o, d) in dist_csv})
    tareas: list[tuple[str, str, bool]] = []  # (origin, dest, share_reverse)
    for a, b in pares_no_dirigidos:
        k_ab = f"{a} -> {b}"
        k_ba = f"{b} -> {a}"
        simetrico = abs(dist_csv[(a, b)] - dist_csv[(b, a)]) <= SYMMETRY_THRESHOLD_KM
        if simetrico:
            if k_ab not in rutas_guardadas or k_ba not in rutas_guardadas:
                tareas.append((a, b, True))
        else:
            if k_ab not in rutas_guardadas:
                tareas.append((a, b, False))
            if k_ba not in rutas_guardadas:
                tareas.append((b, a, False))

    if args.limit > 0:
        tareas = tareas[: args.limit]

    print(
        f"Pares no dirigidos: {len(pares_no_dirigidos)} | "
        f"Consultas pendientes: {len(tareas)}"
    )

    osrm_flag = threading.Event()
    quota_lock = threading.Lock()
    save_lock = threading.Lock()
    limiter = RateLimiter(GEOAPIFY_RPS if api_key else OSRM_RPS)
    session = requests.Session()
    session.headers.update({"User-Agent": "Entrega1-Redes-TSP/1.0"})

    def build_payload(n_q: int, n_geo: int, n_osrm: int) -> dict:
        return {
            "metadata": {
                "fecha_generacion_utc": datetime.now(timezone.utc).isoformat(),
                "fuente_primaria": "Geoapify Routing API (mode=drive, type=balanced, traffic=free_flow, details=route_details,tolls)",
                "fuente_fallback": "OSRM public routing API (overview=false, steps=true)",
                "criterio_simetria_km": SYMMETRY_THRESHOLD_KM,
                "consultas_sesion": n_q,
                "consultas_geoapify": n_geo,
                "consultas_osrm": n_osrm,
                "pares_totales_guardados": len(rutas_guardadas),
            },
            "rutas": rutas_guardadas,
        }

    completadas = 0
    n_geo = 0
    n_osrm = 0

    if tareas:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            fut_map = {}
            for o, d, share_rev in tareas:
                lat1, lon1 = ciudades[o]
                lat2, lon2 = ciudades[d]
                fut = pool.submit(
                    consultar_par,
                    session,
                    api_key,
                    limiter,
                    osrm_flag,
                    quota_lock,
                    lat1,
                    lon1,
                    lat2,
                    lon2,
                )
                fut_map[fut] = (o, d, share_rev)

            for fut in as_completed(fut_map):
                o, d, share_rev = fut_map[fut]
                src, dist_km, time_h, tramos, lib = fut.result()

                with save_lock:
                    if src == "geoapify":
                        n_geo += 1
                    else:
                        n_osrm += 1
                    completadas += 1

                    k_ab = f"{o} -> {d}"
                    rutas_guardadas[k_ab] = {
                        "origin": o,
                        "destination": d,
                        "source": src,
                        "symmetric_reused": False,
                        "route_distance_km": dist_km,
                        "route_time_h": time_h,
                        "tramos": tramos,
                    }
                    if lib:
                        rutas_guardadas[k_ab]["liberadas"] = lib

                    if share_rev:
                        k_ba = f"{d} -> {o}"
                        rutas_guardadas[k_ba] = {
                            "origin": d,
                            "destination": o,
                            "source": src,
                            "symmetric_reused": True,
                            "route_distance_km": dist_km,
                            "route_time_h": time_h,
                            "tramos": tramos,
                        }
                        if lib:
                            rutas_guardadas[k_ba]["liberadas"] = lib

                    if completadas % SAVE_EVERY == 0:
                        guardar_json(
                            SALIDA_JSON, build_payload(completadas, n_geo, n_osrm)
                        )
                        print(
                            f"[{completadas}/{len(tareas)}] "
                            f"geoapify={n_geo} osrm={n_osrm} "
                            f"pares_json={len(rutas_guardadas)}"
                        )

    guardar_json(SALIDA_JSON, build_payload(completadas, n_geo, n_osrm))
    print(
        f"Extraccion finalizada. Pares en JSON: {len(rutas_guardadas)} "
        f"(geoapify={n_geo}, osrm={n_osrm}). Archivo: {SALIDA_JSON}"
    )


if __name__ == "__main__":
    main()
