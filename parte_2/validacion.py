
from pathlib import Path
import pandas as pd
import numpy as np

DATA = Path(__file__).resolve().parent / "datos_rutas"

costes = pd.read_csv(DATA / "rutas_costes.csv")
ciudades = pd.read_csv(DATA / "ciudades.csv")

print("=== DIMENSIONES ===")
print("Ciudades:", len(ciudades))
print("Pares dirigidos:", len(costes))

assert len(ciudades) == 47, "Deben ser 47 capitales"
assert len(costes) == 47 * 46, "Numero incorrecto de pares"
assert costes.duplicated(["origin", "destination"]).sum() == 0

for col in ("distance_km", "time_h", "tax_eur", "gas_eur"):
    assert costes[col].notna().all(), f"Faltantes en {col}"
    assert (costes[col] >= 0).all(), f"Negativos en {col}"

print("\n=== EJEMPLO MADRID -> BARCELONA ===")
ej = costes[(costes.origin == "Madrid") & (costes.destination == "Barcelona")]
if len(ej):
    print(f"Distancia: {ej.iloc[0].distance_km:.2f} km")
    print(f"Tiempo: {ej.iloc[0].time_h:.2f} h | tax: {ej.iloc[0].tax_eur:.2f} | gas: {ej.iloc[0].gas_eur:.2f}")

print("\nValidacion basica finalizada.")