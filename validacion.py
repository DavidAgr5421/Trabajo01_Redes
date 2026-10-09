
from pathlib import Path
import pandas as pd
import numpy as np

DATA = Path(__file__).resolve().parent / "datos_rutas"

dist = pd.read_csv(DATA / "matriz_distancias.csv", index_col=0)
tiem = pd.read_csv(DATA / "matriz_tiempos.csv", index_col=0)
rutas = pd.read_csv(DATA / "rutas.csv")

print("=== DIMENSIONES ===")
print("Distancias:", dist.shape)
print("Tiempos:", tiem.shape)
print("Rutas:", len(rutas))

assert dist.shape == (47, 47), "Dimensiones incorrectas en distancias"
assert tiem.shape == (47, 47), "Dimensiones incorrectas en tiempos"
assert len(rutas) == 47 * 46, "Número incorrecto de rutas"

assert dist.index.equals(tiem.index)
assert dist.columns.equals(tiem.columns)

d = dist.to_numpy(dtype=float)
t = tiem.to_numpy(dtype=float)

print("\n=== VALORES ===")
print("Distancias faltantes:", np.isnan(d).sum())
print("Tiempos faltantes:", np.isnan(t).sum())
print("Distancias <= 0:", np.sum(d <= 0))
print("Tiempos <= 0:", np.sum(t <= 0))

print("\n=== DIAGONAL ===")
print("Distancia diagonal máxima:", np.nanmax(np.diag(d)))
print("Tiempo diagonal máximo:", np.nanmax(np.diag(t)))

print("\n=== RUTAS ===")
print("Pares origen-destino duplicados:",
      rutas.duplicated(["origin", "destination"]).sum())
print("Peajes pendientes:", rutas["tax"].isna().sum())
print("Combustible pendiente:", rutas["gas"].isna().sum())

print("\n=== EJEMPLO MADRID → BARCELONA ===")
if "Madrid" in dist.index and "Barcelona" in dist.columns:
    print(f"Distancia: {dist.loc['Madrid', 'Barcelona']:.2f} km")
    print(f"Tiempo: {tiem.loc['Madrid', 'Barcelona']:.2f} horas")

# Las rutas por carretera no tienen por qué ser simétricas.
print("\n=== ASIMETRÍA ===")
print("Máxima diferencia distancia A→B vs B→A:",
      np.nanmax(np.abs(d - d.T)))
print("Máxima diferencia tiempo A→B vs B→A:",
      np.nanmax(np.abs(t - t.T)))

print("\nValidación básica finalizada.")