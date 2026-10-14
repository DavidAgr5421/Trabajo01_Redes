# Importar librerías.

import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from deap import base, creator, tools, algorithms

# 1. Cargar el archivo de configuración.
ruta_script = Path(__file__).resolve()
directorio_actual = ruta_script.parent
ruta_config = directorio_actual / 'Configuracion.json'

try:
    with open(ruta_config, 'r', encoding='utf-8') as f:
        Configuracion = json.load(f)
except FileNotFoundError:
    print(f"❌ Error: No se encontró el archivo 'Configuracion.json' en {ruta_config}")
    exit()

VALOR_HORA_VENDEDOR = Configuracion["VALOR_HORA_VENDEDOR"]
SEMILLA = Configuracion["SEMILLA"]
TAMANO_POBLACION = Configuracion["POBLACION"]
NUM_GENERACIONES = Configuracion["GENERACIONES"]
PROBABILIDAD_CRUCE = Configuracion["PROBABILIDAD_CRUCE"]
PROBABILIDAD_MUTACION = Configuracion["PROBABILIDAD_MUTACION"]

# 2. Leer la base de datos de las rutas.
carpeta_maestra = ruta_script.parent.parent
ruta_csv = carpeta_maestra / 'Base_de_Datos' / 'rutas_costes.csv'

try:
    df = pd.read_csv(ruta_csv)
except FileNotFoundError:
    print(f"❌ Error: No se encontró el CSV en {ruta_csv}")
    exit()

# 3. Calcular y crear la matriz de costos.
df['costo_total_tramo'] = (df['time_h'] * VALOR_HORA_VENDEDOR) + df['tax_eur'] + df['gas_eur']

ciudades = sorted(list(set(df['origin'].astype(str)).union(set(df['destination'].astype(str)))))
num_ciudades = len(ciudades)
ciudad_a_idx = {nombre: idx for idx, nombre in enumerate(ciudades)}

matriz_costos = np.full((num_ciudades, num_ciudades), np.inf)

for _, fila in df.iterrows():
    orig_idx = ciudad_a_idx[str(fila['origin'])]
    dest_idx = ciudad_a_idx[str(fila['destination'])]
    matriz_costos[orig_idx][dest_idx] = fila['costo_total_tramo']
    
np.fill_diagonal(matriz_costos, 0)

# 4. Configurar el algoritmo genético usando la librería DEAP.
if hasattr(creator, "FitnessMin"):
    del creator.FitnessMin
if hasattr(creator, "Individual"):
    del creator.Individual

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(num_ciudades), num_ciudades)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluar_ruta(individual):
    costo_viaje = 0
    for i in range(num_ciudades - 1):
        origen = individual[i]
        destino = individual[i+1]
        costo_viaje += matriz_costos[origen][destino]
    
    costo_viaje += matriz_costos[individual[-1]][individual[0]]
    
    if np.isinf(costo_viaje):
        return (99999999,) 
        
    return (costo_viaje,)

toolbox.register("evaluate", evaluar_ruta)
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Definir la función para ejecutar el algoritmo genético e imprimir el resultado.
def algoritmo_genetico():
    random.seed(SEMILLA)
    poblacion = toolbox.population(n=TAMANO_POBLACION)
    
    poblacion, logbook = algorithms.eaSimple(
        poblacion, 
        toolbox, 
        cxpb=PROBABILIDAD_CRUCE, 
        mutpb=PROBABILIDAD_MUTACION, 
        ngen=NUM_GENERACIONES, 
        verbose=False
    )
    
    mejor_individuo = tools.selBest(poblacion, k=1)[0]
    ruta_nombres = [ciudades[idx] for idx in mejor_individuo]
    costo_optimo = mejor_individuo.fitness.values[0]
    
    print(f"Mejor orden de recorrido: {' -> '.join(ruta_nombres)} -> {ruta_nombres[0]}")
    print(f"Costo total óptimo calculado: €{costo_optimo:,.2f}")

if __name__ == "__main__":
    algoritmo_genetico()