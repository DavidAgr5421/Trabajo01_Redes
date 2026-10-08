import json
import random
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms

# --- 1. CARGAR CONFIGURACIÓN DESDE EL ARCHIVO EXTERNO ---
try:
    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
except FileNotFoundError:
    print("❌ Error: No se encontró el archivo 'config.json'. Asegúrate de que esté en la misma carpeta.")
    exit()

# Asignación automática de parámetros
VALOR_HORA_VENDEDOR = config["VALOR_HORA_VENDEDOR"]
ARCHIVO_CSV = config["ARCHIVO_CSV"]
TAMANO_POBLACION = config["POBLACION"]
NUM_GENERACIONES = config["GENERACIONES"]
CXPB = config["PROBABILIDAD_CRUCE"]
MUTPB = config["PROBABILIDAD_MUTACION"]

# --- 2. LEER CSV Y CALCULAR MATRIZ DE COSTOS ---
df = pd.read_csv(ARCHIVO_CSV)

# Tu regla de negocio basada en las 3 condiciones
df['costo_total_tramo'] = (df['time'] * VALOR_HORA_VENDEDOR) + df['tax'] + df['gas']

# Identificar ciudades únicas
ciudades = sorted(list(set(df['route_origen'].astype(str)).union(set(df['route_destino'].astype(str)))))
num_ciudades = len(ciudades)
ciudad_a_idx = {nombre: idx for idx, nombre in enumerate(ciudades)}

matriz_costos = np.full((num_ciudades, num_ciudades), np.inf)

for _, fila in df.iterrows():
    orig_idx = ciudad_a_idx[str(fila['route_origen'])]
    dest_idx = ciudad_a_idx[str(fila['route_destino'])]
    matriz_costos[orig_idx][dest_idx] = fila['costo_total_tramo']
    
np.fill_diagonal(matriz_costos, 0)


# --- 3. CONFIGURACIÓN DEL ALGORITMO GENÉTICO (DEAP) ---
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
    
    costo_viaje += matriz_costos[individual[-1]][individual]
    
    if np.isinf(costo_viaje):
        return (99999999,) 
        
    return (costo_viaje,)

toolbox.register("evaluate", evaluar_ruta)
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# --- 4. EJECUCIÓN ---
def ejecutar_ag():
    random.seed(42)
    poblacion = toolbox.population(n=TAMANO_POBLACION)
    
    # Se ejecuta usando los parámetros dinámicos del JSON
    poblacion, logbook = algorithms.eaSimple(
        poblacion, toolbox, 
        cxpb=CXPB, 
        mutpb=MUTPB, 
        ngen=NUM_GENERACIONES, 
        verbose=False
    )
    
    mejor_individuo = tools.selBest(poblacion, k=1)
    ruta_nombres = [ciudades[idx] for idx in mejor_individuo]
    
    print("🎯 ¡Optimización Completada usando parámetros externos!")
    print(f"Mejor orden de recorrido: {' -> '.join(ruta_nombres)} -> {ruta_nombres}")
    print(f"Costo total óptimo calculado: ${mejor_individuo.fitness.values:,.2f}")

if __name__ == "__main__":
    ejecutar_ag()