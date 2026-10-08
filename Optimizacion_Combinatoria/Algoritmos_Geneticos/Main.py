from Funciones import *

# Carga de la configuración desde archivo externo
with open("config.json", "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

# Configuración de semilla fija para reproducibilidad
SEED = CONFIG["seed"]
random.seed(SEED)

# Parámetros del algoritmo genético
POP_SIZE = CONFIG["ga_parameters"]["population_size"]
GENERATIONS = CONFIG["ga_parameters"]["generations"]
CROSSOVER_RATE = CONFIG["ga_parameters"]["crossover_rate"]
MUTATION_RATE = CONFIG["ga_parameters"]["mutation_rate"]
TOURNAMENT_SIZE = CONFIG["ga_parameters"]["tournament_size"]

# Ejecución del algoritmo genético
def ejecutar_algoritmo_genetico(lista_ciudades, matriz_costos):
    n_ciudades = len(lista_ciudades)
    poblacion = [random.sample(lista_ciudades, n_ciudades) for _ in range(POP_SIZE)]
    costos = [calcular_costo_ruta(ind, matriz_costos, n_ciudades) for ind in poblacion]
    
    mejor_idx = costos.index(min(costos))
    mejor_ruta = poblacion[mejor_idx]
    mejor_costo = costos[mejor_idx]

    for _ in range(GENERATIONS):
        nueva_poblacion = [mejor_ruta]
        
        while len(nueva_poblacion) < POP_SIZE:
            padre1 = seleccion_torneo(poblacion, costos)
            padre2 = seleccion_torneo(poblacion, costos)
            
            hijo = cruce_ox(padre1, padre2, n_ciudades) if random.random() < CROSSOVER_RATE else padre1.copy()
            
            if random.random() < MUTATION_RATE:
                mutacion_inversion(hijo, n_ciudades)
                
            nueva_poblacion.append(hijo)
            
        poblacion = nueva_poblacion
        costos = [calcular_costo_ruta(ind, matriz_costos, n_ciudades) for ind in poblacion]
        
        min_costo_gen = min(costos)
        if min_costo_gen < mejor_costo:
            mejor_costo = min_costo_gen
            mejor_ruta = poblacion[costos.index(min_costo_gen)]
            
    return mejor_ruta, mejor_costo


if __name__ == "__main__":
    db_path = CONFIG["database"]["db_name"]
    conn = sqlite3.connect(db_path)

    ciudades = cargar_ciudades(conn)
    consumo, precio_fuel = cargar_parametros_vehiculo(conn)
    valor_hora = CONFIG["cost_parameters"]["hourly_rate"]

    matriz_costos = cargar_matriz_costos_db(conn, valor_hora, consumo, precio_fuel)
    conn.close()

    ruta_optima, costo_total = ejecutar_algoritmo_genetico(ciudades, matriz_costos)

    print("--- Recorrido Óptimo Encontrado ---")
    for i, ciudad in enumerate(ruta_optima, 1):
        print(f"{i}. {ciudad}")
    print(f"{len(ruta_optima) + 1}. {ruta_optima[0]} (Retorno al origen)")
    print(f"\nCosto total estimado: {costo_total:.2f} €")