import json
import random
import sqlite3

# Consulta a la base de datos para cargar las ciudades registradas
def cargar_ciudades(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT nombre FROM ciudades ORDER BY nombre ASC;")
    filas = cursor.fetchall()
    return [f[0] for f in filas]


# Consulta a la base de datos para cargar los parámetros del vehículo
def cargar_parametros_vehiculo(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT consumo_litros_100km, precio_combustible_litro FROM parametros_vehiculo LIMIT 1;")
    fila = cursor.fetchone()
    if not fila:
        raise ValueError("La tabla 'parametros_vehiculo' no contiene datos.")
    return fila[0], fila[1]


# Carga y construcción de la matriz de costo total entre pares de ciudades directamente desde la base de datos
def cargar_matriz_costos_db(conn, valor_hora, consumo_100km, precio_combustible):
    cursor = conn.cursor()
    query = """
        SELECT ciudad_origen, ciudad_destino, distancia_km, tiempo_horas, costo_peaje 
        FROM matriz_tramos;
    """
    cursor.execute(query)
    filas = cursor.fetchall()

    matriz = {}
    for c_origen, c_destino, dist_km, tiempo_h, peaje in filas:
        if c_origen not in matriz:
            matriz[c_origen] = {}
        
        costo_tiempo = tiempo_h * valor_hora
        costo_combustible = (dist_km / 100.0) * consumo_100km * precio_combustible
        costo_total_tramo = costo_tiempo + costo_combustible + peaje
        
        matriz[c_origen][c_destino] = costo_total_tramo

    return matriz


# Evaluación del costo total del recorrido cerrado
def calcular_costo_ruta(ruta, matriz_costos, n_ciudades):
    costo_total = sum(matriz_costos[ruta[i]][ruta[i + 1]] for i in range(n_ciudades - 1))
    costo_total += matriz_costos[ruta[-1]][ruta[0]]
    return costo_total


# Cruzamiento de orden (OX)
def cruce_ox(padre1, padre2, n_ciudades):
    p1, p2 = random.sample(range(n_ciudades), 2)
    inicio, fin = min(p1, p2), max(p1, p2)
    
    hijo = [None] * n_ciudades
    hijo[inicio:fin + 1] = padre1[inicio:fin + 1]
    
    pos_hijo = (fin + 1) % n_ciudades
    for ciudad in padre2[fin + 1:] + padre2[:fin + 1]:
        if ciudad not in hijo:
            hijo[pos_hijo] = ciudad
            pos_hijo = (pos_hijo + 1) % n_ciudades
            
    return hijo


# Mutación por inversión
def mutacion_inversion(ruta, n_ciudades):
    p1, p2 = random.sample(range(n_ciudades), 2)
    inicio, fin = min(p1, p2), max(p1, p2)
    ruta[inicio:fin + 1] = reversed(ruta[inicio:fin + 1])


# Selección por torneo
def seleccion_torneo(poblacion, costos):
    aspirantes = random.sample(list(range(len(poblacion))), TOURNAMENT_SIZE)
    mejor = min(aspirantes, key=lambda idx: costos[idx])
    return poblacion[mejor]