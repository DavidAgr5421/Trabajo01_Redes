"""
================================================================================
OPTIMIZACIÓN NUMÉRICA: ALGORITMO EVOLUTIVO CONTINUO
Curso: Redes Neuronales y Algoritmos Bioinspirados (Semestre 2026-02)
Universidad Nacional de Colombia - Facultad de Minas
================================================================================
Implementación de Algoritmo Evolutivo Continuo basado en notas de clase.
Optimiza funciones de prueba continuas (Rosenbrock y Rastrigin) en 2D y 3D.
Incluye:
  - Conteo estricto de evaluaciones de f(x).
  - Operadores: Elitismo, Cruzamiento en punto aleatorio y Mutación continua.
  - Ejecución individual y modo de 30 corridas con resumen estadístico.
  - Parámetros configurables por línea de comandos.
"""

import argparse
import sys
import numpy as np

# Asegurar codificación UTF-8 en consola de Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from funciones import rosenbrock, rastrigin


# ==============================================================================
# 1. CONFIGURACIÓN DE FUNCIONES DE PRUEBA Y LÍMITES
# ==============================================================================
CONFIG_FUNCIONES = {
    "rosenbrock": {
        "func": rosenbrock,
        "nombre": "Rosenbrock",
        "lb": -5.0,
        "ub": 5.0,
        "optimo_f": 0.0,
        "umbral_exito": 1e-1,
    },
    "rastrigin": {
        "func": rastrigin,
        "nombre": "Rastrigin",
        "lb": -5.12,
        "ub": 5.12,
        "optimo_f": 0.0,
        "umbral_exito": 1e-2,
    }
}


# ==============================================================================
# 2. OPERADORES GENÉTICOS
# ==============================================================================

def poblacion_inicial(n_pop, dimension, lb, ub, semilla=None):
    """
    Genera la población inicial distribuida uniformemente en [lb, ub].
    """
    if semilla is not None:
        np.random.seed(semilla)
    return np.random.uniform(lb, ub, size=(n_pop, dimension))


def evaluar_poblacion(poblacion, f_obj, contador):
    """
    Evalúa la función objetivo para cada individuo de la población
    e incrementa el contador de evaluaciones de f de forma explícita.
    """
    n = len(poblacion)
    valores = np.zeros(n)
    for i in range(n):
        valores[i] = f_obj(poblacion[i])
        contador["evals"] += 1
    return valores


def elitismo(poblacion_ordenada, n_elite):
    """
    Conserva los mejores n_elite individuos (menor valor de la función objetivo).
    """
    if n_elite <= 0:
        return np.empty((0, poblacion_ordenada.shape[1]))
    return poblacion_ordenada[:n_elite, :].copy()


def cruzamiento(poblacion_padres, n_hijos, dimension):
    """
    Cruzamiento por corte en punto aleatorio entre parejas de padres.
    """
    if n_hijos <= 0:
        return np.empty((0, dimension))
    
    n_padres = len(poblacion_padres)
    hijos = np.zeros((n_hijos, dimension))
    
    for i in range(0, n_hijos, 2):
        idx1, idx2 = np.random.choice(n_padres, size=2, replace=False if n_padres > 1 else True)
        padre1 = poblacion_padres[idx1]
        padre2 = poblacion_padres[idx2]
        
        punto = np.random.randint(1, dimension) if dimension > 1 else 0
            
        hijo1 = np.concatenate([padre1[:punto], padre2[punto:]])
        hijos[i, :] = hijo1
        
        if i + 1 < n_hijos:
            hijo2 = np.concatenate([padre2[:punto], padre1[punto:]])
            hijos[i + 1, :] = hijo2
            
    return hijos


def mutacion(poblacion_base, n_mutantes, lb, ub):
    """
    Mutación con perturbación gaussiana basada en la dispersión de la población,
    acotando los valores resultantes al dominio con np.clip.
    """
    if n_mutantes <= 0:
        return np.empty((0, poblacion_base.shape[1]))
        
    n_base, dimension = poblacion_base.shape
    idx_seleccionados = np.random.choice(n_base, size=n_mutantes, replace=True)
    mutantes = poblacion_base[idx_seleccionados].copy()
    
    std_coords = np.std(poblacion_base, axis=0)
    std_coords = np.where(std_coords < 1e-4, (ub - lb) * 0.05, std_coords)
    
    for i in range(n_mutantes):
        coord = np.random.randint(0, dimension)
        ruido = np.random.normal(0, std_coords[coord])
        mutantes[i, coord] = np.clip(mutantes[i, coord] + ruido, lb[coord], ub[coord])
        
    return mutantes


# ==============================================================================
# 3. ALGORITMO EVOLUTIVO PRINCIPAL
# ==============================================================================

def optimizar_algoritmo_evolutivo(
    f_obj,
    dimension=2,
    lb=-5.0,
    ub=5.0,
    n_pop=50,
    max_generaciones=100,
    frac_elitismo=0.15,
    frac_mutacion=0.20,
    frac_cruzamiento=0.65,
    semilla=None
):
    """
    Ejecuta una corrida del Algoritmo Evolutivo Continuo.
    """
    if semilla is not None:
        np.random.seed(semilla)
        
    lim_inf = np.full(dimension, lb) if np.isscalar(lb) else np.asarray(lb)
    lim_sup = np.full(dimension, ub) if np.isscalar(ub) else np.asarray(ub)
    
    contador = {"evals": 0}
    
    # 1. Población inicial
    pob = poblacion_inicial(n_pop, dimension, lim_inf, lim_sup)
    fitness = evaluar_poblacion(pob, f_obj, contador)
    
    orden = np.argsort(fitness)
    pob = pob[orden]
    fitness = fitness[orden]
    
    mejor_x = pob[0].copy()
    mejor_f = fitness[0]
    historial_mejor_f = [mejor_f]
    
    # Tamaños de subpoblaciones
    n_elite = int(np.round(n_pop * frac_elitismo))
    n_mutantes = int(np.round(n_pop * frac_mutacion))
    n_hijos = n_pop - n_elite - n_mutantes
    if n_hijos < 0:
        n_hijos = 0
        n_mutantes = n_pop - n_elite
        
    for gen in range(1, max_generaciones + 1):
        elite = elitismo(pob, n_elite)
        hijos = cruzamiento(pob, n_hijos, dimension)
        mutantes = mutacion(pob, n_mutantes, lim_inf, lim_sup)
        
        nueva_pob = np.vstack([elite, hijos, mutantes])[:n_pop]
        
        # Conteo transparente: los élite conservan su fitness, se evalúan hijos y mutantes
        fitness_nuevos = np.zeros(n_pop)
        fitness_nuevos[:n_elite] = fitness[:n_elite]
        
        for k in range(n_elite, n_pop):
            fitness_nuevos[k] = f_obj(nueva_pob[k])
            contador["evals"] += 1
            
        orden = np.argsort(fitness_nuevos)
        pob = nueva_pob[orden]
        fitness = fitness_nuevos[orden]
        
        if fitness[0] < mejor_f:
            mejor_f = fitness[0]
            mejor_x = pob[0].copy()
            
        historial_mejor_f.append(mejor_f)
            
    return {
        "mejor_x": mejor_x,
        "mejor_f": mejor_f,
        "evaluaciones": contador["evals"],
        "historial_mejor_f": historial_mejor_f
    }


# ==============================================================================
# 4. PROTOCOLO DE EXPERIMENTACIÓN: 30 CORRIDAS INDEPENDIENTES
# ==============================================================================

def ejecutar_30_corridas(
    f_obj_nombre,
    dimension=2,
    n_corridas=30,
    n_pop=50,
    max_generaciones=100,
    umbral_exito=None
):
    """
    Ejecuta corridas independientes con semillas fijas documentadas
    y reporta estadísticas (Media, Desv. Estándar, Mejor, Peor y Tasa de Éxito).
    """
    cfg = CONFIG_FUNCIONES[f_obj_nombre]
    f_obj = cfg["func"]
    lb, ub = cfg["lb"], cfg["ub"]
    umbral = umbral_exito if umbral_exito is not None else cfg["umbral_exito"]
    
    print("\n" + "=" * 75)
    print(f" EXPERIMENTO: ALGORITMO EVOLUTIVO - 30 CORRIDAS INDEPENDIENTES")
    print(f" Función: {cfg['nombre']} | Dimensión: {dimension}D | Umbral de Éxito: {umbral}")
    print(f" Población: {n_pop} | Generaciones: {max_generaciones}")
    print("=" * 75)
    
    mejores_valores = []
    evaluaciones_totales = []
    exitos = 0
    
    for corrida in range(n_corridas):
        semilla = corrida * 100 + 42
        res = optimizar_algoritmo_evolutivo(
            f_obj=f_obj,
            dimension=dimension,
            lb=lb,
            ub=ub,
            n_pop=n_pop,
            max_generaciones=max_generaciones,
            semilla=semilla
        )
        
        f_val = res["mejor_f"]
        evals = res["evaluaciones"]
        es_exito = abs(f_val - cfg["optimo_f"]) <= umbral
        
        mejores_valores.append(f_val)
        evaluaciones_totales.append(evals)
        if es_exito:
            exitos += 1
            
        print(f" Corrida {corrida+1:02d} | Semilla: {semilla:5d} | Mejor f: {f_val:12.6e} | Evals f: {evals} | Exito: {'[OK]' if es_exito else '[X]'}")
        
    mejores_valores = np.array(mejores_valores)
    evaluaciones_totales = np.array(evaluaciones_totales)
    
    print("\n" + "-" * 75)
    print(" RESUMEN ESTADISTICO (Regla de comparacion justa)")
    print("-" * 75)
    print(f" * Media del valor objetivo (f):     {np.mean(mejores_valores):.6e}")
    print(f" * Desviacion estandar (f):          {np.std(mejores_valores):.6e}")
    print(f" * Mejor valor encontrado (Min):     {np.min(mejores_valores):.6e}")
    print(f" * Peor valor encontrado (Max):      {np.max(mejores_valores):.6e}")
    print(f" * Mediana del valor objetivo:       {np.median(mejores_valores):.6e}")
    print(f" * Tasa de exito:                    {exitos}/{n_corridas} ({exitos / n_corridas * 100:.2f}%)")
    print(f" * Evaluaciones de f por corrida:    {evaluaciones_totales[0]} (Promedio: {np.mean(evaluaciones_totales):.1f})")
    print(f" * Evaluaciones de gradiente (grad_f): 0 (Metodo Heuristico libre de gradiente)")
    print("=" * 75 + "\n")
    
    return {
        "media": np.mean(mejores_valores),
        "std": np.std(mejores_valores),
        "mejor": np.min(mejores_valores),
        "peor": np.max(mejores_valores),
        "mediana": np.median(mejores_valores),
        "tasa_exito": exitos / n_corridas,
        "evaluaciones_f": evaluaciones_totales[0]
    }


# ==============================================================================
# 5. INTERFAZ CLI
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimización Numérica con Algoritmos Evolutivos"
    )
    parser.add_argument(
        "--funcion",
        type=str,
        default="rastrigin",
        choices=["rosenbrock", "rastrigin"],
        help="Función de prueba a optimizar."
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=2,
        choices=[2, 3],
        help="Dimensión del problema (2 o 3)."
    )
    parser.add_argument(
        "--pop",
        type=int,
        default=50,
        help="Tamaño de la población."
    )
    parser.add_argument(
        "--generaciones",
        type=int,
        default=100,
        help="Número máximo de generaciones."
    )
    parser.add_argument(
        "--elitismo",
        type=float,
        default=0.15,
        help="Fracción de individuos élite (0 a 1)."
    )
    parser.add_argument(
        "--mutacion",
        type=float,
        default=0.20,
        help="Fracción de mutación (0 a 1)."
    )
    parser.add_argument(
        "--cruce",
        type=float,
        default=0.65,
        help="Fracción de cruzamiento (0 a 1)."
    )
    parser.add_argument(
        "--corridas",
        type=int,
        default=1,
        help="Número de corridas independientes (usar 30 para el reporte)."
    )
    parser.add_argument(
        "--umbral",
        type=float,
        default=None,
        help="Umbral de error para considerar éxito."
    )
    parser.add_argument(
        "--semilla",
        type=int,
        default=42,
        help="Semilla aleatoria."
    )

    args = parser.parse_args()

    if args.corridas > 1:
        ejecutar_30_corridas(
            f_obj_nombre=args.funcion,
            dimension=args.dim,
            n_corridas=args.corridas,
            n_pop=args.pop,
            max_generaciones=args.generaciones,
            umbral_exito=args.umbral
        )
    else:
        cfg = CONFIG_FUNCIONES[args.funcion]
        print("\n" + "=" * 60)
        print(f" OPTIMIZACIÓN: ALGORITMO EVOLUTIVO")
        print(f" Función: {cfg['nombre']} | Dimensión: {args.dim}D")
        print(f" Población: {args.pop} | Generaciones: {args.generaciones}")
        print("=" * 60)
        res = optimizar_algoritmo_evolutivo(
            f_obj=cfg["func"],
            dimension=args.dim,
            lb=cfg["lb"],
            ub=cfg["ub"],
            n_pop=args.pop,
            max_generaciones=args.generaciones,
            frac_elitismo=args.elitismo,
            frac_mutacion=args.mutacion,
            frac_cruzamiento=args.cruce,
            semilla=args.semilla
        )
        print(f" * Mejor solución x*: {np.round(res['mejor_x'], 5)}")
        print(f" * Mejor valor f(x*): {res['mejor_f']:.6e}")
        print(f" * Evaluaciones f:    {res['evaluaciones']}")
        print(f" * Evaluaciones grad_f: 0")
        print("=" * 60 + "\n")
