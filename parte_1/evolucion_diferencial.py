"""
================================================================================
OPTIMIZACIÓN NUMÉRICA: EVOLUCIÓN DIFERENCIAL (DE/rand/1/bin)
Curso: Redes Neuronales y Algoritmos Bioinspirados (Semestre 2026-02)
Universidad Nacional de Colombia - Facultad de Minas
================================================================================
Implementación de Evolución Diferencial canónica (Storn & Price, 1997).
Estrategia: DE/rand/1/bin con manejo de límites y selección codiciosa 1 a 1.
Optimiza funciones de prueba continuas (Rosenbrock y Rastrigin) en 2D y 3D.
Incluye:
  - Conteo estricto de evaluaciones de f(x).
  - Operadores: Mutación diferencial, Cruce binomial y Selección greedy.
  - Ejecución individual y modo de 30 corridas independientes con resumen estadístico.
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
# 2. ALGORITMO DE EVOLUCIÓN DIFERENCIAL (DE/rand/1/bin)
# ==============================================================================

def optimizar_evolucion_diferencial(
    f_obj,
    dimension=2,
    lb=-5.0,
    ub=5.0,
    n_pop=40,
    max_generaciones=100,
    F=0.7,
    CR=0.8,
    semilla=None
):
    """
    Ejecuta el algoritmo de Evolución Diferencial (DE/rand/1/bin).
    
    Parámetros:
      - f_obj: función objetivo a minimizar.
      - dimension: dimensión del problema (2 o 3).
      - lb, ub: límites inferior y superior.
      - n_pop: tamaño de la población (N >= 4).
      - max_generaciones: número máximo de iteraciones.
      - F: factor de escala diferencial [0.4, 0.9].
      - CR: probabilidad de cruce binomial [0.0, 1.0].
      - semilla: semilla aleatoria para reproducibilidad.
      
    Retorna:
      dict con: mejor solución x*, mejor f(x*), total evaluaciones f e historial.
    """
    if semilla is not None:
        np.random.seed(semilla)
        
    lim_inf = np.full(dimension, lb) if np.isscalar(lb) else np.asarray(lb)
    lim_sup = np.full(dimension, ub) if np.isscalar(ub) else np.asarray(ub)
    
    contador = {"evals": 0}
    
    # 1. Población inicial uniformemente distribuida
    pob = np.random.uniform(lim_inf, lim_sup, size=(n_pop, dimension))
    
    fitness = np.zeros(n_pop)
    for i in range(n_pop):
        fitness[i] = f_obj(pob[i])
        contador["evals"] += 1
        
    mejor_idx = np.argmin(fitness)
    mejor_x = pob[mejor_idx].copy()
    mejor_f = fitness[mejor_idx]
    
    historial_mejor_f = [mejor_f]
    
    # 2. Bucle generacional
    for gen in range(1, max_generaciones + 1):
        nueva_pob = pob.copy()
        
        for i in range(n_pop):
            # A. Mutación diferencial: selección de 3 vectores r1 != r2 != r3 != i
            candidatos = [idx for idx in range(n_pop) if idx != i]
            r1, r2, r3 = np.random.choice(candidatos, size=3, replace=False)
            
            v_mutante = pob[r1] + F * (pob[r2] - pob[r3])
            v_mutante = np.clip(v_mutante, lim_inf, lim_sup)
            
            # B. Recombinación / Cruce binomial
            j_rand = np.random.randint(0, dimension)
            u_prueba = np.zeros(dimension)
            
            for j in range(dimension):
                if np.random.rand() < CR or j == j_rand:
                    u_prueba[j] = v_mutante[j]
                else:
                    u_prueba[j] = pob[i, j]
                    
            # C. Selección codiciosa (Greedy 1 a 1)
            f_prueba = f_obj(u_prueba)
            contador["evals"] += 1
            
            if f_prueba <= fitness[i]:
                nueva_pob[i] = u_prueba
                fitness[i] = f_prueba
                
                if f_prueba < mejor_f:
                    mejor_f = f_prueba
                    mejor_x = u_prueba.copy()
                    
        pob = nueva_pob
        historial_mejor_f.append(mejor_f)
            
    return {
        "mejor_x": mejor_x,
        "mejor_f": mejor_f,
        "evaluaciones": contador["evals"],
        "historial_mejor_f": historial_mejor_f
    }


# ==============================================================================
# 3. PROTOCOLO DE EXPERIMENTACIÓN: 30 CORRIDAS INDEPENDIENTES
# ==============================================================================

def ejecutar_30_corridas(
    f_obj_nombre,
    dimension=2,
    n_corridas=30,
    n_pop=40,
    max_generaciones=100,
    F=0.7,
    CR=0.8,
    umbral_exito=None
):
    """
    Ejecuta 30 corridas independientes de Evolución Diferencial con semillas fijas
    y reporta estadísticas requeridas por la regla de comparación justa.
    """
    cfg = CONFIG_FUNCIONES[f_obj_nombre]
    f_obj = cfg["func"]
    lb, ub = cfg["lb"], cfg["ub"]
    umbral = umbral_exito if umbral_exito is not None else cfg["umbral_exito"]
    
    print("\n" + "=" * 75)
    print(f" EXPERIMENTO: EVOLUCIÓN DIFERENCIAL - 30 CORRIDAS INDEPENDIENTES")
    print(f" Función: {cfg['nombre']} | Dimensión: {dimension}D | Umbral de Éxito: {umbral}")
    print(f" Población: {n_pop} | Generaciones: {max_generaciones} | F: {F} | CR: {CR}")
    print("=" * 75)
    
    mejores_valores = []
    evaluaciones_totales = []
    exitos = 0
    
    for corrida in range(n_corridas):
        semilla = corrida * 100 + 42
        res = optimizar_evolucion_diferencial(
            f_obj=f_obj,
            dimension=dimension,
            lb=lb,
            ub=ub,
            n_pop=n_pop,
            max_generaciones=max_generaciones,
            F=F,
            CR=CR,
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
    print(f" * Evaluaciones de f por corrida:    {evaluaciones_totales[0]} (Formula: N * (G_max + 1) = {n_pop} * {max_generaciones+1})")
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
# 4. INTERFAZ CLI
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimización Numérica con Evolución Diferencial DE/rand/1/bin"
    )
    parser.add_argument(
        "--funcion",
        type=str,
        default="rastrigin",
        choices=["rosenbrock", "rastrigin"],
        help="Función de prueba continua a optimizar."
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
        default=40,
        help="Tamaño de la población (N >= 4)."
    )
    parser.add_argument(
        "--generaciones",
        type=int,
        default=100,
        help="Número máximo de generaciones."
    )
    parser.add_argument(
        "-F",
        type=float,
        default=0.7,
        help="Factor de mutación diferencial F (típicamente [0.4, 0.9])."
    )
    parser.add_argument(
        "-CR",
        type=float,
        default=0.8,
        help="Probabilidad de cruce binomial CR (entre 0.0 y 1.0)."
    )
    parser.add_argument(
        "--corridas",
        type=int,
        default=1,
        help="Número de corridas independientes (usar 30 para el reporte oficial)."
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
            F=args.F,
            CR=args.CR,
            umbral_exito=args.umbral
        )
    else:
        cfg = CONFIG_FUNCIONES[args.funcion]
        print("\n" + "=" * 60)
        print(f" OPTIMIZACIÓN: EVOLUCIÓN DIFERENCIAL (DE/rand/1/bin)")
        print(f" Función: {cfg['nombre']} | Dimensión: {args.dim}D")
        print(f" Población: {args.pop} | Generaciones: {args.generaciones} | F: {args.F} | CR: {args.CR}")
        print("=" * 60)
        res = optimizar_evolucion_diferencial(
            f_obj=cfg["func"],
            dimension=args.dim,
            lb=cfg["lb"],
            ub=cfg["ub"],
            n_pop=args.pop,
            max_generaciones=args.generaciones,
            F=args.F,
            CR=args.CR,
            semilla=args.semilla
        )
        print(f" * Mejor solución x*: {np.round(res['mejor_x'], 5)}")
        print(f" * Mejor valor f(x*): {res['mejor_f']:.6e}")
        print(f" * Evaluaciones f:    {res['evaluaciones']}")
        print(f" * Evaluaciones grad_f: 0")
        print("=" * 60 + "\n")
