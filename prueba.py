import numpy as np

from funciones import (
    rosenbrock,
    grad_rosenbrock,
    rastrigin,
    grad_rastrigin
)

# IMPORTANTE: usa la versión corregida de gradient_descent
# (la que devuelve "diverge" y pone f_final = inf al divergir)
from gradienteDecendente import gradient_descent
from PSO import pso


# ============================================================
# CONFIGURACIÓN
# ============================================================

N_CORRIDAS = 30
SEMILLA_BASE = 42
UMBRAL_EXITO = 1e-3

# Rosenbrock: alpha=0.001 es estable en el valle; max_paso evita saltos
# enormes lejos del mínimo (pon None para el algoritmo "puro").
ALPHA_ROSENBROCK = 0.001
MAX_PASO_ROSENBROCK = None

# Rastrigin: estabilidad requiere alpha < 2/L, con L ≈ 2 + 40π² ≈ 397
# => alpha < ~0.005. Con 0.002 el factor de contracción es ≈ 0.21.
ALPHA_RASTRIGIN = 0.002
MAX_PASO_RASTRIGIN = None


# ============================================================
# ESTADÍSTICAS
# ============================================================

def calcular_estadisticas(valores, umbral=UMBRAL_EXITO):
    """
    Las corridas divergentes (f_final = inf) se cuentan aparte y
    se excluyen de media, desviación, mejor y peor.
    """

    valores = np.array(valores, dtype=float)

    finitos = valores[np.isfinite(valores)]
    divergentes = int(np.sum(~np.isfinite(valores)))

    exitos = int(np.sum(valores <= umbral))  # inf nunca cuenta como éxito

    if len(finitos) > 0:
        media = np.mean(finitos)
        desviacion = np.std(finitos)
        mejor = np.min(finitos)
        peor = np.max(finitos)
    else:
        media = desviacion = mejor = peor = np.nan

    return {
        "media": media,
        "desviacion": desviacion,
        "mediana": np.median(finitos) if len(finitos) > 0 else np.nan,
        "mejor": mejor,
        "peor": peor,
        "exitos": exitos,
        "divergentes": divergentes,
        "tasa_exito": (exitos / len(valores)) * 100
    }


# ============================================================
# EXPERIMENTO: DESCENSO POR GRADIENTE
# ============================================================

def experimento_gradiente(
    funcion,
    gradiente,
    dimension,
    limites,
    alpha,
    max_paso=None
):

    valores = []
    evaluaciones_f = []
    evaluaciones_gradiente = []

    for corrida in range(N_CORRIDAS):

        rng = np.random.default_rng(SEMILLA_BASE + corrida)

        x0 = rng.uniform(limites[0], limites[1], dimension)

        resultado = gradient_descent(
            funcion=funcion,
            gradiente=gradiente,
            x0=x0,
            alpha=alpha,
            max_iter=10000,
            max_paso=max_paso
        )

        # f_final ya es inf si la corrida divergió
        valores.append(resultado["f_final"])

        evaluaciones_f.append(resultado["evaluaciones_f"])
        evaluaciones_gradiente.append(resultado["evaluaciones_gradiente"])

    return {
        "valores": valores,
        "evaluaciones_f": evaluaciones_f,
        "evaluaciones_gradiente": evaluaciones_gradiente
    }


# ============================================================
# EXPERIMENTO: PSO
# ============================================================

def experimento_pso(
    funcion,
    dimension,
    limites,
    n_particulas=30,
    max_iter=500,
    w=0.7,
    c1=1.5,
    c2=1.5
):

    valores = []
    evaluaciones_f = []

    for corrida in range(N_CORRIDAS):

        resultado = pso(
            funcion=funcion,
            dimension=dimension,
            n_particulas=n_particulas,
            max_iter=max_iter,
            limites=limites,
            w=w,
            c1=c1,
            c2=c2,
            semilla=SEMILLA_BASE + corrida
        )

        valores.append(resultado["f_final"])
        evaluaciones_f.append(resultado["evaluaciones_f"])

    return {
        "valores": valores,
        "evaluaciones_f": evaluaciones_f
    }


# ============================================================
# MOSTRAR RESULTADOS
# ============================================================

def mostrar_resultados(funcion, dimension, metodo, resultado):

    est = calcular_estadisticas(resultado["valores"])

    print()
    print("=" * 65)
    print(f"Función:    {funcion}")
    print(f"Dimensión:  {dimension}D")
    print(f"Método:     {metodo}")
    print("=" * 65)

    print(f"Media:               {est['media']:.3e}")
    print(f"Mediana:             {est['mediana']:.3e}")
    print(f"Desviación estándar: {est['desviacion']:.3e}")
    print(f"Mejor:               {est['mejor']:.3e}")
    print(f"Peor:                {est['peor']:.3e}")
    print(f"Éxitos:              {est['exitos']}/{N_CORRIDAS}")
    print(f"Tasa de éxito:       {est['tasa_exito']:.2f}%")
    print(f"Divergentes:         {est['divergentes']}/{N_CORRIDAS}")

    ev_f = np.mean(resultado["evaluaciones_f"])
    print(f"Evaluaciones f (promedio):         {ev_f:.0f}")

    if "evaluaciones_gradiente" in resultado:
        ev_g = np.mean(resultado["evaluaciones_gradiente"])
        print(f"Evaluaciones gradiente (promedio): {ev_g:.0f}")
        print(f"Total f + gradiente (promedio):    {ev_f + ev_g:.0f}")


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":

    print()
    print("#" * 65)
    print("# EXPERIMENTOS DE OPTIMIZACIÓN")
    print("#" * 65)
    print(f"# Corridas por configuración: {N_CORRIDAS}")
    print(f"# Semilla base: {SEMILLA_BASE}")
    print(f"# Umbral de éxito: {UMBRAL_EXITO:g}")
    print(f"# Alpha Rosenbrock: {ALPHA_ROSENBROCK} (max_paso={MAX_PASO_ROSENBROCK})")
    print(f"# Alpha Rastrigin:  {ALPHA_RASTRIGIN} (max_paso={MAX_PASO_RASTRIGIN})")
    print("#" * 65)

    # ========================================================
    # ROSENBROCK
    # ========================================================

    for dimension in [2, 3]:

        resultado = experimento_gradiente(
            funcion=rosenbrock,
            gradiente=grad_rosenbrock,
            dimension=dimension,
            limites=(-2, 2),
            alpha=ALPHA_ROSENBROCK,
            max_paso=MAX_PASO_ROSENBROCK
        )
        mostrar_resultados(
            "Rosenbrock", dimension, "Descenso por gradiente", resultado
        )

        resultado = experimento_pso(
            funcion=rosenbrock,
            dimension=dimension,
            limites=(-2, 2)
        )
        mostrar_resultados("Rosenbrock", dimension, "PSO", resultado)

    # ========================================================
    # RASTRIGIN
    # ========================================================

    for dimension in [2, 3]:

        resultado = experimento_gradiente(
            funcion=rastrigin,
            gradiente=grad_rastrigin,
            dimension=dimension,
            limites=(-5.12, 5.12),
            alpha=ALPHA_RASTRIGIN,
            max_paso=MAX_PASO_RASTRIGIN
        )
        mostrar_resultados(
            "Rastrigin", dimension, "Descenso por gradiente", resultado
        )

        resultado = experimento_pso(
            funcion=rastrigin,
            dimension=dimension,
            limites=(-5.12, 5.12)
        )
        mostrar_resultados("Rastrigin", dimension, "PSO", resultado)