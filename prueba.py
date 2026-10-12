import numpy as np

from funciones import (
    rosenbrock,
    grad_rosenbrock,
    rastrigin,
    grad_rastrigin
)

from gradienteDecendente import gradient_descent
from PSO import pso


# ============================================================
# CONFIGURACIÓN
# ============================================================

N_CORRIDAS = 30

SEMILLA_BASE = 42


# ============================================================
# ESTADÍSTICAS
# ============================================================

def calcular_estadisticas(valores, umbral=1e-3):

    valores = np.array(valores)

    exitos = np.sum(valores <= umbral)

    return {
        "media": np.mean(valores),
        "desviacion": np.std(valores),
        "mejor": np.min(valores),
        "peor": np.max(valores),
        "exitos": exitos,
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
    alpha
):

    resultados = []

    evaluaciones_f = []
    evaluaciones_gradiente = []

    for corrida in range(N_CORRIDAS):

        # Semilla diferente para cada corrida
        rng = np.random.default_rng(
            SEMILLA_BASE + corrida
        )

        # Punto inicial aleatorio
        x0 = rng.uniform(
            limites[0],
            limites[1],
            dimension
        )

        resultado = gradient_descent(
            funcion=funcion,
            gradiente=gradiente,
            x0=x0,
            alpha=alpha,
            max_iter=10000
        )

        resultados.append(
            resultado["f_final"]
        )

        evaluaciones_f.append(
            resultado["evaluaciones_f"]
        )

        evaluaciones_gradiente.append(
            resultado["evaluaciones_gradiente"]
        )

    return {
        "valores": resultados,
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

    resultados = []

    evaluaciones_f = []

    for corrida in range(N_CORRIDAS):

        # Semilla diferente para cada corrida
        semilla = SEMILLA_BASE + corrida

        resultado = pso(
            funcion=funcion,
            dimension=dimension,
            n_particulas=n_particulas,
            max_iter=max_iter,
            limites=limites,
            w=w,
            c1=c1,
            c2=c2,
            semilla=semilla
        )

        resultados.append(
            resultado["f_final"]
        )

        evaluaciones_f.append(
            resultado["evaluaciones_f"]
        )

    return {
        "valores": resultados,
        "evaluaciones_f": evaluaciones_f
    }


# ============================================================
# MOSTRAR RESULTADOS
# ============================================================

def mostrar_resultados(
    funcion,
    dimension,
    metodo,
    resultado
):

    estadisticas = calcular_estadisticas(
        resultado["valores"]
    )

    print()
    print("=" * 65)
    print(f"Función:    {funcion}")
    print(f"Dimensión:  {dimension}D")
    print(f"Método:     {metodo}")
    print("=" * 65)

    print(
        f"Media:               "
        f"{estadisticas['media']:.10f}"
    )

    print(
        f"Desviación estándar: "
        f"{estadisticas['desviacion']:.10f}"
    )

    print(
        f"Mejor:               "
        f"{estadisticas['mejor']:.10f}"
    )

    print(
        f"Peor:                "
        f"{estadisticas['peor']:.10f}"
    )

    print(
        f"Éxitos:              "
        f"{estadisticas['exitos']}/{N_CORRIDAS}"
    )

    print(
        f"Tasa de éxito:       "
        f"{estadisticas['tasa_exito']:.2f}%"
    )

    print(
        f"Evaluaciones f "
        f"(promedio):         "
        f"{np.mean(resultado['evaluaciones_f']):.0f}"
    )


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
    print("#" * 65)


    # ========================================================
    # ROSENBROCK
    # ========================================================

    for dimension in [2, 3]:

        # ----------------------------------------------------
        # GRADIENTE
        # ----------------------------------------------------

        resultado = experimento_gradiente(
            funcion=rosenbrock,
            gradiente=grad_rosenbrock,
            dimension=dimension,
            limites=(-2, 2),
            alpha=0.001
        )

        mostrar_resultados(
            "Rosenbrock",
            dimension,
            "Descenso por gradiente",
            resultado
        )


        # ----------------------------------------------------
        # PSO
        # ----------------------------------------------------

        resultado = experimento_pso(
            funcion=rosenbrock,
            dimension=dimension,
            limites=(-2, 2),
            n_particulas=30,
            max_iter=500,
            w=0.7,
            c1=1.5,
            c2=1.5
        )

        mostrar_resultados(
            "Rosenbrock",
            dimension,
            "PSO",
            resultado
        )


    # ========================================================
    # RASTRIGIN
    # ========================================================

    for dimension in [2, 3]:

        # ----------------------------------------------------
        # GRADIENTE
        # ----------------------------------------------------

        resultado = experimento_gradiente(
            funcion=rastrigin,
            gradiente=grad_rastrigin,
            dimension=dimension,
            limites=(-5.12, 5.12),
            alpha=0.01
        )

        mostrar_resultados(
            "Rastrigin",
            dimension,
            "Descenso por gradiente",
            resultado
        )


        # ----------------------------------------------------
        # PSO
        # ----------------------------------------------------

        resultado = experimento_pso(
            funcion=rastrigin,
            dimension=dimension,
            limites=(-5.12, 5.12),
            n_particulas=30,
            max_iter=500,
            w=0.7,
            c1=1.5,
            c2=1.5
        )

        mostrar_resultados(
            "Rastrigin",
            dimension,
            "PSO",
            resultado
        )