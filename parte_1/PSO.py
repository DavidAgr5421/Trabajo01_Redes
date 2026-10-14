from funciones import rosenbrock, grad_rosenbrock


import numpy as np


def pso(
    funcion,
    dimension,
    n_particulas=30,
    max_iter=500,
    limites=(-5.12, 5.12),
    w=0.7,
    c1=1.5,
    c2=1.5,
    semilla=None
):
    """
    Particle Swarm Optimization (PSO) - Global Best.

    Parámetros
    ----------
    funcion : 
        Función objetivo.

    dimension : int
        Número de dimensiones del problema.

    n_particulas : int
        Número de partículas del enjambre.

    max_iter : int
        Número máximo de iteraciones.

    limites : tuple
        Límite inferior y superior de las posiciones.

    w : float
        Peso de inercia.

    c1 : float
        Coeficiente cognitivo.

    c2 : float
        Coeficiente social.

    semilla : int, optional
        Semilla para reproducibilidad.

    Retorna
    -------
    resultado : dict
        Información completa de la optimización.
    """

    rng = np.random.default_rng(semilla)

    limite_inferior, limite_superior = limites

    # ==========================================================
    # 1. INICIALIZACIÓN
    # ==========================================================

    # Posiciones iniciales aleatorias
    posiciones = rng.uniform(
        limite_inferior,
        limite_superior,
        size=(n_particulas, dimension)
    )

    # Velocidades iniciales
    rango = limite_superior - limite_inferior

    velocidades = rng.uniform(
        -rango,
        rango,
        size=(n_particulas, dimension)
    )

    # ==========================================================
    # 2. EVALUACIÓN INICIAL
    # ==========================================================

    valores = np.array([
        funcion(x)
        for x in posiciones
    ])

    evaluaciones_f = n_particulas

    # ==========================================================
    # 3. pbest
    # ==========================================================

    mejor_posicion_personal = posiciones.copy()
    mejor_valor_personal = valores.copy()

    # ==========================================================
    # 4. gbest
    # ==========================================================

    indice_mejor = np.argmin(mejor_valor_personal)

    mejor_posicion_global = (
        mejor_posicion_personal[indice_mejor].copy()
    )

    mejor_valor_global = (
        mejor_valor_personal[indice_mejor]
    )

    # ==========================================================
    # HISTORIA
    # ==========================================================

    historial_mejor_valor = [
        mejor_valor_global
    ]

    historial_mejor_posicion = [
        mejor_posicion_global.copy()
    ]

    historial_posiciones = [
        posiciones.copy()
    ]

    # ==========================================================
    # 5. ITERACIONES
    # ==========================================================

    for _ in range(max_iter):

        # Números aleatorios para cada partícula
        r1 = rng.random(
            size=(n_particulas, dimension)
        )

        r2 = rng.random(
            size=(n_particulas, dimension)
        )

        # ======================================================
        # Actualización de velocidad
        # ======================================================

        velocidades = (
            w * velocidades
            + c1 * r1
            * (mejor_posicion_personal - posiciones)
            + c2 * r2
            * (mejor_posicion_global - posiciones)
        )

        # ======================================================
        # Actualización de posición
        # ======================================================

        posiciones = posiciones + velocidades

        # Mantener partículas dentro de los límites
        posiciones = np.clip(
            posiciones,
            limite_inferior,
            limite_superior
        )

        # ======================================================
        # Evaluación
        # ======================================================

        valores = np.array([
            funcion(x)
            for x in posiciones
        ])

        evaluaciones_f += n_particulas

        # ======================================================
        # Actualizar pbest
        # ======================================================

        mejor_personal = valores < mejor_valor_personal

        mejor_posicion_personal[mejor_personal] = (
            posiciones[mejor_personal]
        )

        mejor_valor_personal[mejor_personal] = (
            valores[mejor_personal]
        )

        # ======================================================
        # Actualizar gbest
        # ======================================================

        indice_mejor = np.argmin(
            mejor_valor_personal
        )

        if (
            mejor_valor_personal[indice_mejor]
            < mejor_valor_global
        ):
            mejor_valor_global = (
                mejor_valor_personal[indice_mejor]
            )

            mejor_posicion_global = (
                mejor_posicion_personal[indice_mejor].copy()
            )

        # ======================================================
        # Guardar historia
        # ======================================================

        historial_mejor_valor.append(
            mejor_valor_global
        )

        historial_mejor_posicion.append(
            mejor_posicion_global.copy()
        )

        historial_posiciones.append(
            posiciones.copy()
        )

    # ==========================================================
    # RESULTADO
    # ==========================================================

    resultado = {
        "x_final": mejor_posicion_global,
        "f_final": mejor_valor_global,
        "iteraciones": max_iter,
        "evaluaciones_f": evaluaciones_f,
        "evaluaciones_gradiente": 0,
        "historial_x": np.array(
            historial_mejor_posicion
        ),
        "historial_f": np.array(
            historial_mejor_valor
        ),
        "historial_particulas": np.array(
            historial_posiciones
        ),
        "n_particulas": n_particulas,
        "w": w,
        "c1": c1,
        "c2": c2
    }

    return resultado