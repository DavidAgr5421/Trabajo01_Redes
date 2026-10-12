import numpy as np


def gradient_descent(
    funcion,
    gradiente,
    x0,
    alpha=0.00085,
    max_iter=10000,
    tolerancia=1e-8
):

    x = np.asarray(x0, dtype=float).copy()

    historial_x = [x.copy()]
    historial_f = []

    evaluaciones_f = 0
    evaluaciones_gradiente = 0

    # ==========================================================
    # EVALUACIÓN INICIAL
    # ==========================================================

    valor_actual = funcion(x)

    evaluaciones_f += 1

    historial_f.append(valor_actual)

    # ==========================================================
    # ITERACIONES
    # ==========================================================

    for iteracion in range(max_iter):

        grad = gradiente(x)

        evaluaciones_gradiente += 1

        # ------------------------------------------------------
        # Verificar que el gradiente sea válido
        # ------------------------------------------------------

        if not np.all(np.isfinite(grad)):
            break

        norma_gradiente = np.linalg.norm(grad)

        # ------------------------------------------------------
        # Criterio de convergencia
        # ------------------------------------------------------

        if norma_gradiente < tolerancia:
            break

        # ------------------------------------------------------
        # Actualización
        # ------------------------------------------------------

        x_nuevo = x - alpha * grad

        # ------------------------------------------------------
        # Verificar que la nueva posición sea válida
        # ------------------------------------------------------

        if not np.all(np.isfinite(x_nuevo)):
            break

        valor_nuevo = funcion(x_nuevo)

        evaluaciones_f += 1

        # ------------------------------------------------------
        # Verificar que la función sea válida
        # ------------------------------------------------------

        if not np.isfinite(valor_nuevo):
            break

        # ------------------------------------------------------
        # Guardar
        # ------------------------------------------------------

        x = x_nuevo

        valor_actual = valor_nuevo

        historial_x.append(x.copy())
        historial_f.append(valor_actual)

    # ==========================================================
    # RESULTADO
    # ==========================================================

    iteraciones = len(historial_f) - 1

    resultado = {
        "x_final": x,
        "f_final": valor_actual,
        "iteraciones": iteraciones,
        "evaluaciones_f": evaluaciones_f,
        "evaluaciones_gradiente": evaluaciones_gradiente,
        "historial_x": np.array(historial_x),
        "historial_f": np.array(historial_f),
        "norma_gradiente": norma_gradiente
    }

    return resultado