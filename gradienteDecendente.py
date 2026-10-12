import numpy as np


def gradient_descent(
    funcion,
    gradiente,
    x0,
    alpha=0.00085,
    max_iter=10000,
    tolerancia=1e-8,
    max_paso=None,       # longitud máxima del paso (recorte); None = sin recorte
    umbral_diverge=1e12  # si f supera esto, se declara divergencia
):
    x = np.asarray(x0, dtype=float).copy()

    historial_x = [x.copy()]
    historial_f = []

    evaluaciones_f = 0
    evaluaciones_gradiente = 0

    valor_actual = funcion(x)
    evaluaciones_f += 1
    historial_f.append(valor_actual)

    norma_gradiente = np.inf
    diverge = False

    for _ in range(max_iter):

        grad = gradiente(x)
        evaluaciones_gradiente += 1

        if not np.all(np.isfinite(grad)):
            diverge = True
            break

        norma_gradiente = np.linalg.norm(grad)

        if norma_gradiente < tolerancia:
            break

        paso = alpha * grad

        # Recorte del paso: evita saltos enormes lejos del mínimo
        if max_paso is not None:
            norma_paso = np.linalg.norm(paso)
            if norma_paso > max_paso:
                paso *= max_paso / norma_paso

        x_nuevo = x - paso

        if not np.all(np.isfinite(x_nuevo)):
            diverge = True
            break

        valor_nuevo = funcion(x_nuevo)
        evaluaciones_f += 1

        # Detecta divergencia ANTES de llegar al overflow
        if not np.isfinite(valor_nuevo) or valor_nuevo > umbral_diverge:
            diverge = True
            break

        x = x_nuevo
        valor_actual = valor_nuevo
        historial_x.append(x.copy())
        historial_f.append(valor_actual)

    return {
        "x_final": x,
        "f_final": np.inf if diverge else valor_actual,
        "diverge": diverge,
        "iteraciones": len(historial_f) - 1,
        "evaluaciones_f": evaluaciones_f,
        "evaluaciones_gradiente": evaluaciones_gradiente,
        "historial_x": np.array(historial_x),
        "historial_f": np.array(historial_f),
        "norma_gradiente": norma_gradiente,
    }