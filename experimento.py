import numpy as np

from funciones import *

from gradienteDecendente import *
from PSO import *


# ==========================================================
# ROSENBROCK - DESCENSO POR GRADIENTE
# ==========================================================

x0 = np.random.uniform(-2, 2, size=2)

resultado_gd = gradient_descent(
    funcion=rosenbrock,
    gradiente=grad_rosenbrock,
    x0=x0,
    alpha=0.001,
    max_iter=10000
)

print("===================================")
print("ROSENBROCK - GRADIENTE")
print("===================================")

print("Punto inicial:", x0)
print("Punto final:", resultado_gd["x_final"])
print("Valor final:", resultado_gd["f_final"])
print("Iteraciones:", resultado_gd["iteraciones"])
print("Evaluaciones de f:", resultado_gd["evaluaciones_f"])
print(
    "Evaluaciones de gradiente:",
    resultado_gd["evaluaciones_gradiente"]
)


# ==========================================================
# ROSENBROCK - PSO
# ==========================================================

resultado_pso = pso(
    funcion=rosenbrock,
    dimension=2,
    n_particulas=30,
    max_iter=500,
    limites=(-2, 2),
    w=0.7,
    c1=1.5,
    c2=1.5,
    semilla=42
)

print("\n===================================")
print("ROSENBROCK - PSO")
print("===================================")

print("Punto final:", resultado_pso["x_final"])
print("Valor final:", resultado_pso["f_final"])
print("Iteraciones:", resultado_pso["iteraciones"])
print("Evaluaciones de f:", resultado_pso["evaluaciones_f"])


# ==========================================================
# rastrigin - DESCENSO POR GRADIENTE
# ==========================================================

x0 = np.random.uniform(-2, 2, size=2)

resultado_gd = gradient_descent(
    funcion=rastrigin,
    gradiente=grad_rastrigin,
    x0=x0,
    alpha=0.001,
    max_iter=10000
)

print("===================================")
print("rastrigin - GRADIENTE")
print("===================================")

print("Punto inicial:", x0)
print("Punto final:", resultado_gd["x_final"])
print("Valor final:", resultado_gd["f_final"])
print("Iteraciones:", resultado_gd["iteraciones"])
print("Evaluaciones de f:", resultado_gd["evaluaciones_f"])
print(
    "Evaluaciones de gradiente:",
    resultado_gd["evaluaciones_gradiente"]
)


# ==========================================================
# rastrigin - PSO
# ==========================================================

resultado_pso = pso(
    funcion=rastrigin,
    dimension=2,
    n_particulas=30,
    max_iter=500,
    limites=(-2, 2),
    w=0.7,
    c1=1.5,
    c2=1.5,
    semilla=42
)

print("\n===================================")
print("rastrigin - PSO")
print("===================================")

print("Punto final:", resultado_pso["x_final"])
print("Valor final:", resultado_pso["f_final"])
print("Iteraciones:", resultado_pso["iteraciones"])
print("Evaluaciones de f:", resultado_pso["evaluaciones_f"])