"""
Algoritmo Genetico (GA) para el TSP cerrado de las 47 capitales peninsulares.
Basado en la implementacion con DEAP del repositorio (cruce OX + mutacion shuffle).
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
from deap import algorithms, base, creator, tools

from costes_comunes import BASE_DIR, VALOR_HORA_POR_DEFECTO, cargar_costes

DIR_RESULTADOS = BASE_DIR / "resultados"


def ejecutar_ga(
    M: np.ndarray,
    *,
    tam_poblacion: int = 100,
    generaciones: int = 500,
    prob_cruce: float = 0.8,
    prob_mutacion: float = 0.2,
    semilla: int = 42,
    verbose: bool = False,
    registro: list | None = None,
    grabar_cada: int = 10,
    usar_or_opt: bool = False,
    max_evaluaciones: int | None = None,
) -> dict:
    """Ejecuta el Algoritmo Genetico con DEAP sobre la matriz de costos dirigida."""
    random.seed(semilla)
    np.random.seed(semilla)
    n = M.shape[0]
    t_inicio = time.perf_counter()

    # Limpiar definiciones previas de clases DEAP si existen
    if hasattr(creator, "FitnessMin"):
        del creator.FitnessMin
    if hasattr(creator, "Individual"):
        del creator.Individual

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, range(n), n)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    contador = {"evals": 0}

    def evaluar_ruta(individual):
        contador["evals"] += 1
        costo_viaje = 0.0
        for i in range(n - 1):
            costo_viaje += M[individual[i], individual[i + 1]]
        costo_viaje += M[individual[-1], individual[0]]
        if not np.isfinite(costo_viaje):
            return (1e9,)
        return (float(costo_viaje),)

    toolbox.register("evaluate", evaluar_ruta)
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    poblacion = toolbox.population(n=tam_poblacion)

    # Evaluar poblacion inicial
    invalid_ind = [ind for ind in poblacion if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    mejor_historico = tools.selBest(poblacion, k=1)[0]
    mejor_coste_global = mejor_historico.fitness.values[0]
    pasadas_or_opt = 0

    if usar_or_opt:
        from colonia_hormigas import coste_tour, or_opt
        tour_opt = or_opt(mejor_historico, M)
        pasadas_or_opt += 1
        coste_opt = coste_tour(M, tour_opt)
        if coste_opt < mejor_coste_global:
            mejor_coste_global = coste_opt
            mejor_historico[:] = tour_opt.tolist()
            mejor_historico.fitness.values = (coste_opt,)
            mejor_historico = toolbox.clone(mejor_historico)

    historia_mejor = [mejor_coste_global]
    historia_media = [np.mean([ind.fitness.values[0] for ind in poblacion])]
    historia_evals_tour = [contador["evals"]]
    historia_tiempos = [round(time.perf_counter() - t_inicio, 4)]

    if registro is not None:
        registro.append((0, mejor_coste_global, list(mejor_historico)))

    for gen in range(1, generaciones + 1):
        if max_evaluaciones is not None and contador["evals"] >= max_evaluaciones:
            break

        # Seleccion
        offspring = toolbox.select(poblacion, len(poblacion))
        offspring = list(map(toolbox.clone, offspring))

        # Cruce
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < prob_cruce:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Mutacion
        for mutant in offspring:
            if random.random() < prob_mutacion:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluar modificados respetando max_evaluaciones si esta definido
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_ind:
            if max_evaluaciones is not None and contador["evals"] >= max_evaluaciones:
                ind.fitness.values = (1e12,)
            else:
                ind.fitness.values = toolbox.evaluate(ind)

        poblacion[:] = offspring

        gen_best = tools.selBest(poblacion, k=1)[0]
        gen_best_cost = gen_best.fitness.values[0]
        if usar_or_opt and gen_best_cost < 1e11:
            from colonia_hormigas import coste_tour, or_opt
            tour_opt = or_opt(gen_best, M)
            pasadas_or_opt += 1
            coste_opt = coste_tour(M, tour_opt)
            if coste_opt < gen_best_cost:
                gen_best_cost = coste_opt
                gen_best[:] = tour_opt.tolist()
                gen_best.fitness.values = (coste_opt,)

        if gen_best_cost < mejor_coste_global:
            mejor_coste_global = gen_best_cost
            mejor_historico = toolbox.clone(gen_best)

        historia_mejor.append(mejor_coste_global)
        vals_validos = [ind.fitness.values[0] for ind in poblacion if ind.fitness.values[0] < 1e11]
        historia_media.append(float(np.mean(vals_validos)) if vals_validos else float(mejor_coste_global))
        historia_evals_tour.append(contador["evals"])
        historia_tiempos.append(round(time.perf_counter() - t_inicio, 4))

        if registro is not None and (gen % grabar_cada == 0 or gen == generaciones):
            registro.append((gen, mejor_coste_global, list(mejor_historico)))

        if verbose and (gen % 50 == 0 or gen == generaciones):
            print(f"  [GA] Gen {gen:4d}/{generaciones}: mejor={mejor_coste_global:.2f} EUR "
                  f"| media={historia_media[-1]:.2f} EUR")

    return {
        "tour": np.asarray(mejor_historico, dtype=np.int64),
        "coste": float(mejor_coste_global),
        "historia_mejor": historia_mejor,
        "historia_media": historia_media,
        "historia_evals_tour": historia_evals_tour,
        "historia_tiempos": historia_tiempos,
        "evaluaciones_tour": int(contador["evals"]),
        "pasadas_or_opt": int(pasadas_or_opt),
        "usar_or_opt": bool(usar_or_opt),
        "poblacion": tam_poblacion,
        "generaciones": generaciones,
        "semilla": semilla,
    }


def _main():
    parser = argparse.ArgumentParser(description="Algoritmo Genetico para el TSP de 47 capitales")
    parser.add_argument("--valor-hora", type=float, default=VALOR_HORA_POR_DEFECTO)
    parser.add_argument("--poblacion", type=int, default=100)
    parser.add_argument("--generaciones", type=int, default=500)
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    datos = cargar_costes()
    M = datos.matriz_coste(args.valor_hora)

    print(f"Iniciando GA (pob={args.poblacion}, gen={args.generaciones}, semilla={args.semilla})...")
    t0 = time.time()
    res = ejecutar_ga(
        M,
        tam_poblacion=args.poblacion,
        generaciones=args.generaciones,
        semilla=args.semilla,
        verbose=args.verbose or True,
    )
    dt = time.time() - t0
    desglose = datos.desglose_tour(res["tour"], args.valor_hora)
    print(f"\n[GA FINALIZADO en {dt:.2f}s]")
    print(f"Mejor coste: {desglose['coste_eur']:.2f} EUR | Distancia: {desglose['distancia_km']:.1f} km")


if __name__ == "__main__":
    _main()
