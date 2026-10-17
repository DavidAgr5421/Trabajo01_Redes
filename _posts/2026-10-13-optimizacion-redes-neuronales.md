---
layout: post
title: "De Paisajes Continuos a Carreteras Peninsulares: Optimización Numérica y Combinatoria mediante Metaheurísticas Bioinspiradas y Descenso por Gradiente"
date: 2026-10-13 12:00:00 -0500
author: "Alejandro Agudelo Anaya, David Agudelo Arboleda, Santiago Álvarez Martínez, Daniel Felipe López Zapata, Sara Osorio Piñerez"
categories: [investigacion, bioinspirados, optimizacion]
tags: [redes-neuronales, tsp, aco, pso, algoritmos-geneticos, evolucion-diferencial]
math: true
---

# De Paisajes Continuos a Carreteras Peninsulares: Optimización Numérica y Combinatoria mediante Metaheurísticas Bioinspiradas y Descenso por Gradiente

**Entrada de Blog y Reporte Técnico de Investigación — Documentación Oficial Trabajo 1**  
**Curso**: Redes Neuronales y Algoritmos Bioinspirados (Semestre 2026-02)  
**Universidad Nacional de Colombia — Facultad de Minas, Sede Medellín**  
**Autores**:
- Alejandro Agudelo Anaya — CC: 1017271139
- David Agudelo Arboleda — CC: 1000899797
- Santiago Álvarez Martínez — CC: 1022142463
- Daniel Felipe López Zapata — CC: 1045046468
- Sara Osorio Piñerez — CC: 1001370113

**Profesor Tutor**: Juan David Ospina Arango (`jdospina@unal.edu.co`)  
**Repositorio GitHub**: [https://github.com/DavidAgr5421/Trabajo01_Redes](https://github.com/DavidAgr5421/Trabajo01_Redes)  
**Fecha Oficial**: 13 de octubre de 2026  

---

## Resumen Ejecutivo

Este trabajo aborda dos de las ramas fundamentales de la optimización moderna: la **optimización numérica continua** en paisajes topológicos no convexos y la **optimización combinatoria discreta** mediante el Problema del Vendedor Viajero (TSP) aplicado a las 47 capitales de provincia de la España peninsular. 

En la primera parte, se contrasta el rendimiento del Descenso por Gradiente (GD) frente a tres metaheurísticas poblacionales —Optimización por Enjambre de Partículas (PSO), Algoritmo Evolutivo Continuo (EA) y Evolución Diferencial (DE)— evaluando las funciones arquetípicas de Rosenbrock y Rastrigin en 2 y 3 dimensiones bajo una regla de comparación computacionalmente justa (basada en el costo analítico de $2d$ evaluaciones de función por aproximación de gradiente: $E_{\text{equiv}} = E_f + 2d \times E_{\nabla}$).

En la segunda parte, se modela y resuelve el TSP cerrado peninsular incorporando una función de costo tripartita real: tiempo laboral de conducción (salario mínimo fijado por el **Real Decreto 126/2026 a 5,09 €/h**), costo de peajes reales (0,00 € mediante autovías gratuitas de alta capacidad) e impacto del consumo de combustible de un **Toyota Corolla híbrido (4,5 L/100 km)**. Mediante una Colonia de Hormigas (ACO) enriquecida con búsqueda local Or-opt (Or, 1976), se alcanza un recorrido óptimo de **845,09 €** (6.307,79 km y 66,49 h), superando en un 3,46% al ACO puro y demostrando una reducción cuádruple en el tiempo de convergencia. 

Finalmente, se documenta la sensibilidad del recorrido ante variaciones en el valor horario del conductor, se expone el protocolo de asistencia por Inteligencia Artificial y se detalla una rigurosa "Cacería de la Alucinación" técnica y regulatoria.

---

## 1. Parte 1: Optimización Numérica Continua

### 1.1 Funciones de Prueba y Gradientes Analíticos

#### 1. Función de Rosenbrock ("Valle en Banana")
$$f(\mathbf{x}) = \sum_{i=1}^{n-1} \left[ 100\left(x_{i+1} - x_i^2\right)^2 + (1 - x_i)^2 \right]$$
* **Mínimo Global**: $\mathbf{x}^* = (1, 1, \dots, 1)$, con $f(\mathbf{x}^*) = 0$.
* **Gradiente Analítico Vectorizado**:
  $$\frac{\partial f}{\partial x_1} = -400 x_1 (x_2 - x_1^2) - 2(1 - x_1)$$
  $$\frac{\partial f}{\partial x_i} = 200(x_i - x_{i-1}^2) - 400 x_i (x_{i+1} - x_i^2) - 2(1 - x_i), \quad 2 \le i \le n-1$$
  $$\frac{\partial f}{\partial x_n} = 200(x_n - x_{n-1}^2)$$

#### 2. Función de Rastrigin (Multimodalidad Extrema)
$$f(\mathbf{x}) = 10n + \sum_{i=1}^n \left[ x_i^2 - 10\cos(2\pi x_i) \right]$$
* **Mínimo Global**: $\mathbf{x}^* = (0, 0, \dots, 0)$, con $f(\mathbf{x}^*) = 0$.
* **Gradiente Analítico**: $\frac{\partial f}{\partial x_i} = 2x_i + 20\pi \sin(2\pi x_i)$.

![Visualización 3D de Rosenbrock y Rastrigin]({{ site.baseurl }}/assets/img/figura1_superficies_3d.png)
*Figura 1 — Visualización 3D de las superficies de Rosenbrock y Rastrigin.*

---

### 1.2 Regla de Comparación Justa y Equivalencia Computacional

Evaluar el vector gradiente en dimensión $d$ equivale, mediante diferencias finitas centrales, a $2d$ evaluaciones directas de la función objetivo. La métrica unificada de evaluaciones equivalentes se define como:
$$E_{\text{equiv}} = E_f + 2d \times E_{\nabla}$$

---

### 1.3 Animaciones de Convergencia en Rosenbrock 2D

| Descenso por Gradiente | Optimización por Enjambre de Partículas (PSO) |
| :---: | :---: |
| ![GIF GD]({{ site.baseurl }}/assets/img/animacion_gd_rosenbrock_2d.gif) | ![GIF PSO]({{ site.baseurl }}/assets/img/animacion_pso_rosenbrock_2d.gif) |
| *Figura 2 — Trayectoria de GD en Rosenbrock 2D.* | *Figura 3 — Movimiento de partículas PSO en Rosenbrock 2D.* |

---

### 1.4 Resultados Estadísticos Oficiales (30 Corridas Independientes)

#### Tabla Oficial: Desempeño Comparativo de los Cuatro Algoritmos (Tabla 12)
| Función | Dim | Algoritmo | Media $f$ | Mediana $f$ | Desv. Estándar | Mejor | Peor | Tasa Éxito | Evals $f$ | Evals Grad |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rastrigin** | **2D** | Descenso por gradiente | 15,82 | 16,42 | 10,03 | 0,995 | 31,84 | 0,0% | 33 | 33 |
| **Rastrigin** | **2D** | PSO | 0,000 | 0,000 | 0,000 | 0,000 | 0,000 | 100,0% | 15.030 | 0 |
| **Rastrigin** | **2D** | Algoritmo evolutivo | 0,298 | 0,000 | 0,458 | 0,000 | 0,995 | 70,0% | 4.250 | 0 |
| **Rastrigin** | **2D** | Evolución diferencial | 0,000 | 0,000 | 0,000 | 0,000 | 0,000 | 100,0% | 4.040 | 0 |
| **Rastrigin** | **3D** | Descenso por gradiente | 24,48 | 22,88 | 14,16 | 1,990 | 53,73 | 0,0% | 34 | 34 |
| **Rastrigin** | **3D** | PSO | 0,066 | 0,000 | 0,248 | 0,000 | 0,995 | 93,33% | 15.030 | 0 |
| **Rastrigin** | **3D** | Algoritmo evolutivo | 1,426 | 0,995 | 0,891 | 0,000 | 3,980 | 16,67% | 4.250 | 0 |
| **Rastrigin** | **3D** | Evolución diferencial | 3,31e-11 | 0,000 | 1,79e-10 | 0,000 | 0,995 | 96,67% | 4.040 | 0 |
| **Rosenbrock** | **2D** | Descenso por gradiente | 2,79e-5 | 3,16e-5 | 1,61e-5 | 2,64e-10 | 6,56e-5 | 100,0% | 2.227 | 2.226 |
| **Rosenbrock** | **2D** | PSO | 3,37e-18 | 4,01e-21 | 1,19e-17 | 5,84e-26 | 5,83e-17 | 100,0% | 15.030 | 0 |
| **Rosenbrock** | **2D** | Algoritmo evolutivo | 1,84e-2 | 3,15e-3 | 3,89e-2 | 4,22e-5 | 0,185 | 40,0% | 4.250 | 0 |
| **Rosenbrock** | **2D** | Evolución diferencial | 1,20e-8 | 3,12e-10 | 2,14e-8 | 4,81e-14 | 8,95e-8 | 100,0% | 4.040 | 0 |
| **Rosenbrock** | **3D** | Descenso por gradiente | 1,16e-5 | 9,22e-6 | 7,31e-6 | 1,92e-7 | 3,89e-5 | 90,0% | 339 | 338 |
| **Rosenbrock** | **3D** | PSO | 7,39e-4 | 3,84e-4 | 1,06e-3 | 6,93e-8 | 4,78e-3 | 76,67% | 15.030 | 0 |
| **Rosenbrock** | **3D** | Algoritmo evolutivo | 0,312 | 0,184 | 0,342 | 1,45e-3 | 1,458 | 3,33% | 4.250 | 0 |
| **Rosenbrock** | **3D** | Evolución diferencial | 4,18e-2 | 1,20e-2 | 6,85e-2 | 8,12e-5 | 0,242 | 26,67% | 4.040 | 0 |

![Tasa de Éxito]({{ site.baseurl }}/assets/img/figura4_tasa_exito.png)
*Figura 4 — Comparación de la Tasa de Éxito en 30 corridas independientes.*

![Mediana y Costo]({{ site.baseurl }}/assets/img/figura5_mediana_valor_final.png)
*Figura 5 — Mediana del valor final alcanzado (escala logarítmica).*

![Costo Computacional]({{ site.baseurl }}/assets/img/figura6_costo_computacional.png)
*Figura 6 — Comparación del número promedio de evaluaciones directas de la función.*

---

## 2. Parte 2: Optimización Combinatoria (TSP 47 Capitales Peninsulares)

### 2.1 Modelo Económico Real y Justificación Jurídica

$$\text{Costo}_{ij} = (\text{ValorHora} \times T_{ij}) + \text{Peajes}_{ij} + \text{Combustible}_{ij}$$

* **Valor de la Hora de Trabajo**: Establecido conforme al **Real Decreto 126/2026, de 18 de febrero** (BOE núm. 44). El SMI diario es de 40,70 €/día. Para una jornada laboral continua de 8 horas:
  $$\text{ValorHora} = \frac{40{,}70\text{ \euro}}{8\text{ h}} = 5{,}0875 \approx 5{,}09\text{ \euro/h}$$
* **Vehículo Seleccionado**: **Toyota Corolla 140H e-CVT 1.8** híbrido (sexto turismo más vendido en España). Consumo medio homologado WLTP: **4,5 L/100 km** ($0,045\text{ L/km}$). Gasolina 95 E5: 1,785 €/L (Geoportal MITECO). Costo por kilómetro: $0,080325\text{ \euro/km}$. Autonomía de ~956 km por depósito de 43 L.
* **Peajes**: 0,00 € en el recorrido óptimo gracias a la red estatal libre de autovías paralelas.

---

### 2.2 Desempeño Cuantitativo: ACO con Or-opt vs ACO Puro (Tabla 14)

| Métrica | ACO con Or-opt | ACO sin Or-opt | ACO sin Or-opt (Alto Presupuesto) | Ahorro / Impacto con Or-opt |
| :--- | :---: | :---: | :---: | :---: |
| **Costo Total (€)** | **845,09 €** | 875,41 € | 867,51 € | **-30,32 € (-3,46%)** |
| **Distancia (km)** | **6.307,79 km** | 6.517,53 km | 6.499,57 km | **-209,74 km menos** |
| **Tiempo de Viaje** | **66,49 h** (66h 29m) | 69,13 h (69h 08m) | 67,86 h (67h 51m) | **-2h 39m ahorradas** |
| **Costo Conductor (€)**| **338,42 €** | 351,89 € | 345,40 € | -13,47 € en nómina |
| **Combustible (€)** | **506,67 €** | 523,52 € | 522,11 € | -16,85 € en gasolina 95 |
| **Peajes (€)** | **0,00 €** | 0,00 € | 0,00 € | Red gratuita de alta capacidad |
| **Iteración Convergencia**| **Iteración 20** | Iteración 85 | Iteración 31 | **4x más rápida** |
| **Tiempo Cómputo (s)**| **4,77 s** | 3,97 s | 19,81 s | Balance óptimo |

![Convergencia ACO]({{ site.baseurl }}/assets/img/figura7_convergencia_aco.png)
*Figura 7 — Curva de convergencia comparativa: ACO con Or-opt (845,09 €) vs ACO sin Or-opt.*

#### Secuencia del Tour Óptimo Peninsular (845,09 €):
```text
Logroño -> Soria -> Zaragoza -> Huesca -> Lleida -> Girona -> Barcelona -> 
Tarragona -> Castellón de la Plana -> Valencia -> Teruel -> Cuenca -> 
Guadalajara -> Madrid -> Toledo -> Ciudad Real -> Albacete -> Alicante -> 
Murcia -> Almería -> Granada -> Jaén -> Córdoba -> Málaga -> Cádiz -> 
Sevilla -> Huelva -> Badajoz -> Cáceres -> Salamanca -> Ávila -> Segovia -> 
Valladolid -> Palencia -> Burgos -> León -> Zamora -> Ourense -> Pontevedra -> 
Lugo -> A Coruña -> Oviedo -> Santander -> Bilbao -> Vitoria-Gasteiz -> 
San Sebastián -> Pamplona -> (retorno a Logroño)
```

---

### 2.3 Animaciones GIF del TSP en España

| Colonia de Hormigas con Or-opt (845,09 €) | Algoritmo Genético DEAP (1.024,28 €) |
| :---: | :---: |
| ![GIF ACO]({{ site.baseurl }}/assets/img/evolucion_aco_con_oropt.gif) | ![GIF GA]({{ site.baseurl }}/assets/img/evolucion_ga.gif) |
| *Figura 8 — Animación de ACO con búsqueda local Or-opt.* | *Figura 9 — Animación del Algoritmo Genético en DEAP.* |

---

### 2.4 Estudio de Sensibilidad del Salario Horario

| Valor Hora (€/h) | Costo Total (€) | Distancia (km) | Tiempo (h) | Combustible (€) | ¿Idéntica a Base? | Régimen Operativo Predominante |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0,00 €/h** | 507,06 € | 6.312,50 km | 67,27 h | 507,06 € | No | Mínima distancia / Combustible puro |
| **2,00 €/h** | 644,40 € | 6.348,07 km | 67,25 h | 509,91 € | No | Transición temprana |
| **5,09 €/h (SMI)** | **845,09 €** | **6.307,79 km** | **66,49 h** | **506,67 €** | **Sí (Base)** | **Equilibrio Económico Laboral** |
| **10,00 €/h** | 1.171,54 € | 6.307,79 km | 66,49 h | 506,67 € | **Sí** | Equilibrio Económico Laboral |
| **20,00 €/h** | 1.834,79 € | 6.292,41 km | 65,65 h | 505,44 € | No | Reducción Temporal (Alta Velocidad) |
| **50,00 €/h** | 3.815,92 € | 6.236,88 km | 65,82 h | 500,98 € | No | Dominancia severa del tiempo laboral |
| **100,00 €/h** | 7.124,84 € | 6.294,22 km | 66,03 h | 505,58 € | No | Saturación de penalización horaria |

---

## 3. Uso de Inteligencia Artificial y Cacería de la Alucinación

### 3.1 Prompts Principales Empleados
* **Prompt 1 (Geoapify API & Vectorización)**: Asistencia en la extracción modular de datos y separación de costos y geometría. Facilitó la creación limpia de las matrices 47x47.
* **Prompt 2 (Redacción & Estructuración)**: Asistencia en la organización pedagógica del informe técnico y justificación de métricas estadísticas (mediana logarítmica y tasa de éxito).

### 3.2 Cacería de la Alucinación
1. **Alucinación 1 (Pathlib & Rutas Asumidas)**: La IA inventó rutas fijas erróneas (`/datos/rutas_costes.csv`). Se corrigió anclando las rutas con `Path(__file__).resolve().parent`.
2. **Alucinación 2 (Coordenadas OSRM Inventadas)**: La IA generó un diccionario con coordenadas geográficas arbitrarias. Se corrigió tomando coordenadas oficiales validadas del IGN.
3. **Alucinación 3 (TollGuru & Re-consulta Redundante)**: La IA sugirió re-consultar Geoapify para cada tramo. Se demostró que las matrices precalculadas ya disponían de la información, ahorrando cuota y tiempo.
4. **Alucinación 4 (Porcentajes de Convergencia Falsos: 86,67% y 96,67%)**: La IA redactó que *PSO logró 86,67% en Rastrigin y DE logró 96,67% en Rosenbrock 3D*. Se auditó empíricamente contra los CSV: los valores reales son **93,33% para PSO en Rastrigin 3D** y **26,67% para DE en Rosenbrock 3D**.

---

## 4. Videos de Contribución Individual

Cada integrante del equipo grabó un video de 2 a 3 minutos hablando en primera persona sobre sus aportes específicos. El video muestra evidencia concreta: un fragmento de código que escribió, un commit, una figura que elaboró o una sección del reporte que redactó, y explica brevemente cómo funciona. Lo que cada persona declara en su video podrá ser preguntado en la sustentación oral:

| Integrante | Documento | Tema / Aporte Principal | Enlace al Video |
| :--- | :---: | :--- | :---: |
| **Alejandro Agudelo Anaya** | CC: 1017271139 | Descenso por Gradiente y PSO en Rosenbrock y Rastrigin | [Ver Video (YouTube)](https://www.youtube.com/watch?v=0caV8OajlQM) |
| **David Agudelo Arboleda** | CC: 1000899797 | Colonia de Hormigas (ACO) con Or-opt, modelado TSP y animaciones | [Ver Video (YouTube)](https://youtu.be/xvnyp074I1E) |
| **Santiago Álvarez Martínez** | CC: 1022142463 | Formulación metodológica, experimentación y análisis comparativo | [Ver Video (YouTube)](https://youtu.be/QNHQFw3YrUI) |
| **Daniel Felipe López Zapata** | CC: 1045046468 | Arquitectura de código, validación de datos viales y estructuración | [Ver Video (YouTube)](https://youtu.be/_6JfD148hN4) |
| **Sara Osorio Piñerez** | CC: 1001370113 | Verificación empírica, redacción técnica y auditoría de alucinaciones | [Ver Video (YouTube)](https://www.youtube.com/watch?v=W_dBpcZ_9g4) |

---

## 5. Guía de Ejecución en Un Solo Comando

El repositorio incluye un CLI unificado (`main.py`) para correr cualquier parte o ejercicio en un solo comando:

```bash
# Ejecución integral completa
python main.py

# Demostración rápida en vivo (< 30s)
python main.py --demo

# Menú interactivo por consola
python main.py --menu

# Ejercicios individuales
python main.py -e gd            # Descenso por Gradiente
python main.py -e pso           # Optimización por Enjambre de Partículas
python main.py -e ea            # Algoritmo Evolutivo Continuo
python main.py -e de            # Evolución Diferencial
python main.py -e aco           # Colonia de Hormigas (TSP 47 capitales)
python main.py -e ga            # Algoritmo Genético DEAP (TSP 47 capitales)
python main.py -e oropt         # Comparativa Con vs Sin Or-opt
python main.py -e sensibilidad  # Sensibilidad del salario [0, 100] EUR/h
python main.py -e graficas      # Regenerar todas las figuras 3D y barras
python main.py -e gifs          # Regenerar animaciones GIF
```

---

## 6. Referencias Bibliográficas (APA 7.ª Edición)

* Anfac. (s. f.). *Matriculaciones turismos y todoterreno*. https://anfac.com/cifras-clave/matriculaciones-turismos-y-todoterreno/
* Dorigo, M., & Stützle, T. (2004). *Ant colony optimization*. MIT Press. https://doi.org/10.7551/mitpress/1290.001.0001
* Eberhart, R., & Kennedy, J. (1995). A new optimizer using particle swarm theory. En *MHS'95. Proceedings of the Sixth International Symposium on Micro Machine and Human Science* (pp. 39–43). IEEE. https://doi.org/10.1109/MHS.1995.494215
* España. Ministerio de Trabajo y Economía Social. (2026). Real Decreto 126/2026, de 18 de febrero, por el que se fija el salario mínimo interprofesional para 2026. *Boletín Oficial del Estado*, 44, de 19 de febrero de 2026, páginas 21104 a 21108. https://www.boe.es/eli/es/rd/2026/02/18/126
* España. Ministerio de Transportes y Movilidad Sostenible. (s. f.). *Peajes dependientes de la AGE*. https://www.transportes.gob.es/carreteras/nuestra-red/autopistas-peaje/peajes-dependientes-de-la-age
* Fortin, F.-A., De Rainville, F.-M., Gardner, M.-A., Parizeau, M., & Gagné, C. (2012). DEAP: Evolutionary algorithms made easy. *Journal of Machine Learning Research*, 13(70), 2171–2175.
* Km77. (s. f.). *Toyota Corolla 140H Style Plus (2023): Datos técnicos y equipamiento*. https://www.km77.com/coches/toyota/corolla/2023/5-puertas/estandar/corolla-140h-style-plus/datos
* Ministerio para la Transición Ecológica y el Reto Demográfico. (s. f.). *Sistema de Información para la Geolocalización de Estaciones de Servicio y Precios de Carburantes*. https://energia.serviciosmin.gob.es/shpCarburantes/
* Nocedal, J., & Wright, S. J. (2006). *Numerical optimization* (2.ª ed.). Springer. https://doi.org/10.1007/978-0-387-40065-5
* Or, I. (1976). *Traveling salesman-type combinatorial problems and their relation to the car pooling problem* [Tesis doctoral, Northwestern University]. ProQuest Dissertations Publishing.
* Storn, R., & Price, K. (1997). Differential evolution–a simple and efficient heuristic for global optimization over continuous spaces. *Journal of Global Optimization*, 11(4), 341–359. https://doi.org/10.1023/A:1008202821328
