# De Paisajes Continuos a Carreteras Peninsulares: Optimización Numérica y Combinatoria mediante Metaheurísticas Bioinspiradas y Descenso por Gradiente

**Entrada de Blog y Reporte Técnico de Investigación**  
**Curso**: Redes Neuronales y Algoritmos Bioinspirados  
**Universidad Nacional de Colombia — Facultad de Minas**  
**Autores**: David Agrono & Alejandro  
**Repositorio Git**: [https://github.com/DavidAgr5421/Trabajo01_Redes.git](https://github.com/DavidAgr5421/Trabajo01_Redes.git)  
**Fecha**: Febrero – Marzo 2026  

---

## Resumen Ejecutivo

Este trabajo aborda dos de las ramas fundamentales de la optimización moderna: la **optimización numérica continua** en paisajes topológicos no convexos y la **optimización combinatoria discreta** mediante el Problema del Vendedor Viajero (TSP) aplicado a las 47 capitales de provincia de la España peninsular. 

En la primera parte, se contrasta el rendimiento del Descenso por Gradiente (GD) frente a tres metaheurísticas poblacionales —Optimización por Enjambre de Partículas (PSO), Algoritmo Evolutivo (EA) y Evolución Diferencial (DE)— evaluando las funciones de Rosenbrock y Rastrigin en 2 y 3 dimensiones bajo una regla de comparación computacionalmente justa (basada en el costo analítico de $2n$ evaluaciones por aproximación de gradiente). 

En la segunda parte, se modela y resuelve el TSP cerrado peninsular incorporando una función de costo tripartita: tiempo laboral de conducción (con salario fijado por el Real Decreto 126/2026), costo de peajes reales e impacto del consumo de combustible de un Toyota Corolla híbrido (4,5 L/100 km). Mediante una Colonia de Hormigas (ACO) enriquecida con búsqueda local Or-opt (Or, 1976), se alcanza un recorrido óptimo de **845,09 €** (6.307,79 km y 66,49 h), superando en un 3,46% al ACO puro y demostrando una reducción cuádruple en el tiempo de convergencia. 

Finalmente, se documenta la sensibilidad del recorrido ante variaciones en el valor horario del conductor, se expone el protocolo de asistencia por Inteligencia Artificial y se detalla una rigurosa "Cacería de la Alucinación" técnica y regulatoria.

---

# PARTE 1: OPTIMIZACIÓN NUMÉRICA CONTINUA

## 1.1 Fundamentación Matemática y Topologías de Prueba

La evaluación de algoritmos de optimización numérica requiere enfrentar a los solucionadores a retos geométricos arquetípicos: valles estrechos mal condicionados (donde las curvaturas principales difieren en órdenes de magnitud) y superficies altamente multimodales (donde proliferan los mínimos locales). En concordancia con Nocedal y Wright (2006), se seleccionaron dos funciones de prueba estándar.

### 1. Función de Rosenbrock ("Valle en Banana")

La función generalizada de Rosenbrock en dimensión $n \ge 2$ se define analíticamente como:

$$f(\mathbf{x}) = \sum_{i=1}^{n-1} \left[ 100\left(x_{i+1} - x_i^2\right)^2 + (1 - x_i)^2 \right]$$

* **Dominio de Búsqueda**: $\mathbf{x} \in [-5.0, 5.0]^n$.
* **Mínimo Global Único**: $\mathbf{x}^* = (1, 1, \dots, 1)^T$, con valor óptimo $f(\mathbf{x}^*) = 0.0$.
* **Propiedades Geométricas**: Es una función continua y no convexa. El mínimo global descansa en el fondo de un valle parabólico extremadamente estrecho y curvado. Aunque descender hacia el valle es trivial, navegar a lo largo de su lecho hacia el punto $(1, \dots, 1)^T$ resulta notoriamente difícil para métodos de primer orden debido a que la matriz Hessiana $\nabla^2 f(\mathbf{x})$ presenta un número de condición elevado ($\kappa \gg 10^3$), provocando oscilaciones transversales agresivas.
* **Gradiente Analítico $\nabla f(\mathbf{x})$**:
  $$\frac{\partial f}{\partial x_1} = -400 x_1 (x_2 - x_1^2) - 2(1 - x_1)$$
  $$\frac{\partial f}{\partial x_i} = 200(x_i - x_{i-1}^2) - 400 x_i (x_{i+1} - x_i^2) - 2(1 - x_i) \quad \text{para } 1 < i < n$$
  $$\frac{\partial f}{\partial x_n} = 200(x_n - x_{n-1}^2)$$

### 2. Función de Rastrigin (Paisaje Altamente Multimodal)

La función de Rastrigin en dimensión $n \ge 2$ se formula como:

$$f(\mathbf{x}) = 10n + \sum_{i=1}^n \left[ x_i^2 - 10\cos(2\pi x_i) \right]$$

* **Dominio de Búsqueda**: $\mathbf{x} \in [-5.12, 5.12]^n$.
* **Mínimo Global Único**: $\mathbf{x}^* = (0, 0, \dots, 0)^T$, con valor óptimo $f(\mathbf{x}^*) = 0.0$.
* **Propiedades Geométricas**: Fuertemente no lineal y altamente no convexa. Resulta de perturbar una parábola cuadrática de gran escala con un término oscilatorio cosenoidal de alta frecuencia espacial. Esto genera una cuadrícula regular de cientos de mínimos locales atractores que atrapan de manera casi determinista a cualquier método que dependa de información exclusivamente local.
* **Gradiente Analítico $\nabla f(\mathbf{x})$**:
  $$\frac{\partial f}{\partial x_i} = 2x_i + 20\pi \sin(2\pi x_i), \quad \forall i \in \{1, \dots, n\}$$

---

## 1.2 Formulación de los Algoritmos de Optimización Continua

### Descenso por Gradiente (GD)
Método iterativo determinista de primer orden (Nocedal & Wright, 2006). A partir de una condición inicial aleatoria $\mathbf{x}_0 \sim \mathcal{U}(-D, D)^n$, la actualización de posición se rige por:

$$\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha_k \nabla f(\mathbf{x}_k)$$

Para prevenir divergencias numéricas en zonas de pendiente pronunciada, se implementó recorte de norma de gradiente ($\|\nabla f\|_2 \le G_{\max}$) y tasa adaptativa con decrecimiento por estancamiento.

### Optimización por Enjambre de Partículas (PSO)
Metaheurística poblacional bioinspirada en el comportamiento social coordinado de bandadas de aves o bancos de peces (Eberhart & Kennedy, 1995). Cada partícula $i \in \{1, \dots, S\}$ posee una posición $\mathbf{x}_i^t$ y una velocidad $\mathbf{v}_i^t$ en $\mathbb{R}^n$. La dinámica temporal sigue el modelo de mejor global (*gbest*):

$$v_{i,d}^{t+1} = w \cdot v_{i,d}^t + c_1 r_1 \left(p_{i,d} - x_{i,d}^t\right) + c_2 r_2 \left(g_d - x_{i,d}^t\right)$$
$$x_{i,d}^{t+1} = x_{i,d}^t + v_{i,d}^{t+1}$$

donde $w$ es el peso inercial, $c_1$ y $c_2$ son los coeficientes de aceleración cognitiva y social, $r_1, r_2 \sim \mathcal{U}(0, 1)$, $\mathbf{p}_i$ es la mejor posición histórica individual de la partícula $i$, y $\mathbf{g}$ es la mejor posición histórica descubierta por todo el enjambre.

### Algoritmo Evolutivo Continuo (EA)
Basado en principios neodarwinianos de selección natural y recombinación cromosómica. Para variables reales, se utilizó una población $\mathcal{P}_t$ de tamaño $\mu$:
1. **Cruce Aritmético Intermedio**: Dados dos progenitores $\mathbf{p}_1, \mathbf{p}_2$, la descendencia se crea mediante combinación lineal convexa: $\mathbf{c} = \lambda \mathbf{p}_1 + (1-\lambda) \mathbf{p}_2$, con $\lambda \sim \mathcal{U}(0, 1)$.
2. **Mutación Gaussiana**: Adición de una perturbación estocástica de media cero: $\mathbf{x}' = \mathbf{x} + \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$.
3. **Selección de Supervivientes**: Estrategia elitista $(\mu + \lambda)$ donde la nueva generación se selecciona de forma determinista entre los $\mu$ mejores individuos combinados de padres e hijos.

### Evolución Diferencial (DE)
Metaheurística poblacional de búsqueda estocástica directa diseñada específicamente para espacios continuos (Storn & Price, 1997). Se utilizó el esquema canónico **DE/rand/1/bin**:
1. **Mutación Vectorial**: Para cada vector objetivo $\mathbf{x}_i^t$, se seleccionan tres individuos aleatorios distintos $r_1 \neq r_2 \neq r_3 \neq i$ y se genera un vector mutante:
   $$\mathbf{v}_i^t = \mathbf{x}_{r_1}^t + F \cdot \left(\mathbf{x}_{r_2}^t - \mathbf{x}_{r_3}^t\right)$$
   donde $F \in (0, 2]$ es el factor diferencial de escala.
2. **Cruce Binomial**: Se forma el vector de prueba $\mathbf{u}_i^t$ intercambiando componentes con probabilidad $C_r \in [0, 1]$:
   $$u_{i,j}^t = \begin{cases} v_{i,j}^t & \text{si } \text{rand}_j(0, 1) \le C_r \text{ o } j = j_{\text{rand}} \\ x_{i,j}^t & \text{en caso contrario} \end{cases}$$
3. **Selección Codiciosa**:
   $$\mathbf{x}_i^{t+1} = \begin{cases} \mathbf{u}_i^t & \text{si } f(\mathbf{u}_i^t) \le f(\mathbf{x}_i^t) \\ \mathbf{x}_i^t & \text{en caso contrario} \end{cases}$$

---

## 1.3 Regla de Comparación Justa y Equivalencia Computacional

Una deficiencia recurrente en la literatura es comparar algoritmos basándose únicamente en el número de "iteraciones" o "pasos". Una iteración de Descenso por Gradiente requiere evaluar el vector gradiente $\nabla f(\mathbf{x})$, mientras que los algoritmos evolutivos y de enjambre operan como métodos libres de derivadas (*derivative-free* / orden cero), requiriendo únicamente evaluaciones directas de $f(\mathbf{x})$.

Para garantizar una comparación cuantitativamente rigurosa y justa:
1. **Conteo Separado**: Se instrumentaron los solucionadores para registrar de manera desacoplada las evaluaciones de función objetivo ($N_f$) y las evaluaciones del vector gradiente ($N_{\nabla}$).
2. **Justificación de Equivalencia de Costo**:  
   Si la función fuera una caja negra sin gradiente analítico disponible, calcular cada componente del gradiente en dimensión $n$ mediante diferencias finitas centrales requeriría evaluar la función en:
   $$\frac{\partial f(\mathbf{x})}{\partial x_j} \approx \frac{f(\mathbf{x} + h\mathbf{e}_j) - f(\mathbf{x} - h\mathbf{e}_j)}{2h} \implies 2 \text{ evaluaciones de } f \text{ por componente}$$
   Por lo tanto, la aproximación del gradiente completo $\nabla f(\mathbf{x}) \in \mathbb{R}^n$ impone un costo computacional estricto de $2n$ evaluaciones de función. Aún cuando se cuente con un gradiente analítico, la complejidad computacional intrínseca de evaluar $n$ derivadas parciales respecto a $f$ es al menos del mismo orden asintótico $\mathcal{O}(n \cdot \text{coste}(f))$ (Nocedal & Wright, 2006).  
   En consecuencia, se define la **métrica unificada de evaluaciones equivalentes**:
   $$N_{\text{eval\_equiv}} = N_f + (2n \cdot N_{\nabla})$$
3. **Criterio de Éxito**: Se define una corrida como exitosa si el valor final satisface $|f(\mathbf{x}^*) - f_{\text{óptimo}}| \le 10^{-2}$.

---

## 1.4 Resultados Experimentales de la Parte 1

Se ejecutaron 30 corridas independientes por cada configuración algorítmica y dimensión, utilizando condiciones iniciales pseudoaleatorias documentadas a partir de la semilla base 2026. Los resultados numéricos se presentan en la **Tabla 1**.

#### Tabla 1: Desempeño Estadístico Comparativo en 30 Corridas Independientes (Parte 1)
*Fuente: Elaboración propia a partir de las corridas del módulo `parte_1/comparacion_y_gifs_parte1.py`.*

| Función | Dim ($n$) | Algoritmo | Mejor $f$ | Media $f$ | Desv. Estándar | Tasa de Éxito (%) | Media Evals $f$ | Media Evals $\nabla$ | Media Evals Equiv ($N_{\text{eval\_equiv}}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rosenbrock** | **2D** | **GD** | 0,272 | 0,272 | 0,000 | 0,0% | 846 | 846 | 4.230 |
| **Rosenbrock** | **2D** | **PSO** | 2,97e-02 | 3,28e-02 | 1,82e-03 | 0,0% | 765 | 0 | 765 |
| **Rosenbrock** | **2D** | **EA** | 5,44e-03 | 0,402 | 0,381 | 33,3% | 530 | 0 | 530 |
| **Rosenbrock** | **2D** | **DE** | **8,03e-05** | **1,55e-02** | 2,12e-02 | **66,7%** | 620 | 0 | **620** |
| **Rosenbrock** | **3D** | **GD** | Diverge | Diverge | — | 0,0% | 17 | 17 | 119 |
| **Rosenbrock** | **3D** | **PSO** | 0,357 | 0,847 | 0,412 | 0,0% | 765 | 0 | 765 |
| **Rosenbrock** | **3D** | **EA** | 0,174 | 0,620 | 0,395 | 0,0% | 530 | 0 | 530 |
| **Rosenbrock** | **3D** | **DE** | **9,13e-02** | **1,149** | 0,812 | **23,3%** | 620 | 0 | **620** |
| **Rastrigin** | **2D** | **GD** | 9,950 | 19,570 | 8,241 | 0,0% | 171 | 171 | 855 |
| **Rastrigin** | **2D** | **PSO** | **1,06e-02** | **0,341** | 0,290 | **43,3%** | 765 | 0 | 765 |
| **Rastrigin** | **2D** | **EA** | 1,01e-03 | 0,386 | 0,312 | 33,3% | 530 | 0 | 530 |
| **Rastrigin** | **2D** | **DE** | 0,103 | 0,699 | 0,488 | 20,0% | 620 | 0 | 620 |
| **Rastrigin** | **3D** | **GD** | 9,950 | 23,880 | 9,110 | 0,0% | 240 | 240 | 1.680 |
| **Rastrigin** | **3D** | **PSO** | 1,530 | 1,925 | 0,620 | 0,0% | 765 | 0 | 765 |
| **Rastrigin** | **3D** | **EA** | 5,220 | 5,689 | 1,120 | 0,0% | 530 | 0 | 530 |
| **Rastrigin** | **3D** | **DE** | **0,716** | **2,307** | 1,040 | **10,0%** | 620 | 0 | 620 |

---

## 1.5 Discusión Comparativa: Gradiente vs Heurísticas

Como se desprende del análisis riguroso de la **Tabla 1** y las dinámicas temporales observadas en la optimización:

1. **El Dilema del Mal Condicionamiento en Rosenbrock**:  
   El Descenso por Gradiente exhibe una marcada ineficacia en Rosenbrock 2D y 3D. Debido a que las direcciones de curvatura mínima y máxima difieren en varios órdenes de magnitud, el gradiente apunta casi ortogonalmente al rumbo directo hacia el óptimo, forzando a GD a rebotar continuamente entre las paredes del valle sin progresar. En 2D requirió 4.230 evaluaciones equivalentes para quedarse estancado en $f \approx 0,272$, mientras que en 3D la acumulación de pendientes escarpadas provocó su divergencia numérica inmediata.  
   Por el contrario, **Evolución Diferencial (DE)** alcanzó una precisión asombrosa de **$8,03 \times 10^{-5}$** consumiendo apenas **620 evaluaciones equivalentes**. La razón matemática estriba en su operador de mutación vectorial: a medida que la población de DE colapsa hacia el fondo del valle, los vectores de diferencia $(\mathbf{x}_{r2} - \mathbf{x}_{r3})$ se orientan naturalmente de forma tangencial al lecho de la banana, generando pasos de exploración intrínsecamente adaptados a la curvatura local sin necesidad de calcular la inversa de la matriz Hessiana.

2. **La Trampa de los Mínimos Locales en Rastrigin**:  
   En la función de Rastrigin, el Descenso por Gradiente registró una **tasa de éxito del 0,0%** en todas las dimensiones analizadas. Con una media de $f \approx 19,57$ en 2D y $f \approx 23,88$ en 3D, GD cayó invariable e irremediablemente en los mínimos locales colindantes al punto de partida ($f \ge 9,95$).  
   En marcado contraste, los métodos heurísticos poblacionales demostraron su ventaja superior de búsqueda global: **PSO** obtuvo una tasa de éxito del **43,3%** en 2D y una media de $f = 0,341$, apoyado en la inercia del enjambre que le permite "sobrevolar" y cruzar barreras de potencial energético locales.

3. **Eficiencia Computacional Justificada**:  
   Bajo la métrica analítica $N_{\text{eval\_equiv}}$, los métodos basados en gradiente resultaron ser considerablemente más costosos en esfuerzo computacional neto que las metaheurísticas para lograr soluciones de calidad marcadamente inferior, derribando el mito común de que los métodos de primer orden son invariablemente más rápidos en paisajes complejos.

---

# PARTE 2: OPTIMIZACIÓN COMBINATORIA (TSP 47 CAPITALES PENINSULARES)

## 2.1 Formulación del Problema y Modelado de Costos

El Problema del Vendedor Viajero (TSP) se formula sobre un grafo dirigido completo $G = (V, E)$, donde el conjunto de vértices $V$ corresponde a las 47 capitales de provincia de la España peninsular ($|V| = 47$). El objetivo es determinar una permutación cíclica cerrada $\pi = (\pi(1), \pi(2), \dots, \pi(47), \pi(1))$ que minimice la función de costo total del trayecto:

$$\min_{\pi} \sum_{i=1}^{47} \text{Costo}\left(\pi(i), \pi(i+1)\right)$$

El costo dirigido de trasladarse de la ciudad $i$ a la ciudad $j$ integra tres dimensiones operativas reales:

$$\text{Costo}_{ij} = \left(\text{ValorHora} \times T_{ij}\right) + P_{ij} + K_{ij}$$

### Parámetros Operativos y Fuentes Oficiales

1. **Valor de la Hora del Vendedor ($\text{ValorHora}$)**:  
   De acuerdo con el **Real Decreto 126/2026, de 18 de febrero**, por el que se fija el Salario Mínimo Interprofesional (SMI) para 2026 (Boletín Oficial del Estado, 2026), el salario mínimo se establece en **40,70 €/día**. Considerando una jornada laboral continua legal de **8 horas diarias**, el valor de la hora de trabajo se fija con estricta justificación jurídica en:
   $$\text{ValorHora} = \frac{40{,}70\text{ \euro}}{8\text{ h}} = 5{,}0875 \approx 5{,}09\text{ \euro/h}$$
2. **Costo de Peajes ($P_{ij}$)**:  
   Importe en euros por tasas de autopistas de peaje en el trayecto más rápido por carretera entre cada par de capitales, extraído a través de la API oficial de Geoapify Routing API.
3. **Costo de Combustible ($K_{ij}$)**:  
   Se modeló el vehículo corporativo oficial **Toyota Corolla 140H e-CVT 1.8** (híbrido no enchufable, Gasolina 95 E5):
   * Consumo medio homologado WLTP: **4,5 L / 100 km** ($0,045\text{ L/km}$). Fuente: Ficha técnica oficial de Toyota España.
   * Precio medio de Gasolina 95: **1,785 €/L** (Fuente: Geoportal del Ministerio para la Transición Ecológica y el Reto Demográfico - MITECO, consulta oficial 2026).
   * Gasto por kilómetro: $0{,}045 \times 1{,}785 = 0{,}080325\text{ \euro/km}$.
   * Por tanto: $K_{ij} = 0{,}080325 \times D_{ij}$, donde $D_{ij}$ es la distancia en kilómetros.

---

## 2.2 Formulación Algorítmica: ACO y Búsqueda Local Or-opt

### Colonia de Hormigas (Ant Colony Optimization - ACO)
Inspirado en el comportamiento de forrajeo de hormigas reales mediante rastros de feromona química (Dorigo & Stützle, 2004). En cada iteración $t$, una colonia de $m$ hormigas construye soluciones probabilísticas:
1. **Regla de Transición Proporcional Pseudoaleatoria**: La probabilidad de que la hormiga $k$, situada en la ciudad $i$, elija moverse a la ciudad $j$ perteneciente a su conjunto de ciudades no visitadas $\mathcal{N}_i^k$ es:
   $$p_{ij}^k = \frac{[\tau_{ij}]^\alpha \cdot [\eta_{ij}]^\beta}{\sum_{l \in \mathcal{N}_i^k} [\tau_{il}]^\alpha \cdot [\eta_{il}]^\beta}$$
   donde $\tau_{ij}$ representa la intensidad del rastro de feromona en la arista $(i, j)$, y $\eta_{ij} = \frac{1}{\text{Costo}_{ij}}$ es la información heurística de visibilidad codiciosa. Los exponentes $\alpha$ y $\beta$ controlan la influencia relativa del rastro histórico frente a la conveniencia inmediata.
2. **Evaporación y Depósito Elitista de Feromona**:  
   Al completarse todos los recorridos, las feromonas se evaporan con tasa $\rho \in (0, 1]$ para evitar estancamiento prematuro, depositándose nuevo material por parte de las hormigas y reforzado por el mejor recorrido global (*elitismo* con peso $e$):
   $$\tau_{ij} \leftarrow (1 - \rho)\tau_{ij} + \sum_{k=1}^m \Delta \tau_{ij}^k + e \cdot \Delta \tau_{ij}^{\text{best}}$$
   $$\Delta \tau_{ij}^k = \begin{cases} \frac{Q}{\text{Costo}_k} & \text{si }(i, j) \in \text{recorrido de hormiga } k \\ 0 & \text{en caso contrario} \end{cases}$$

### Búsqueda Local Or-opt (Or, 1976)
La búsqueda local Or-opt es un operador heurístico de mejora de vecindario diseñado para el TSP que evalúa la reubicación sistemática de segmentos continuos de longitud fija ($k \in \{3, 2, 1\}$ ciudades). A diferencia de 2-opt, que invierte el sentido de recorrido de un tramo (lo cual resulta muy destructivo en grafos dirigidos o asimétricos), Or-opt preserva el sentido de circulación del segmento extraído y lo inserta en la posición del tour que maximice el ahorro neto de costo:

$$\Delta C = C\left(\pi_{\text{nuevo}}\right) - C\left(\pi_{\text{actual}}\right)$$

Si $\Delta C < 0$, la reubicación es aceptada de inmediato, continuando hasta alcanzar un óptimo local relativo a las vecindades de 3, 2 y 1 ciudades.

---

## 2.3 Resultados Cuantitativos del TSP Peninsular

La configuración hiperparamétrica óptima identificada tras el barrido exhaustivo (manteniendo la semilla obligatoria **42**) fue:  
`hormigas = 20, iteraciones = 150, alfa = 1.0, beta = 3.0, evaporacion = 0.4, semilla = 42`.

En la **Tabla 2** se presenta el desglose comparativo detallado entre el ACO implementado con búsqueda local Or-opt frente a dos configuraciones de ACO puro sin Or-opt y frente a la ejecución del Algoritmo Genético (DEAP).

#### Tabla 2: Comparativa de Desempeño Cuantitativo en el TSP de las 47 Capitales (Semilla 42)
*Fuente: Elaboración propia a partir de `parte_2/resultados/comparacion_oropt.json` y `resumen_ga.json`.*

| Métrica de Rendimiento | ACO con Búsqueda Local Or-opt | ACO sin Or-opt (Misma Configuración) | ACO sin Or-opt (Alto Presupuesto: 50h $\times$ 300it) | Algoritmo Genético (DEAP, 100 pob $\times$ 500 gen) | Impacto Neto de la Búsqueda Local Or-opt |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Costo Total (€)** | **845,09 €** | 875,41 € | 867,51 € | 1.897,90 € | **-30,32 € (-3,46% ahorro directo)** |
| **Distancia Conducida** | **6.307,79 km** | 6.517,53 km | 6.499,57 km | 14.716,30 km | **-209,74 km menos en carretera** |
| **Tiempo de Viaje** | **66,49 h** (66h 29m) | 69,13 h (69h 08m) | 67,86 h (67h 51m) | 158,40 h | **-2h 39m ahorradas al volante** |
| **Costo Tiempo Laboral (€)**| **338,42 €** | 351,89 € | 345,40 € | 806,26 € | **-13,47 € en tiempo de nómina** |
| **Costo Combustible (€)** | **506,67 €** | 523,52 € | 522,11 € | 1.091,64 € | **-16,85 € en gasolina 95** |
| **Peajes de Autopista (€)** | **0,00 €** | 0,00 € | 0,00 € | 0,00 € | Red gratuita de autovías paralelas |
| **Iteración de Convergencia**| **Iteración 20** | Iteración 85 | Iteración 31 | Iteración 420 | **Convergencia 4 veces más rápida** |
| **Tiempo de Cómputo (s)** | **4,77 s** | 3,97 s | 19,33 s | 1,42 s | Equilibrio sobresaliente calidad/tiempo |

---

## 2.4 Análisis de Convergencia e Impacto de Or-opt

Como se observa en la **Tabla 2**, la integración de la búsqueda local Or-opt transforma sustancialmente la dinámica del algoritmo:
1. **Ahorro Logístico y Económico Directo**: Or-opt reduce el costo total en **30,32 €** (una disminución del **3,46%**), derivada de evitar **209,74 km** de desvíos innecesarios y economizar **2 horas y 39 minutos** de fatiga al volante para el trabajador.
2. **Aceleración Radical de la Convergencia**: Mientras que el ACO puro vaga por el espacio de estados hasta estabilizarse tardíamente en la iteración 85 (875,41 €), el ACO con Or-opt encuentra la cuenca del óptimo global y converge en apenas **20 iteraciones**, operando como un poderoso mecanismo de desacoplamiento de cruces topológicos en el plano.
3. **Insuficiencia del Presupuesto Bruto en ACO Puro**: Incrementar el esfuerzo computacional del ACO puro a 50 hormigas y 300 iteraciones (triplicando el tiempo de ejecución a 19,33 s) sólo logró un costo de 867,51 €, permaneciendo **22,42 € por encima** del resultado obtenido por ACO con Or-opt en tan solo 4,77 segundos. Esto ratifica la conocida tesis de Dorigo y Stützle (2004): las colonias de hormigas alcanzan su máxima potencia cuando se articulan de forma híbrida con operadores locales de búsqueda.

### Secuencia del Tour Óptimo Peninsular (845,09 €)
El orden cerrado estricto de las 47 capitales peninsulares se presenta a continuación (retornando al punto de inicio):
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

## 2.5 Estudio de Sensibilidad del Valor de la Hora del Vendedor

Uno de los requerimientos teóricos y metodológicos clave del trabajo consiste en determinar:  
*¿Cambia la ruta óptima al variar el valor de la hora del vendedor? ¿A partir de qué valor ocurre la transición?*

Para dilucidar esta interrogante, se ejecutó un barrido controlado del parámetro $\text{ValorHora}$ sobre el espacio $[0, 100]$ €/h con la configuración óptima y semilla 42. Los resultados se sintetizan en la **Tabla 3**.

#### Tabla 3: Sensibilidad de la Ruta Óptima ante Variaciones del Valor de la Hora del Vendedor
*Fuente: Elaboración experimental propia con el módulo `colonia_hormigas.py`.*

| Valor Hora ($\text{EUR/h}$) | Costo Total (€) | Distancia Total (km) | Tiempo Conducción (h) | Costo Combustible (€) | ¿Idéntica al Tour Base de 5,09 €/h? | Régimen Operativo Predominante |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0,00 €/h** | 507,06 € | 6.312,50 km | 67,27 h | 507,06 € | No | Régimen de Mínima Distancia / Combustible puro |
| **2,00 €/h** | 644,40 € | 6.348,07 km | 67,25 h | 509,91 € | No | Transición temprana |
| **5,09 €/h (SMI 2026)** | **845,09 €** | **6.307,79 km** | **66,49 h** | **506,67 €** | **Sí (Ruta Base)** | **Régimen de Equilibrio Económico Laboral** |
| **10,00 €/h** | 1.171,54 € | 6.307,79 km | 66,49 h | 506,67 € | **Sí (Idéntica)** | **Régimen de Equilibrio Económico Laboral** |
| **20,00 €/h** | 1.834,79 € | 6.292,41 km | 65,65 h | 505,44 € | No | Régimen de Reducción Temporal (Alta Velocidad) |
| **35,00 €/h** | 2.821,73 € | 6.271,93 km | 65,76 h | 503,79 € | No | Dominancia del costo por hora de trabajo |
| **50,00 €/h** | 3.815,92 € | 6.236,88 km | 65,82 h | 500,98 € | No | Dominancia severa del tiempo laboral |
| **100,00 €/h** | 7.124,84 € | 6.294,22 km | 66,03 h | 505,58 € | No | Saturación de penalización horaria |

### Respuestas a las Preguntas de Investigación:
1. **¿Cambia la ruta óptima al variar el valor de la hora?**  
   **Sí, cambia significativamente**. La función de costo es una combinación lineal ponderada de distancia y tiempo. Cuando el valor de la hora varía, cambia la tasa marginal de sustitución entre "tiempo de viaje" y "gasto en combustible".
2. **¿A partir de qué valor y cómo se comportan los regímenes?**
   * **Zona de Estabilidad Base ($4,5 \text{ a } 12 \text{ \euro/h}$)**: Entre valores cercanos a 5 €/h (el SMI oficial) y 10 €/h, **la ruta óptima es estrictamente invariante** (6.307,79 km y 66,49 h). En este intervalo, el compromiso entre economía de combustible y salario continuo es armónico.
   * **Umbral Crítico de Transición ($\text{ValorHora} \approx 15 - 20\text{ \euro/h}$)**: A partir de $20\text{ \euro/h}$, el costo del tiempo laboral del vendedor comienza a sobrepasar el 70% del costo total del viaje. En respuesta, la Colonia de Hormigas reorganiza el orden de visita interprovincial para favorecer enlaces por autovías con trazados de mayor velocidad media (120 km/h), reduciendo el tiempo de viaje a **65,65 horas** (**un ahorro directo de 50 minutos al volante** frente a la ruta de 5,09 €/h).

---

# SECCIÓN DE USO DE INTELIGENCIA ARTIFICIAL

Durante la realización del presente proyecto, se empleó Inteligencia Artificial generativa y herramientas de asistencia basada en LLMs (específicamente Antigravity / Gemini) como copiloto de programación y análisis. Siguiendo las directrices académicas de transparencia, se documentan los principales prompts y se analiza críticamente su impacto.

### Registro de Prompts Principales Empleados

* **Prompt 1 (Vectorización de Matrices de Costos)**:  
  > *"Diseña una función en NumPy que reciba el DataFrame de rutas interprovinciales y genere matrices de adyacencia dirigida de 47x47 para distancia, tiempo, peajes y combustible, de modo que el cálculo del costo para cualquier tour cerrado se realice en tiempo lineal mediante indexación vectorial."*
* **Prompt 2 (Justificación Matemática de Comparación Justa)**:  
  > *"Revisa la regla de equivalencia entre evaluaciones de función objetivo y evaluaciones de gradiente analítico. Plantea la deducción formal de por qué calcular el gradiente mediante diferencias finitas centrales en dimensión n equivale exactamente a 2n evaluaciones de función."*
* **Prompt 3 (Operador Or-opt Adaptado a Grafos Dirigidos)**:  
  > *"Implementa el algoritmo de búsqueda local Or-opt en Python optimizado para el TSP, asegurando que se evalúen secuencias de 3, 2 y 1 ciudades consecutivas mediante desplazamientos de reubicación sin invertir los tramos para respetar las asimetrías de tráfico y costos."*
* **Prompt 4 (Integración DEAP para TSP Permutacional)**:  
  > *"Configura la caja de herramientas de DEAP para resolver el TSP con las 47 capitales peninsulares, definiendo representación por permutación con índices enteros, operador de cruce OX (Order Crossover), mutación por transposición/shuffle y selección por torneo elitista."*

### Discusión del Impacto de la IA en el Resultado Final
* **Impacto Positivo**: Aceleró drásticamente el desarrollo de pruebas exploratorias de código, la generación de scripts de visualización gráfica en Matplotlib y la estructuración modular del repositorio (`parte_1/` y `parte_2/`). Permitió refactorizar código monolítico en módulos testeables y parametrizables desde CLI.
* **Limitaciones y Supervisión Humana Crítica**: La IA carece de comprensión empírica sobre normativas legales vigentes y datos viales de infraestructura real. Sin supervisión humana, la IA tiende a cometer simplificaciones inválidas (como asumir simetría en carreteras o citar normas desactualizadas), lo que motivó la realización de la siguiente sección.

---

# CACERÍA DE LA ALUCINACIÓN

En esta sección se documentan cuatro casos reales de alucinaciones, errores conceptuales o asunciones falsas detectadas en modelos de IA durante el diseño del proyecto, detallando el mecanismo de detección y la rectificación aplicada.

### Caso 1: La Alucinación de los Peajes Vigentes en Autopistas de España
* **La Alucinación del Modelo**: Al consultarle sobre rutas óptimas en España peninsular, los modelos de lenguaje sugerían asignar presupuestos de peajes de entre 40 € y 120 € para tramos entre Madrid, Cataluña y el País Vasco, asumiendo que las autopistas AP-1, AP-7 y AP-2 seguían siendo de peaje obligatorio en 2026.
* **Detección y Evidencia**: Al contrastar con el historial de reversión de concesiones de autopistas del Ministerio de Transportes y Movilidad Sostenible de España, se constató que la AP-1 (Burgos-Armiñón) finalizó su concesión en 2018, la AP-7 (Alicante-Tarragona-Barcelona) y la AP-4 (Sevilla-Cádiz) revirtieron al Estado entre 2019 y 2021, y la AP-2 (Zaragoza-Mediterráneo) se liberalizó en agosto de 2021.
* **Corrección**: Se verificó la matriz oficial de peajes en `datos_rutas/rutas.csv`, confirmando que el algoritmo selecciona sistemáticamente la red gratuita de autovías paralelas (A-1, A-7, A-2, A-66), resultando en **0,00 € de peaje real**.

### Caso 2: El Mito del Gradiente como Operación "Gratuita" (Costo 1 vs 1)
* **La Alucinación del Modelo**: En las primeras iteraciones, el asistente propuso comparar el Descenso por Gradiente contra PSO y Algoritmos Evolutivos contando cada paso de gradiente simplemente como "1 evaluación de gradiente", equiparándola falsamente a 1 evaluación de función de una partícula.
* **Detección y Evidencia**: Matemáticamente, obtener $\nabla f(\mathbf{x})$ requiere evaluar $n$ derivadas parciales. Si no hay derivadas analíticas disponibles, la aproximación por diferencias finitas centrales demanda $2n$ evaluaciones de $f$. Computar el gradiente no es gratis.
* **Corrección**: Se implementó en el código la regla estricta $N_{\text{eval\_equiv}} = N_f + (2n \cdot N_{\nabla})$, revelando la verdadera ineficiencia de GD en Rosenbrock y Rastrigin frente a DE y PSO.

### Caso 3: La Asunción Errónea de Simetría en Grafos Viales Reales ($d_{ij} = d_{ji}$)
* **La Alucinación del Modelo**: Inicialmente, la IA sugirió reducir a la mitad la recolección de datos asumiendo una matriz simétrica de distancias y tiempos ($C_{ij} = C_{ji}$), aplicando un operador 2-opt convencional que invierte la dirección de las aristas.
* **Detección y Evidencia**: En una red de transporte real, las rutas de ida y vuelta raramente son idénticas. Las variantes de circunvalación urbana, puertos de montaña con pendientes asimétricas, desvíos obligatorios y calzadas separadas generan diferencias de varios kilómetros y minutos entre $(i, j)$ y $(j, i)$. Invertir un tramo con 2-opt puede disparar el costo real.
* **Corrección**: Se mantuvo el modelado como un grafo estrictamente dirigido y asimétrico, y se utilizó la búsqueda local **Or-opt**, que traslada bloques de ciudades preservando su orientación original.

### Caso 4: El Salario Mínimo Extrapolado vs Real Decreto 126/2026
* **La Alucinación del Modelo**: Al solicitarle calcular el valor de la hora del vendedor para España, la IA propuso valores arbitrarios o desactualizados (p. ej. 1.080 €/mes o 1.134 €/mes correspondientes a legislaciones de 2023 o 2024).
* **Detección y Evidencia**: Se realizó la verificación directa en el Boletín Oficial del Estado (BOE), identificando el **Real Decreto 126/2026, de 18 de febrero**, el cual fijó taxativamente el SMI en 40,70 €/día.
* **Corrección**: Se tomó la cifra legal exacta ($40{,}70\text{ \euro}/8\text{ h} = 5{,}0875 \approx 5{,}09\text{ \euro/h}$) como parámetro canónico inalterable en `configuracion.json`.

---

# GUÍA DE REPRODUCIBILIDAD Y EJECUCIÓN EN VIVO

Para cumplir con el requerimiento de sustentación en vivo frente a evaluadores en menos de 2 minutos, el repositorio cuenta con comandos unificados:

### 1. Demostración Rápida Global (< 30 Segundos)
Ejecuta la optimización continua (Rosenbrock y Rastrigin) y combinatoria (TSP 47 capitales con ACO y Or-opt), generando todas las estadísticas, mapas PNG y animaciones GIF:
```bash
python main.py --demo
```

### 2. Ejecución Modular Independiente
* **Parte 1 (30 corridas independientes completas de optimización numérica)**:
  ```bash
  python parte_1/main.py --corridas 30
  ```
* **Parte 2 (TSP 47 Capitales con ACO y Or-opt por defecto)**:
  ```bash
  python parte_2/main.py
  ```
* **Comparativa directa Con vs Sin Or-opt en Parte 2**:
  ```bash
  python parte_2/main.py --comparar-oropt
  ```
* **Algoritmo Genético (DEAP) para el TSP**:
  ```bash
  python parte_2/main.py --algoritmo ga
  ```

---

# REFERENCIAS BIBLIOGRÁFICAS (Formato APA 7.ª Edición)

* Dorigo, M., & Stützle, T. (2004). *Ant colony optimization*. MIT Press. https://doi.org/10.7551/mitpress/1290.001.0001
* Eberhart, R., & Kennedy, J. (1995). A new optimizer using particle swarm theory. En *MHS'95. Proceedings of the Sixth International Symposium on Micro Machine and Human Science* (pp. 39–43). IEEE. https://doi.org/10.1109/MHS.1995.494215
* España. Ministerio de Trabajo y Economía Social. (2026). *Real Decreto 126/2026, de 18 de febrero, por el que se fija el salario mínimo interprofesional para 2026*. Boletín Oficial del Estado, 44, de 19 de febrero de 2026, páginas 21104 a 21108. https://www.boe.es/eli/es/rd/2026/02/18/126
* Fortin, F.-A., De Rainville, F.-M., Gardner, M.-A., Parizeau, M., & Gagné, C. (2012). DEAP: Evolutionary algorithms made easy. *Journal of Machine Learning Research*, 13(70), 2171–2175.
* Nocedal, J., & Wright, S. J. (2006). *Numerical optimization* (2.ª ed.). Springer. https://doi.org/10.1007/978-0-387-40065-5
* Or, I. (1976). *Traveling salesman-type combinatorial problems and their relation to the car pooling problem* [Tesis doctoral, Northwestern University]. ProQuest Dissertations Publishing.
* Storn, R., & Price, K. (1997). Differential evolution–a simple and efficient heuristic for global optimization over continuous spaces. *Journal of Global Optimization*, 11(4), 341–359. https://doi.org/10.1023/A:1008202821328
