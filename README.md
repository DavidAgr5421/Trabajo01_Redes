# Trabajo 01: Optimización Numérica y Combinatoria (Algoritmos Bioinspirados)
**Curso**: Redes Neuronales y Algoritmos Bioinspirados  
**Universidad Nacional de Colombia — Facultad de Minas**  
**Autores**: David Agrono & Alejandro  

---

## Estructura General del Repositorio

El repositorio está organizado en dos módulos independientes y autosuficientes, gobernados por un archivo de configuración centralizado (`configuracion.json`) y una interfaz de línea de comandos unificada (`main.py`):

```
Entrega1/
├── configuracion.json            # Configuración global central (Parte 1, Parte 2 y Perfil Demo)
├── config.py                     # Módulo Python para acceso y sobreescritura de configuración
├── main.py                       # CLI unificada global (ejecuta Parte 1, Parte 2 o ambas)
├── requirements.txt              # Entorno con versiones fijadas (numpy, pandas, matplotlib, deap, etc.)
├── README.md                     # Reporte técnico completo y guía de ejecución
│
├── parte_1/                      # PARTE 1: OPTIMIZACIÓN NUMÉRICA
│   ├── config.py                 # Conector local a la configuración global
│   ├── funciones.py              # Definición de funciones de prueba y gradientes analíticos
│   ├── gradienteDecendente.py    # Descenso por Gradiente con recorte de paso y detección de divergencia
│   ├── PSO.py                    # Optimización por Enjambre de Partículas (gbest)
│   ├── algoritmo_evolutivo.py    # Algoritmo Evolutivo real (cruce aritmético, mutación gaussiana, élite)
│   ├── evolucion_diferencial.py  # Evolución Diferencial (DE/rand/1/bin con selección codiciosa)
│   ├── comparacion_y_gifs_parte1.py # Protocolo de 30 corridas independientes y generación de animaciones
│   ├── main.py                   # Runner CLI específico de Parte 1
│   └── resultados/               # CSVs de estadísticas, JSONs y animaciones GIF de trayectorias
│
└── parte_2/                      # PARTE 2: OPTIMIZACIÓN COMBINATORIA (TSP 47 CAPITALES)
    ├── config.py                 # Conector local a la configuración global
    ├── colonia_hormigas.py       # Algoritmo ACO (Ant System + Depósito Elitista + Búsqueda Or-opt)
    ├── algoritmo_genetico.py     # Algoritmo Genético para TSP (DEAP, cruce OX, mutación shuffle)
    ├── costes_comunes.py         # Modelado y cálculo de matrices dirigidas de costo, tiempo y distancia
    ├── experimentos_aco.py       # Barrido sistemático de hiperparámetros (semilla 42 fija) y comparativas
    ├── generar_gif.py            # Generación de mapas PNG y animaciones GIF sobre el mapa peninsular
    ├── validacion.py             # Validador de consistencia e integridad del grafo de 47 capitales
    ├── main.py                   # Runner CLI específico de Parte 2
    ├── datos_rutas/              # Base de datos oficial de distancias, tiempos, peajes y combustible
    │   ├── ciudades.csv, fuentes.csv, rutas.csv, rutas_costes.csv
    │   └── scripts_preparacion/  # Scripts de extracción histórica de Geoapify e inventario
    └── resultados/               # Rutas óptimas, CSVs de convergencia, mapas PNG y animaciones GIF
```

---

# PARTE 1: OPTIMIZACIÓN NUMÉRICA

## 1.1 Funciones de Prueba Seleccionadas

Se seleccionaron dos funciones continuas con geometrías radicalmente distintas para evaluar el comportamiento local frente al global:

### 1. Función de Rosenbrock (Valle Estrecho / Mal Condicionado)
$$f(\mathbf{x}) = \sum_{i=1}^{n-1} \left[ 100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2 \right]$$
* **Dominio**: $\mathbf{x} \in [-5.0, 5.0]^n$
* **Mínimo Global**: $\mathbf{x}^* = (1, 1, \dots, 1)$, con $f(\mathbf{x}^*) = 0.0$
* **Propiedades**: No convexa, altamente mal condicionada; el gradiente desciende rápidamente hacia el fondo de una parábola en forma de banana pero avanza con lentitud extrema hacia el mínimo global.

### 2. Función de Rastrigin (Altamente Multimodal)
$$f(\mathbf{x}) = 10n + \sum_{i=1}^n \left[ x_i^2 - 10\cos(2\pi x_i) \right]$$
* **Dominio**: $\mathbf{x} \in [-5.12, 5.12]^n$
* **Mínimo Global**: $\mathbf{x}^* = (0, 0, \dots, 0)$, con $f(\mathbf{x}^*) = 0.0$
* **Propiedades**: Fuertemente no convexa y multimodal, con una gran densidad de mínimos locales dispuestos regularmente que atrapan a métodos basados en gradiente local.

---

## 1.2 Regla de Comparación Justa y Equivalencia de Evaluaciones

Para comparar de forma objetiva métodos deterministas de primer orden con metaheurísticas poblacionales de orden cero:

1. **Conteo Separado**: Se registran de forma explícita las evaluaciones de la función objetivo ($N_f$) y las evaluaciones del vector gradiente ($N_{\nabla}$).
2. **Equivalencia Analítica de Costo Computacional**:
   Aproximar numéricamente el vector gradiente en dimensión $n$ mediante diferencias finitas centrales requiere evaluar la función en $2n$ puntos:
   $$\nabla_i f(\mathbf{x}) \approx \frac{f(\mathbf{x} + h\mathbf{e}_i) - f(\mathbf{x} - h\mathbf{e}_i)}{2h} \implies 2n \text{ evaluaciones de } f$$
   Por tanto, la métrica unificada de esfuerzo computacional es:
   $$N_{\text{eval\_equiv}} = N_f + (2n \cdot N_{\nabla})$$
3. **Criterio de Éxito**: Se considera corrida exitosa cuando $|f(\mathbf{x}^*) - f_{\text{óptimo}}| \le 10^{-2}$.

---

## 1.3 Resultados Estadísticos (30 Corridas Independientes)

Las corridas se ejecutaron con semillas pseudoaleatorias documentadas a partir de la base 2026:

| Función | Dim | Método | Mejor $f$ | Media $f$ | Desv. Estándar | Tasa Éxito (%) | Media Evals $f$ | Media Evals $\nabla$ | Media Evals Equiv |
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

## 1.4 Discusión: Descenso por Gradiente vs Métodos Heurísticos

1. **Rendimiento en Topologías Convexa-Estrechas (Rosenbrock)**:
   - El **Descenso por Gradiente (GD)** sufre el clásico efecto de zig-zag debido al mal condicionamiento de la matriz Hessiana en el valle de Rosenbrock.
   - **Evolución Diferencial (DE)** superó con creces a todos los demás métodos, logrando precisiones de $10^{-5}$ en 2D gracias a que las diferencias vectoriales ($\mathbf{x}_{r1} - \mathbf{x}_{r2}$) se alinean naturalmente con la orientación del valle curvado.
2. **Rendimiento en Topologías Multimodales (Rastrigin)**:
   - El **Descenso por Gradiente** fracasó en el 100% de las ocasiones para alcanzar el óptimo global, quedando atrapado en los primeros mínimos locales colindantes ($f \approx 9,95$ o superior).
   - Los métodos heurísticos poblacionales (**PSO**, **EA**, **DE**) mostraron una gran capacidad exploratoria para escapar de atractores locales mediante dispersión del enjambre y operadores de cruce. **PSO** demostró la mayor consistencia en Rastrigin 2D.
3. **Eficiencia en Evaluaciones**:
   - Aunque GD requiere pocas iteraciones, cada paso consume $2n$ evaluaciones equivalentes. Por tanto, para obtener resultados subóptimos, GD consumió más evaluaciones equivalentes que los métodos heurísticos, los cuales lograron aproximaciones mucho más cercanas al óptimo global con menos evaluaciones computacionales netas.

---

# PARTE 2: OPTIMIZACIÓN COMBINATORIA (TSP 47 CAPITALES)

## 2.1 Formulación y Parámetros

El costo entre cada par de capitales $(i, j)$ es:
$$\text{Costo}_{ij} = (\text{Valor Hora} \times \text{Tiempo}_{ij}) + \text{Peajes}_{ij} + \text{Combustible}_{ij}$$

* **Valor de la Hora del Vendedor**:
  Según el **Real Decreto 126/2026, de 18 de febrero**, el Salario Mínimo Interprofesional (SMI 2026) se fija en **40,70 €/día**. En una jornada laboral continua legal de **8 horas diarias**:
  $$\text{Valor Hora} = \frac{40{,}70\text{ \euro}}{8\text{ h}} = 5{,}0875 \approx 5{,}09\text{ \euro/h}$$
* **Semilla Aleatoria de Control**: **42** (`semilla = 42`) fijada en todas las comparativas.
* **Vehículo**: Toyota Corolla 140H e-CVT 1.8 (Gasolina 95 E5, consumo 4,5 L/100 km, precio combustible 1,785 €/L oficial MITECO).

---

## 2.2 Comparativa Cuantitativa: Colonia de Hormigas (ACO) con Or-opt vs sin Or-opt

Configuración óptima: `hormigas=20, iteraciones=150, alfa=1.0, beta=3.0, evaporacion=0.4, semilla=42`.

| Métrica | ACO con Or-opt | ACO sin Or-opt (Misma Config.) | ACO sin Or-opt (Alto Presupuesto: 50h x 300it) | Impacto de la Búsqueda Local Or-opt |
| :--- | :---: | :---: | :---: | :---: |
| **Costo Total (€)** | **845,09 €** | 875,41 € | 867,51 € | **-30,32 € (-3,46% ahorro)** |
| **Distancia (km)** | **6.307,79 km** | 6.517,53 km | 6.499,57 km | **-209,74 km menos** |
| **Tiempo Conducción** | **66,49 h** (66h 29m) | 69,13 h (69h 08m) | 67,86 h (67h 51m) | **-2h 39m ahorradas** |
| **Costo Tiempo (€)** | **338,42 €** | 351,89 € | 345,40 € | **-13,47 € en tiempo laboral** |
| **Combustible (€)** | **506,67 €** | 523,52 € | 522,11 € | **-16,85 € combustible** |
| **Peajes (€)** | **0,00 €** | 0,00 € | 0,00 € | Red gratuita de alta capacidad |
| **Iteración Convergencia**| **Iteración 20** | Iteración 85 | Iteración 31 | **Convergencia 4 veces más rápida** |
| **Tiempo Cómputo (s)** | **4,77 s** | 3,97 s | 19,33 s | Óptimo balance calidad/velocidad |

### Algoritmo Genético (DEAP) para el TSP
Adicionalmente, se integró el Algoritmo Genético con cruce de orden (OX) y mutación de permutación:
* **Resultado GA (población=100, gen=1000)**: Coste de **1.897,90 €** (14.716,3 km).
* **Conclusión**: ACO modela con mucha mayor fidelidad problemas de grafos dirigidos asimétricos con heurística $1/\text{coste}$, mientras que GA requiere operadores de reparación o búsqueda local híbrida para evitar entrecruzamientos de aristas a gran escala.

---

## 2.3 Secuencia del Tour Óptimo Peninsular (845,09 €)

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

# GUÍA DE EJECUCIÓN RÁPIDA

### 1. Demostración en Vivo para Sustentación (<30 segundos)
Ejecuta la corrida rápida de Parte 1 y Parte 2, genera estadísticas y animaciones GIF:
```bash
python main.py --demo
```

### 2. Ejecutar Módulos por Separado
* **Solo Parte 1 (Optimización Numérica - 30 corridas y reporte)**:
  ```bash
  python main.py --parte 1
  # O directamente desde su directorio:
  python parte_1/main.py
  ```
* **Solo Parte 2 (TSP 47 Capitales - ACO)**:
  ```bash
  python main.py --parte 2
  # O directamente desde su directorio:
  python parte_2/main.py
  ```
* **Comparativa Con Or-opt vs Sin Or-opt en Parte 2**:
  ```bash
  python parte_2/main.py --comparar-oropt
  ```
* **Algoritmo Genético en Parte 2**:
  ```bash
  python parte_2/main.py --algoritmo ga
  ```

---

# BIBLIOGRAFÍA (Formato APA 7.ª edición)

* Dorigo, M., & Stützle, T. (2004). *Ant colony optimization*. MIT Press. https://doi.org/10.7551/mitpress/1290.001.0001
* Eberhart, R., & Kennedy, J. (1995). A new optimizer using particle swarm theory. En *MHS'95. Proceedings of the Sixth International Symposium on Micro Machine and Human Science* (pp. 39–43). IEEE. https://doi.org/10.1109/MHS.1995.494215
* España. Ministerio de Trabajo y Economía Social. (2026). *Real Decreto 126/2026, de 18 de febrero, por el que se fija el salario mínimo interprofesional para 2026*. Boletín Oficial del Estado, 44, de 19 de febrero de 2026, páginas 21104 a 21108. https://www.boe.es/eli/es/rd/2026/02/18/126
* Fortin, F.-A., De Rainville, F.-M., Gardner, M.-A., Parizeau, M., & Gagné, C. (2012). DEAP: Evolutionary algorithms made easy. *Journal of Machine Learning Research*, 13(70), 2171–2175.
* Nocedal, J., & Wright, S. J. (2006). *Numerical optimization* (2.ª ed.). Springer. https://doi.org/10.1007/978-0-387-40065-5
* Or, I. (1976). *Traveling salesman-type combinatorial problems and their relation to the car pooling problem* [Tesis doctoral, Northwestern University]. ProQuest Dissertations Publishing.
* Storn, R., & Price, K. (1997). Differential evolution–a simple and efficient heuristic for global optimization over continuous spaces. *Journal of Global Optimization*, 11(4), 341–359. https://doi.org/10.1023/A:1008202821328
