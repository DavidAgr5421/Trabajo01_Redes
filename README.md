# Optimización de Rutas para el Problema del Vendedor Viajero (TSP)
## 47 Capitales de Provincia de la España Peninsular mediante Colonia de Hormigas (ACO)

---

## 1. Formulación del Problema

Un vendedor comercial debe recorrer las **47 capitales de provincia de la España peninsular** en un **circuito cerrado** (regresando a la capital de partida). 
Se excluyen Palma de Mallorca, Las Palmas de Gran Canaria y Santa Cruz de Tenerife (territorios insulares), así como Ceuta y Melilla (ciudades autónomas no accesibles por red de carreteras peninsular).

El costo de desplazamiento entre dos ciudades $i$ y $j$ se define como:

$$\text{Costo}_{ij} = (\text{Valor Hora} \times \text{Tiempo de Viaje}_{ij}) + \text{Peajes}_{ij} + \text{Combustible}_{ij}$$

### Parámetros Financieros y Operativos
1. **Valor de la Hora del Vendedor**:
   - Según el **Real Decreto 126/2026, de 18 de febrero** (SMI 2026), el salario mínimo se fija en **40,70 €/día** o 1.221 €/mes (art. 1).
   - Para una jornada laboral continua de **8 horas diarias**, el valor por hora corresponde a:
     $$\text{Valor Hora} = \frac{40{,}70\text{ \euro}}{8\text{ h}} = 5{,}0875 \approx 5{,}09\text{ \euro/h}$$
2. **Semilla Aleatoria de Control**:
   - Fijada estrictamente en **42** (`semilla = 42`) para garantizar reproducibilidad exacta entre configuraciones.
3. **Vehículo y Combustible**:
   - Modelo: **Toyota Corolla 140H e-CVT 1.8** (1798 cc).
   - Consumo mixto WLTP: **4,5 L / 100 km**.
   - Precio de combustible: **1,785 €/L** (Gasolina 95 E5, media nacional MITECO a fecha de referencia).

---

## 2. Comparativa de Resultados: ACO con Or-opt vs sin Or-opt

Ambas variantes fueron ejecutadas con la **misma configuración óptima** y con la **semilla 42 fijada**:
- **Hormigas**: 20
- **Iteraciones**: 150
- **Alfa ($\alpha$)**: 1.0 (peso de feromona)
- **Beta ($\beta$)**: 3.0 (peso de heurística $1/\text{coste}$)
- **Evaporación ($\rho$)**: 0.4
- **Semilla**: 42

| Métrica | ACO con Or-opt | ACO sin Or-opt (Misma Config.) | ACO sin Or-opt (Alto Presupuesto)* | Diferencia / Ahorro con Or-opt |
| :--- | :---: | :---: | :---: | :---: |
| **Costo Total (€)** | **845,09 €** | 875,41 € | 867,51 € | **-30,32 € (-3,46%)** |
| **Distancia Total (km)** | **6.307,79 km** | 6.517,53 km | 6.499,57 km | **-209,74 km** |
| **Tiempo de Conducción (h)** | **66,49 h** (66h 29m) | 69,13 h (69h 08m) | 67,86 h (67h 51m) | **-2,65 h** |
| **Costo Tiempo Vendedor (€)**| **338,42 €** | 351,89 € | 345,40 € | **-13,47 €** |
| **Costo Combustible (€)** | **506,67 €** | 523,52 € | 522,11 € | **-16,85 €** |
| **Costo Peajes (€)** | **0,00 €** | 0,00 € | 0,00 € | 0,00 € (vías de alta capacidad gratuitas) |
| **Iteración de Convergencia**| **20** | 85 | 31 | **Convergencia 4x más rápida** |
| **Tiempo de Cómputo (s)** | **4,77 s** | 3,97 s | 19,81 s | Óptimo balance calidad/tiempo |

*\* Alto Presupuesto: 50 hormigas y 300 iteraciones (cuadruplica el tiempo de cómputo sin alcanzar la calidad de Or-opt).*

### Análisis de la Comparación
1. **Calidad de la Solución**: La búsqueda local **Or-opt** (reubicación de cadenas de 1 a 3 ciudades contiguas) elimina cruces topológicos y rodeos que las hormigas puras tardan muchas más iteraciones en podar o en los que quedan estancadas.
2. **Eficiencia**: Con Or-opt, el algoritmo alcanza su valor mínimo en la iteración **20**, ahorrando **209,74 km** de carretera y **2 horas y 39 minutos** de tiempo de vendedor.
3. **Alto Presupuesto sin Or-opt**: Aún elevando el presupuesto a 50 hormigas y 300 iteraciones (19,81 s), el ACO puro se queda en **867,51 €**, lo que demuestra que la combinación de metaheurística poblacional (ACO) + búsqueda local (Or-opt) es estrictamente superior.

---

## 3. Barrido de Configuraciones ACO (Semilla 42 Fija)

Se exploraron sistemáticamente los hiperparámetros manteniendo la semilla **42** constante:

- **Hormigas evaluadas**: 10, 15, 20, 30, 50, 80
- **Iteraciones evaluadas**: 50, 100, 150, 200, 300
- **Alfa ($\alpha$)**: 0.5, 1.0, 1.5, 2.0, 2.5
- **Beta ($\beta$)**: 1.0, 2.0, 3.0, 4.0, 5.0, 6.0
- **Evaporación ($\rho$)**: 0.05, 0.10, 0.20, 0.30, 0.40, 0.50

### Top 5 de Configuraciones
1. **Ganadora**: `hormigas=20, iteraciones=150, alfa=1.0, beta=3.0, evaporacion=0.4` $\rightarrow$ **845,09 €** (4,76 s)
2. `hormigas=80, iteraciones=150, alfa=1.0, beta=3.0, evaporacion=0.2` $\rightarrow$ **845,09 €** (20,66 s)
3. `hormigas=20, iteraciones=150, alfa=1.0, beta=1.0, evaporacion=0.2` $\rightarrow$ **845,58 €** (4,66 s)
4. `hormigas=20, iteraciones=150, alfa=1.0, beta=4.0, evaporacion=0.2` $\rightarrow$ **845,58 €** (4,84 s)
5. `hormigas=20, iteraciones=300, alfa=1.0, beta=3.0, evaporacion=0.2` $\rightarrow$ **845,63 €** (9,31 s)

La configuración seleccionada logra el menor costo con solo 20 hormigas en ~4,7 s, logrando la máxima eficiencia computacional.

---

## 4. Secuencia Óptima de Recorrido (47 Capitales)

La ruta óptima describe un anillo perimetral y radial sin entrecruzamientos:

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

## 5. Estructura Limpia del Proyecto

Todos los archivos redundantes o huérfanos han sido organizados y compactados:

```
Entrega1/
│
├── config.py                     # Carga de configuracion central con soporte CLI
├── configuracion.json            # Parametros principales (valor_hora=5.09, semilla=42, aco)
├── colonia_hormigas.py           # Algoritmo ACO principal (Ant System + Elitismo + Or-opt)
├── costes_comunes.py             # Carga y calculo de matrices 47x47 de tiempo, distancia y costo
├── experimentos_aco.py           # Script de barrido de parametros y comparativa con/sin Or-opt
├── generar_gif.py                # Generacion de mapas PNG y animaciones GIF de la convergencia
├── main.py                       # CLI unificada para ejecutar cualquier modo con un comando
├── validacion.py                 # Validador de consistencia de la matriz de costos y ciudades
├── requirements.txt              # Dependencias de Python fijadas
├── README.md                     # Documentacion completa del proyecto
│
├── datos_rutas/                  # Datos base y fuentes oficiales
│   ├── ciudades.csv              # Coordenadas geograficas de las 47 capitales
│   ├── fuentes.csv               # Documentacion de fuentes oficiales (MITECO, BOE, DGT)
│   ├── rutas.csv                 # Pares dirigidos base
│   ├── rutas_costes.csv          # Matriz de costos dirigidos (distancia, tiempo, peaje, gas)
│   └── scripts_preparacion/      # Scripts historicos de extraccion (Geoapify e inventario)
│       ├── calcular_rutas.py
│       └── calcular_costes.py
│
└── resultados/                   # Salidas consolidadas
    ├── barrido_aco.csv           # Registro de todas las configuraciones probadas
    ├── ranking_aco.json          # Ranking ordenado de configuraciones
    ├── mejor_configuracion_aco.json # Configuracion optima ganadora
    ├── comparacion_oropt.json    # Resultados estructurados de la comparativa
    ├── comparacion_oropt.csv     # Tabla comparativa con y sin Or-opt
    ├── comparacion_convergencia.png # Grafica comparativa de curvas de convergencia
    ├── mejor_ruta_aco_con_oropt.csv # Ruta detallada paso a paso (con Or-opt)
    ├── mejor_ruta_aco_sin_oropt.csv # Ruta detallada paso a paso (sin Or-opt)
    ├── evolucion_aco_con_oropt.csv  # Convergencia iteracion a iteracion (con Or-opt)
    ├── evolucion_aco_sin_oropt.csv  # Convergencia iteracion a iteracion (sin Or-opt)
    ├── mapa_aco_con_oropt.png       # Mapa de Espana con la ruta optima final
    ├── mapa_aco_sin_oropt.png       # Mapa de Espana con la ruta sin Or-opt
    ├── evolucion_aco_con_oropt.gif  # GIF animado de la convergencia (con Or-opt)
    └── evolucion_aco_sin_oropt.gif  # GIF animado de la convergencia (sin Or-opt)
```

---

## 6. Instrucciones de Ejecución

Todos los scripts utilizan por defecto `valor_hora = 5.09` y `semilla = 42`.

### 1. Validación de Datos
```bash
python validacion.py
```

### 2. Ejecutar la Mejor Ruta con ACO (con Or-opt y generación de GIF/Mapa)
```bash
python main.py
```

### 3. Ejecutar ACO sin Or-opt
```bash
python main.py --sin-or-opt
```

### 4. Demostración Rápida para Sustentación (<10 segundos)
```bash
python main.py --demo
```

### 5. Comparativa Directa Con vs Sin Or-opt
```bash
python experimentos_aco.py --solo-comparar
```

### 6. Ejecutar el Barrido Completo de Hiperparámetros
```bash
python experimentos_aco.py
```
