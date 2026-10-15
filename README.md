# Trabajo 01: Optimización Numérica y Combinatoria (Algoritmos Bioinspirados)
**Curso**: Redes Neuronales y Algoritmos Bioinspirados (Semestre 2026-02)  
**Universidad Nacional de Colombia — Facultad de Minas, Sede Medellín**  

**Equipo Investigador**:
* Alejandro Agudelo Anaya — CC: 1017271139
* David Agudelo Arboleda — CC: 1000899797
* Santiago Álvarez Martínez — CC: 1022142463
* Daniel Felipe López Zapata — CC: 1045046468
* Sara Osorio Piñerez — CC: 1001370113

**Docente Tutor**: Prof. Juan David Ospina Arango (`jdospina@unal.edu.co`)  
**Repositorio GitHub**: [https://github.com/DavidAgr5421/Trabajo01_Redes.git](https://github.com/DavidAgr5421/Trabajo01_Redes.git)  
**Sitio Web en GitHub Pages**: [https://DavidAgr5421.github.io/Trabajo01_Redes/](https://DavidAgr5421.github.io/Trabajo01_Redes/)  
**Reporte Técnico de Blog**: [REPORTE_BLOG.md](REPORTE_BLOG.md) | **Entrada Web**: [index.html](index.html)  

---

## 🚀 Guía de Ejecución Rápida en Un Solo Comando

El proyecto cuenta con un CLI unificado (`main.py`) que permite ejecutar **cualquier ejercicio, algoritmo o prueba de manera aislada o integral con un único comando**:

### 1. Comandos Principales
```bash
# Ejecución integral completa (Parte 1 + Parte 2 + Figuras)
python main.py

# Demostración rápida para sustentación en vivo (< 30 segundos)
python main.py --demo

# Menú interactivo por consola (permite elegir cualquier ejercicio con un número)
python main.py --menu
```

### 2. Ejecutar Cada Ejercicio / Algoritmo por Separado
```bash
# --- PARTE 1: OPTIMIZACIÓN NUMÉRICA CONTINUA ---
python main.py -e gd            # Descenso por Gradiente (Rosenbrock y Rastrigin)
python main.py -e pso           # Optimización por Enjambre de Partículas (PSO)
python main.py -e ea            # Algoritmo Evolutivo Continuo (EA)
python main.py -e de            # Evolución Diferencial (DE)
python main.py -e parte1        # Protocolo estadístico completo (30 corridas independientes)

# --- PARTE 2: OPTIMIZACIÓN COMBINATORIA (TSP 47 CAPITALES) ---
python main.py -e aco           # Colonia de Hormigas con Or-opt (Ruta óptima peninsular 845,09 €)
python main.py -e ga            # Algoritmo Genético DEAP (TSP permutacional)
python main.py -e oropt         # Comparativa directa Con vs Sin búsqueda local Or-opt
python main.py -e sensibilidad  # Estudio de sensibilidad salario horario del vendedor [0 - 100 €/h]
python main.py -e parte2        # Runner completo de Parte 2

# --- VISUALIZACIONES Y RECURSOS MULTIMEDIA ---
python main.py -e graficas      # Genera todas las figuras 3D, de barras y convergencia
python main.py -e gifs          # Genera las animaciones GIF de Parte 1 y Parte 2
```

---

## 📁 Estructura del Repositorio

```
Entrega1/
├── index.html                    # Entrada de blog académica completa para GitHub Pages (KaTeX, responsive)
├── _config.yml                   # Configuración oficial de GitHub Pages / Jekyll
├── _posts/                       # Entrada de blog en formato post de Jekyll
│   └── 2026-10-13-optimizacion-redes-neuronales.md
├── REPORTE_BLOG.md               # Reporte técnico exhaustivo en Markdown con auditoría completa
├── README.md                     # Guía de usuario y ejecución en un solo comando
├── main.py                       # CLI global y orquestador maestro de ejercicios
├── generar_graficas.py           # Generador automático de figuras 3D, barras y convergencia
├── configuracion.json            # Parámetros hiperparamétricos centrales del proyecto
├── config.py                     # Cargador central con soporte de perfiles y demo
├── requirements.txt              # Dependencias fijadas (numpy, pandas, matplotlib, deap, pillow)
│
├── assets/                       # Recursos visuales y multimedia para GitHub Pages
│   └── img/
│       ├── figura1_superficies_3d.png
│       ├── figura4_tasa_exito.png
│       ├── figura5_mediana_valor_final.png
│       ├── figura6_costo_computacional.png
│       ├── figura7_convergencia_aco.png
│       ├── animacion_gd_rosenbrock_2d.gif
│       ├── animacion_pso_rosenbrock_2d.gif
│       ├── evolucion_aco_con_oropt.gif
│       ├── evolucion_ga.gif
│       ├── mapa_aco_con_oropt.png
│       └── mapa_ga.png
│
├── parte_1/                      # PARTE 1: OPTIMIZACIÓN NUMÉRICA CONTINUA
│   ├── main.py                   # Runner específico de Parte 1 con soporte de métodos individuales
│   ├── funciones.py              # Definición analítica de Rosenbrock, Rastrigin y gradientes
│   ├── gradienteDecendente.py    # Descenso por Gradiente con detección de divergencia
│   ├── PSO.py                    # Algoritmo PSO Global Best
│   ├── algoritmo_evolutivo.py    # Algoritmo Evolutivo Continuo con cruce y mutación adaptativa
│   ├── evolucion_diferencial.py  # Evolución Diferencial canónica DE/rand/1/bin
│   ├── comparacion_y_gifs_parte1.py # Script de experimentación Colab de 30 corridas
│   └── resultados/               # CSVs de estadísticas, JSONs y animaciones
│
└── parte_2/                      # PARTE 2: OPTIMIZACIÓN COMBINATORIA (TSP ESPAÑA)
    ├── main.py                   # Runner específico de Parte 2
    ├── colonia_hormigas.py       # Algoritmo ACO elitista y búsqueda local Or-opt
    ├── algoritmo_genetico.py     # Algoritmo Genético en DEAP (cruce OX, mutación shuffle)
    ├── sensibilidad.py           # Módulo de barrido del valor horario del vendedor
    ├── experimentos_aco.py       # Barrido sistemático OFAT y comparativas
    ├── costes_comunes.py         # Matrices dirigidas de costo, tiempo, distancia y peajes
    ├── generar_gif.py            # Generador de animaciones de ruta sobre el mapa de España
    ├── validacion.py             # Verificador de integridad topológica del grafo
    ├── datos_rutas/              # Bases de datos viales y fuentes oficiales (ciudades.csv, rutas.csv)
    └── resultados/               # JSONs de rutas, CSVs de convergencia y GIFs
```

---

## 🔬 Resumen de Resultados Principales

### Parte 1: Optimización Numérica
* **Regla de Comparación Justa**: $E_{\text{equiv}} = E_f + 2d \times E_{\nabla}$, fundamentada en el costo de aproximar el gradiente analítico mediante diferencias finitas centrales.
* **Rastrigin**: Descenso por Gradiente falló en el 100% de los casos (0% de éxito en 2D y 3D) por atrapamiento en mínimos locales. PSO (100% en 2D, **93,33% en 3D**) y Evolución Diferencial (100% en 2D, **96,67% en 3D**) superaron ampliamente la barrera multimodal.
* **Rosenbrock**: GD converge en 2D con $\alpha=0,001$ requiriendo más de 2.200 pasos, pero sufre divergencias en 3D por mal condicionamiento del valle. La Evolución Diferencial demostró el mejor comportamiento local ($f_{\text{mediana}} \approx 3,12 \times 10^{-10}$) debido a la alineación intrínseca de los vectores de mutación a lo largo del lecho parabólico.

### Parte 2: Optimización Combinatoria (TSP 47 Capitales)
* **Función de Costo Real**: Integra salario mínimo interprofesional (**5,09 €/h**, RD 126/2026), combustible de un **Toyota Corolla híbrido (4,5 L/100 km)** y peajes (0,00 € por autovías de alta capacidad).
* **Impacto de la Búsqueda Local Or-opt**:
  * **Costo Óptimo**: **845,09 €** (vs 875,41 € de ACO puro), representando un **ahorro de 30,32 € (-3,46%)**.
  * **Distancia y Tiempo**: **6.307,79 km** (-209,74 km) y **66,49 h** (-2h 39m al volante).
  * **Convergencia**: Se alcanza el óptimo en la **iteración 20** (4 veces más rápido que ACO sin Or-opt).
* **Sensibilidad Salarial**: La ruta óptima es invariante entre 4,5 y 12 €/h. A partir de 20 €/h, el tiempo de conducción se reduce a 65,65 h priorizando autovías de 120 km/h.

---

## 👥 Auditoría de IA y Cacería de Alucinaciones
El proyecto documenta formalmente la asistencia de LLMs y expone cuatro casos de alucinaciones refutadas con rigurosidad:
1. **Rutas fijas inventadas por Pathlib**: Corregidas con resolución dinámica anclada.
2. **Coordenadas falsas de OSRM**: Rechazadas y reemplazadas por coordenadas oficiales del IGN.
3. **Re-consulta redundante a Geoapify**: Purgada al demostrar que las matrices ya estaban calculadas.
4. **Porcentajes de éxito falsos (86,67% y 96,67%)**: Rectificados con los datos empíricos reales (93,33% para PSO en Rastrigin 3D y 26,67% para DE en Rosenbrock 3D).
