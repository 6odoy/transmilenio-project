# TransMilenio — Análisis Operativo y Simulación de Demanda

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Pipeline de datos, dashboard operativo e inferencia causal sobre el sistema de transporte masivo TransMilenio de Bogotá. Analiza el flujo de validaciones de 152 estaciones y 12 líneas troncales durante 2025, y proyecta la demanda de nuevas troncales mediante métodos de inferencia causal.

---

## Qué hace este proyecto

1. **Ingesta y procesamiento** — descarga los registros de validaciones desde el almacenamiento oficial de TransMilenio S.A., los limpia y los persiste como Parquet.
2. **Dashboard operativo** — visualización interactiva (Dash + Plotly) con KPIs en tiempo real, mapa de estaciones y tabla de métricas por troncal.
3. **Inferencia causal** — estimación del efecto causal de clima, festivos, eventos masivos y otras perturbaciones sobre la demanda; y proyección de demanda para la futura Troncal Avenida 68 mediante control sintético.

---

## Requisitos

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) o Anaconda

---

## Instalación

```bash
git clone https://github.com/6odoy/transmilenio-project.git
cd transmilenio-project

conda env create -f environment.yml
conda activate transmilenio-project
```

Para actualizar el entorno después de cambios en `environment.yml`:

```bash
make requirements
```

---

## Uso

### 1. Ejecutar el pipeline de datos

Descarga los ZIPs de validaciones 2025, extrae los CSV, limpia y genera los Parquet procesados:

```bash
python -m transmilenio_project.dataset
```

Los archivos se guardan en `data/processed/parquet/`.

### 2. Generar artefactos del dashboard (KPIs, tabla, colores del mapa)

```bash
python transmilenio_project/dashboard/generar_kpis.py
```

Genera los JSON en `transmilenio_project/dashboard/models/artefactos/`.

### 3. Levantar el dashboard local

```bash
cd transmilenio_project/dashboard
python app.py
```

Abre [http://localhost:8050](http://localhost:8050) en el navegador.

### 4. Correr los tests

```bash
pytest tests/
```

---

## Datos

| Tipo | Descripción | Fuente |
|------|-------------|--------|
| Validaciones troncales | Entradas y salidas por estación y hora, 2025 | TransMilenio S.A. (GCS público) |
| GeoJSON de estaciones | Geometría y atributos de las 152 estaciones | ArcGIS Hub / Datos Abiertos Bogotá |
| Clima Bogotá | Precipitación y temperatura horaria | Open-Meteo API |
| Festivos Colombia | Calendario oficial 2025 | Nager.Date API |
| Siniestros viales | Accidentes georeferenciados | SDM / Datos Abiertos Bogotá |

Para una descripción completa de las fuentes externas disponibles para inferencia causal, ver [`docs/inferencia_causal.md`](docs/inferencia_causal.md).

---

## Estructura del proyecto

```
├── data/
│   ├── external/              <- Datos de fuentes externas (clima, festivos, etc.)
│   ├── interim/               <- Datos intermedios transformados
│   ├── processed/             <- Parquets finales + dimensiones JSON + GeoJSON limpio
│   └── raw/                   <- ZIPs y CSVs originales (inmutables)
│
├── docs/
│   ├── inferencia_causal.md   <- Mapa completo de análisis causales posibles
│   ├── resultados_inferencia_causal.md  <- Resultados de los 6 análisis
│   └── guion_presentacion_inferencia_causal.md  <- Guión de presentación (largo)
│
├── notebooks/                 <- Análisis causales (event study, panel FE, DiD, etc.)
│
├── reports/figures/            <- Figuras generadas por notebooks
│
├── tests/
│   └── test_dataset.py        <- Tests unitarios del pipeline de datos
│
├── transmilenio_project/
│   ├── config.py              <- Rutas y variables de configuración
│   ├── dataset.py             <- Pipeline ETL completo (ingesta → Parquet)
│   └── dashboard/
│       ├── app.py             <- Aplicación Dash principal
│       ├── config.py          <- Temas, idiomas y rutas del dashboard
│       ├── charts_causal.py   <- Gráficas interactivas de inferencia causal
│       ├── generar_kpis.py    <- Cálculo de KPIs y artefactos JSON
│       ├── models/
│       │   └── predictor.py   <- Módulo de predicción de afluencia (Random Forest)
│       └── api/
│           └── index.py       <- Punto de entrada para despliegue en Vercel
│
├── guion.md                   <- Guión de exposición (20 min)
├── environment.yml            <- Dependencias del entorno conda
├── Makefile                   <- Comandos de conveniencia
└── pyproject.toml             <- Configuración del paquete y herramientas
```

---

## Comandos disponibles

```bash
make create_environment   # Crear el entorno conda
make requirements         # Actualizar dependencias
make data                 # Ejecutar el pipeline de datos
make lint                 # Verificar estilo con ruff
make format               # Formatear código con ruff
```

---

## Dashboard

El dashboard tiene dos páginas:

- **Indicadores Generales** — KPIs del sistema (afluencia diaria, troncal de mayor demanda, hora pico, estación más concurrida), mapa interactivo de estaciones coloreado por línea, y tabla de métricas por troncal.
- **Inferencia Causal y Simulador** — siete pestañas con los resultados interactivos de cada análisis:
  1. Festivos (event study)
  2. Clima (panel FE)
  3. Ciclovía (DiD)
  4. Conciertos Campín (event study)
  5. Combustible (serie temporal)
  6. Control Sintético (Ciudad Bolívar)
  7. Simulador RF (predicción de demanda con Random Forest)

Soporta español, inglés e italiano, y tema claro/oscuro.

---

## Contexto académico

Proyecto desarrollado en el marco del **2026 Urban Data Science Summer School** celebrado en Pieve Tesino, Italia
