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
│   └── inferencia_causal.md   <- Mapa completo de análisis causales posibles
│
├── models/                    <- Artefactos de modelos entrenados
│
├── notebooks/                 <- Exploración y prototipado
│
├── tests/
│   └── test_dataset.py        <- Tests unitarios del pipeline de datos
│
├── transmilenio_project/
│   ├── config.py              <- Rutas y variables de configuración
│   ├── dataset.py             <- Pipeline ETL completo (ingesta → Parquet)
│   ├── features.py            <- Ingeniería de features para modelado
│   ├── plots.py               <- Visualizaciones reutilizables
│   ├── modeling/
│   │   ├── train.py           <- Entrenamiento de modelos
│   │   └── predict.py        <- Inferencia con modelos entrenados
│   └── dashboard/
│       ├── app.py             <- Aplicación Dash principal
│       ├── config.py          <- Temas, idiomas y rutas del dashboard
│       ├── generar_kpis.py    <- Cálculo de KPIs y artefactos JSON
│       ├── models/
│       │   └── predictor.py   <- Módulo de predicción de afluencia
│       └── api/
│           └── index.py       <- Punto de entrada para despliegue en Vercel
│
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

El dashboard tiene dos módulos:

- **Indicadores Generales** — KPIs del sistema (afluencia diaria, troncal de mayor demanda, hora pico, estación más concurrida), mapa interactivo de estaciones coloreado por línea, y tabla de métricas por troncal.
- **Simulador de Demanda** — proyección de afluencia para la futura Troncal Avenida 68 basada en inferencia causal y variables operativas.

Soporta español, inglés e italiano, y tema claro/oscuro.

---

## Inferencia Causal

Ver [`docs/inferencia_causal.md`](docs/inferencia_causal.md) para el mapa completo de:
- 8 preguntas causales identificadas (clima, festivos, Ciclovía, eventos masivos, precio combustible, siniestros, calidad del aire, control sintético Av. 68)
- Métodos formales para cada pregunta (DiD, Event Study, IV, Synthetic Control)
- Fuentes de datos externas con URLs y formato de ingesta
- DAG causal del sistema
- Tabla de priorización

---

## Contexto académico

Proyecto desarrollado en el marco del **Reto de Ciudades Sostenibles e Inferencia Causal** — Uniandes Experimental Computing (UEC).
