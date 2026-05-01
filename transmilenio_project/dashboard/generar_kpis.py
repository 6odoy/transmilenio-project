import os
import json
import tempfile
import polars as pl
from loguru import logger

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "parquet")
DIM_LINEA_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "dim_linea.json")
GEOJSON_FILE_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "stations_clean.geojson")
ARTEFACTOS_DIR = os.path.join(BASE_DIR, "models", "artefactos")
KPI_FILE = os.path.join(ARTEFACTOS_DIR, "kpis.json")
TABLA_FILE = os.path.join(ARTEFACTOS_DIR, "tabla_troncales.json")


def _write_json_atomic(data, path: str) -> None:
    """Escribe JSON de forma atómica: primero a un temporal, luego rename."""
    dir_ = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        os.replace(tmp_path, path)
    except Exception:
        os.unlink(tmp_path)
        raise


def generate_kpis():
    os.makedirs(ARTEFACTOS_DIR, exist_ok=True)

    logger.info("Iniciando escaneo de los datos (archivos .parquet)...")
    parquet_pattern = os.path.join(DATA_DIR, "*.parquet")
    df = pl.scan_parquet(parquet_pattern)

    # ── KPI 1: Afluencia Diaria Promedio ──────────────────────────────────────
    logger.info("Calculando KPI 1: Afluencia Diaria Promedio...")
    totales_diarios = df.group_by("fecha").agg(pl.col("total").sum())
    totales_collected = totales_diarios.collect()

    if totales_collected.is_empty():
        raise RuntimeError(f"No se encontraron datos en {parquet_pattern}. Ejecuta el pipeline de datos primero.")

    promedio_diario = totales_collected.select(pl.col("total").mean())[0, 0]
    kpi1_val = f"{int(promedio_diario):,}".replace(",", ".")

    # ── KPI 2: Troncal de Mayor Demanda ───────────────────────────────────────
    logger.info("Calculando KPI 2: Troncal de Mayor Demanda...")
    demand_por_linea = (
        df.group_by("codigo_linea")
        .agg(pl.col("total").sum())
        .sort("total", descending=True)
        .limit(1)
        .collect()
    )
    codigo_linea_top = str(demand_por_linea["codigo_linea"][0])

    try:
        with open(DIM_LINEA_FILE, "r", encoding="utf-8") as f:
            dim_linea = json.load(f)
    except FileNotFoundError:
        logger.warning(f"No se encontró {DIM_LINEA_FILE}. Ejecuta el pipeline de datos primero.")
        dim_linea = {}
    except json.JSONDecodeError as e:
        logger.error(f"JSON corrupto en {DIM_LINEA_FILE}: {e}")
        dim_linea = {}

    kpi2_val = dim_linea.get(codigo_linea_top, f"Línea {codigo_linea_top}")

    # ── KPI 3: Hora Pico del Sistema ──────────────────────────────────────────
    logger.info("Calculando KPI 3: Hora Pico del Sistema...")
    hora_pico_df = (
        df.with_columns(pl.col("hora").dt.hour().alias("hora_del_dia"))
        .group_by("hora_del_dia")
        .agg(pl.col("total").sum())
        .sort("total", descending=True)
        .limit(1)
        .collect()
    )
    hora_int = hora_pico_df["hora_del_dia"][0]
    kpi3_val = f"{hora_int:02d}:00 - {(hora_int + 1) % 24:02d}:00"

    # ── Dimensión de estaciones ───────────────────────────────────────────────
    dim_estacion = {
        "10000": "Portal 20 de Julio",
        "40000": "Tunal (TransMiCable)",
        "40001": "Juan Pablo II",
        "40002": "Manitas",
        "40003": "Mirador del Paraíso"
    }

    if os.path.exists(GEOJSON_FILE_PATH):
        try:
            with open(GEOJSON_FILE_PATH, "r", encoding="utf-8") as f:
                geojson_data = json.load(f)
            for feat in geojson_data.get("features", []):
                props = feat.get("properties", {})
                nodo = str(props.get("codigo_nodo_estacion", ""))
                dim_estacion[nodo] = props.get("nombre_estacion", "Estación")
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"No se pudo leer GeoJSON de estaciones: {e}")

    # ── KPI 4: Estación Más Concurrida ────────────────────────────────────────
    logger.info("Calculando KPI 4: Estación Más Concurrida...")
    demand_por_estacion = (
        df.group_by("codigo_estacion")
        .agg(pl.col("total").sum())
        .sort("total", descending=True)
        .limit(1)
        .collect()
    )
    codigo_estacion_top = str(demand_por_estacion["codigo_estacion"][0])
    kpi4_val = dim_estacion.get(codigo_estacion_top, f"Estación {codigo_estacion_top}")

    kpis = {
        "kpi1_val": kpi1_val,
        "kpi2_val": kpi2_val,
        "kpi3_val": kpi3_val,
        "kpi4_val": kpi4_val,
        "trend1": "text-success",
        "trend2": "text-success",
        "trend3": "text-warning",
        "trend4": "text-success"
    }

    _write_json_atomic(kpis, KPI_FILE)
    logger.success(f"KPIs guardados en {KPI_FILE}")
    logger.info(f"  Afluencia Diaria Promedio: {kpi1_val}")
    logger.info(f"  Troncal de Mayor Demanda: {kpi2_val}")
    logger.info(f"  Hora Pico del Sistema: {kpi3_val}")
    logger.info(f"  Estación Más Concurrida: {kpi4_val}")

    # ── Métricas por Línea Troncal ────────────────────────────────────────────
    logger.info("Calculando Métricas por Línea Troncal...")

    totales_por_linea = (
        df.group_by("codigo_linea")
        .agg(pl.col("total").sum().alias("total_linea"))
        .sort("total_linea", descending=True)
        .collect()
    )

    estacion_principal_por_linea = (
        df.group_by(["codigo_linea", "codigo_estacion"])
        .agg(pl.col("total").sum())
        .sort(["codigo_linea", "total"], descending=[False, True])
        .group_by("codigo_linea", maintain_order=True)
        .first()
        .collect()
    )

    tabla_df = totales_por_linea.join(estacion_principal_por_linea, on="codigo_linea", how="left")

    tabla_troncales = []
    for row in tabla_df.to_dicts():
        c_linea = str(row["codigo_linea"])
        c_estacion = str(row["codigo_estacion"])
        t_linea = row["total_linea"]

        nombre_linea = dim_linea.get(c_linea, f"Línea {c_linea}")
        nombre_estacion = dim_estacion.get(c_estacion, f"Estación {c_estacion}")
        entradas_fmt = f"{int(t_linea):,}".replace(",", ".")

        tabla_troncales.append({
            "Linea": nombre_linea,
            "Top_Estacion": nombre_estacion,
            "Entradas": entradas_fmt
        })

    _write_json_atomic(tabla_troncales, TABLA_FILE)
    logger.success(f"Tabla de troncales guardada en {TABLA_FILE} ({len(tabla_troncales)} filas)")

    # ── Colores por Estación ──────────────────────────────────────────────────
    logger.info("Generando colores por estación para el mapa...")
    LINE_COLORS = {
        "11": "#CDA53E",
        "12": "#009B8E",
        "30": "#009EE0",
        "31": "#C1001F",
        "32": "#FFD100",
        "33": "#8DC63F",
        "34": "#F37021",
        "35": "#7E3F98",
        "36": "#002855",
        "37": "#E81E6D",
        "38": "#7B523A",
        "40": "#CC5500"
    }

    LINEA_ESTACIONES_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "linea_estaciones.json")
    linea_estaciones = {}
    if os.path.exists(LINEA_ESTACIONES_FILE):
        try:
            with open(LINEA_ESTACIONES_FILE, "r", encoding="utf-8") as f:
                linea_estaciones = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"No se pudo leer {LINEA_ESTACIONES_FILE}: {e}")

    estacion_linea_cruce = {}
    for c_lin, lista_ests in linea_estaciones.items():
        for c_est in lista_ests:
            if str(c_est) not in estacion_linea_cruce:
                estacion_linea_cruce[str(c_est)] = str(c_lin)

    estacion_color = {}
    for c_est in dim_estacion.keys():
        c_lin = estacion_linea_cruce.get(str(c_est), "31")
        color = LINE_COLORS.get(c_lin, "#C1001F")
        estacion_color[str(c_est)] = {"linea": c_lin, "color": color}

    COLOR_ESTACION_FILE = os.path.join(ARTEFACTOS_DIR, "estacion_color.json")
    _write_json_atomic(estacion_color, COLOR_ESTACION_FILE)
    logger.success(f"Mapa de colores guardado en {COLOR_ESTACION_FILE}")


if __name__ == "__main__":
    generate_kpis()
