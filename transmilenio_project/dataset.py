import json
import re
import time
import zipfile
from pathlib import Path
from urllib.error import URLError, HTTPError
from urllib.request import urlopen, urlretrieve

import polars as pl
import geopandas as gpd
from loguru import logger
from tqdm import tqdm
import typer

from transmilenio_project.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

# =======================
# CONSTANTS
# =======================

INDEX_URL = "https://storage.googleapis.com/validaciones_tmsa/Salidas.html"
BASE_URL = "https://storage.googleapis.com/validaciones_tmsa/Salidas/"
ZIP_DIR = RAW_DATA_DIR / "zips"
CSV_DIR = RAW_DATA_DIR / "csv"
PARQUET_DIR = PROCESSED_DATA_DIR / "parquet"
GEOJSON_URL = "https://opendata.arcgis.com/datasets/5365d814bbdd4062a59234eea7d70db7_1.geojson"
GEOJSON_PATH = RAW_DATA_DIR / "stations.geojson"
CLEAN_GEOJSON_PATH = PROCESSED_DATA_DIR / "stations_clean.geojson"

COLUMNS_TO_KEEP = [
    "Fecha_Transaccion",
    "Tiempo",
    "Linea",
    "Estacion",
    "Entradas_E",
    "Salidas_S",
]
GROUP_KEYS = ["Fecha_Transaccion", "Tiempo", "Estacion"]

FINAL_COLUMNS = [
    "timestamp", "fecha", "hora",
    "codigo_linea",
    "codigo_estacion",
    "entradas", "salidas", "total",
]


# ======================================
# 1. INGESTA: descubrimiento y extracción
# ======================================


def ingest_raw_data() -> list[Path]:
    """
    Pipeline completo de extracción:
    - lista archivos remotos (año 2025)
    - descarga cada ZIP
    - extrae los CSV
    - devuelve lista de CSV listos para procesar
    """
    logger.info("=== Iniciando ingesta de datos (2025) ===")

    # Listar fuentes remotas
    zip_names = _list_remote_files()
    if not zip_names:
        logger.warning("No se encontraron archivos ZIP de 2025")
        return []

    all_csvs: list[Path] = []

    # Descargar y extraer
    for name in tqdm(zip_names, desc="Descargando y extrayendo ZIPs"):
        zip_path = _download_file(name)
        csvs = _extract_zip(zip_path)
        all_csvs.extend(csvs)

    logger.success(
        f"Ingesta completa: {len(all_csvs)} CSV extraídos de {len(zip_names)} ZIPs"
    )
    return all_csvs

def download_geojson(force: bool = False) -> Path:
    """
    Descarga el GeoJSON de estaciones troncales de TransMilenio desde
    ArcGIS Hub y lo guarda en RAW_DATA_DIR.
 
    Args:
        force: Si es True, descarga aunque el archivo ya exista.
 
    Returns:
        Path al archivo GeoJSON descargado.
    """
    if GEOJSON_PATH.exists() and not force:
        logger.debug(f"GeoJSON ya existe, saltando descarga: {GEOJSON_PATH.name}")
        return GEOJSON_PATH
 
    GEOJSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Descargando GeoJSON de estaciones troncales → {GEOJSON_PATH}")
    _urlretrieve_with_retry(GEOJSON_URL, GEOJSON_PATH)
    logger.success(f"GeoJSON guardado: {GEOJSON_PATH}")
    return GEOJSON_PATH

def load_stations_geojson(force: bool = False) -> gpd.GeoDataFrame:
    """
    Carga el GeoJSON de estaciones troncales, filtra columnas relevantes
    y guarda una versión limpia en data/processed.
    """
    if CLEAN_GEOJSON_PATH.exists() and not force:
        logger.debug(f"GeoJSON limpio ya existe, saltando: {CLEAN_GEOJSON_PATH.name}")
        return gpd.read_file(CLEAN_GEOJSON_PATH)

    gdf = gpd.read_file(GEOJSON_PATH)
    gdf_clean = gdf[["codigo_nodo_estacion", "nombre_estacion", "geometry"]]

    CLEAN_GEOJSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    gdf_clean.to_file(CLEAN_GEOJSON_PATH, driver="GeoJSON")
    logger.info(f"GeoJSON limpio guardado: {CLEAN_GEOJSON_PATH.name} ({len(gdf_clean)} estaciones)")
    return gdf_clean

# ======================================
# 2. PROCESAMIENTO: limpieza + agregación
# ======================================


def process_raw_data(csv_path: Path, output_path: Path) -> tuple[Path, pl.DataFrame | None, pl.DataFrame | None]:
    """
    Limpia, agrega, transforma y persiste un CSV individual como Parquet.

    Etapas:
      1. Carga y selección de columnas útiles
      2. Tipado de columnas numéricas y creación de timestamp
      3. Agregación por fecha + hora + estación
      4. Extracción de código/nombre de estación y línea
      5. Correcciones de reglas de negocio (Intermedias San Mateo)
      6. Renombrar columnas (entradas, salidas)
      7. Consolidación de variantes de San Mateo
      8. Normalización de texto (titlecase, acentos, Cabecera→Portal)
      9. Selección final y persistencia en Parquet
    """
    if output_path.exists():
        schema = pl.read_parquet_schema(output_path)
        if "nombre_linea" in schema:
            logger.info(f"Convirtiendo parquet existente al nuevo esquema: {output_path.name}")
            df = pl.read_parquet(output_path)
            df_linea_counts = df.select(["codigo_linea", "nombre_linea"]).group_by(["codigo_linea", "nombre_linea"]).len()
            df_estacion_counts = df.select(["codigo_estacion", "nombre_estacion"]).group_by(["codigo_estacion", "nombre_estacion"]).len()
            df.select(FINAL_COLUMNS).write_parquet(output_path)
            return output_path, df_linea_counts, df_estacion_counts, None

        logger.debug(f"Parquet ya existe, saltando: {output_path.name}")
        return output_path, None, None, None

    logger.debug(f"Procesando: {csv_path.name}")

    # 1. Carga y selección de columnas ─────────────────────────────
    df = pl.read_csv(csv_path, try_parse_dates=False, infer_schema_length=5000)

    missing = [c for c in COLUMNS_TO_KEEP if c not in df.columns]
    if missing:
        raise ValueError(f"Columnas faltantes en {csv_path.name}: {missing}. Columnas disponibles: {df.columns}")

    df = df.select(COLUMNS_TO_KEEP)

    # 2. Tipado y timestamp ────────────────────────────────────────
    df = df.with_columns(
        pl.col("Entradas_E").cast(pl.Int64),
        pl.col("Salidas_S").cast(pl.Int64),
        pl.concat_str(
            [pl.col("Fecha_Transaccion"), pl.col("Tiempo")],
            separator=" ",
        )
        .str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False)
        .alias("timestamp"),
    )

    # 3. Agregación por fecha + hora + estación ────────────────────
    df = (
        df.group_by(GROUP_KEYS)
        .agg(
            pl.col("Linea").first(),
            pl.col("Entradas_E").sum(),
            pl.col("Salidas_S").sum(),
            pl.col("timestamp").first(),
        )
        .sort(GROUP_KEYS)
    )

    # 4. Extracción de campos derivados ────────────────────────────
    #    Patrón de Estacion/Linea: "(código)nombre"
    df = df.with_columns(
        # Fechas
        pl.col("timestamp").dt.date().alias("fecha"),
        pl.col("timestamp").dt.time().alias("hora"),
        # Estación
        pl.col("Estacion").str.extract(r"\((\d+)\)", group_index=1)
            .cast(pl.Int64).alias("codigo_estacion"),
        pl.col("Estacion").str.extract(r"\)(.+)", group_index=1)
            .alias("nombre_estacion"),
        # Línea
        pl.col("Linea").str.extract(r"\((\d+)\)", group_index=1)
            .cast(pl.Int64).alias("codigo_linea"),
        pl.col("Linea").str.extract(r"\)(.+)", group_index=1)
            .alias("nombre_linea"),
        # Validaciones
        (pl.col("Entradas_E") + pl.col("Salidas_S")).alias("total"),
    )

    # 5. Correcciones de reglas de negocio ─────────────────────────
    #    "Intermedias San Mateo" de la línea ficticia "Line for Intermedium Gate"
    #    corresponde realmente a la zona G NQS Sur / San Mateo.
    es_intermedias = (
        (pl.col("nombre_linea") == "Line for Intermedium Gate")
    )
    df = df.with_columns(
        pl.when(es_intermedias)
            .then(pl.lit("Zona G NQS Sur"))
            .otherwise(pl.col("nombre_linea"))
            .alias("nombre_linea")
    )

    # Unificar Zona F Calle 13 (39) hacia Zona F Av. Americas (31)
    df = df.with_columns(
        pl.when(pl.col("codigo_linea") == 39)
            .then(pl.lit("Zona F Av. Americas"))
            .otherwise(pl.col("nombre_linea"))
            .alias("nombre_linea"),
        pl.when(pl.col("codigo_linea") == 39)
            .then(pl.lit(31).cast(pl.Int64))
            .otherwise(pl.col("codigo_linea"))
            .alias("codigo_linea")
    )

    # 6. Renombrar columnas ──────────────────────────────────────────
    df = df.rename({
        "Entradas_E": "entradas",
        "Salidas_S": "salidas",
    })

    # 7. Consolidación de variantes ───────────────────────────────
    #    Mapa de códigos de estaciones variantes -> código principal
    variantes_map = {
        40004: 40003,
        50002: 9100,
        50003: 9001,
        50008: 6000,
        57503: 7503,
        59503: 7503,
        9129:  9113,
        9125:  9119,
        9124:  9115,
    }

    variantes_codigos = list(variantes_map.keys())

    # Separar variantes del resto
    variantes = df.filter(pl.col("codigo_estacion").is_in(variantes_codigos))

    # Agregar variantes mapeando al código principal
    variantes_agg = (
        variantes
        .with_columns(
            pl.col("codigo_estacion").replace(variantes_map).alias("codigo_principal")
        )
        .group_by(["timestamp", "codigo_principal"])
        .agg(
            pl.col("entradas").sum(),
            pl.col("salidas").sum(),
            pl.col("total").sum(),
        )
    )

    # Join con los registros principales y sumar
    principales = df.filter(pl.col("codigo_estacion").is_in(list(variantes_map.values())))

    resultado = (
        principales
        .join(
            variantes_agg,
            left_on=["timestamp", "codigo_estacion"],
            right_on=["timestamp", "codigo_principal"],
            how="left",
            suffix="_var"
        )
        .with_columns(
            (pl.col("entradas") + pl.col("entradas_var").fill_null(0)).alias("entradas"),
            (pl.col("salidas") + pl.col("salidas_var").fill_null(0)).alias("salidas"),
            (pl.col("total") + pl.col("total_var").fill_null(0)).alias("total"),
        )
        .drop("entradas_var", "salidas_var", "total_var")
    )

    # Reconstruir el df
    df = df.filter(
        ~pl.col("codigo_estacion").is_in(variantes_codigos) &
        ~pl.col("codigo_estacion").is_in(list(variantes_map.values()))
    ).vstack(resultado)

    # 8. Normalización de texto ────────────────────────────────────
    #    Titlecase, strip, eliminar acentos, y "Cabecera" → "Portal".
    df = (
        df.with_columns(
            pl.col("nombre_linea", "nombre_estacion")
                .str.to_titlecase()
                .str.strip_chars()
                .str.replace_all(r"[áàâä]", "a")
                .str.replace_all(r"[éèêë]", "e")
                .str.replace_all(r"[íìîï]", "i")
                .str.replace_all(r"[óòôö]", "o")
                .str.replace_all(r"[úùûü]", "u")
                .str.replace_all(r"[ñ]", "n")
                .str.replace_all(r"[Á]", "A")
                .str.replace_all(r"[É]", "E")
                .str.replace_all(r"[Í]", "I")
                .str.replace_all(r"[Ó]", "O")
                .str.replace_all(r"[Ú]", "U")
                .str.replace_all(r"[Ñ]", "N")
        )
        .with_columns(
            pl.col("nombre_estacion").str.replace(r"^Cabecera", "Portal")
        )
    )

    df_linea_counts = df.select(["codigo_linea", "nombre_linea"]).group_by(["codigo_linea", "nombre_linea"]).len()
    df_estacion_counts = df.select(["codigo_estacion", "nombre_estacion"]).group_by(["codigo_estacion", "nombre_estacion"]).len()
    df_linea_estaciones = df.select(["codigo_linea", "codigo_estacion"]).group_by(["codigo_linea", "codigo_estacion"]).len()

    # 9. Selección final y persistencia ────────────────────────────
    df = df.select(FINAL_COLUMNS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)
    logger.info(f"Guardado: {output_path.name} ({len(df)} filas)")
    return output_path, df_linea_counts, df_estacion_counts, df_linea_estaciones


# ======================================
# 3. ORQUESTADOR: pipeline completo
# ======================================


def run_data_pipeline() -> list[Path]:
    """
    Pipeline end-to-end:
    1. Ingesta (descubrir, descargar, extraer)
    2. Por cada CSV, aplicar process_raw_data
    Devuelve lista de parquets generados.
    """
    logger.info("========== DATA PIPELINE START ==========")
 
    # 0. GeoJSON de estaciones
    download_geojson()
    load_stations_geojson()
 
    # 1. Ingesta
    csv_files = ingest_raw_data()
    if not csv_files:
        return []
 
    # 2. Procesamiento
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    parquet_files: list[Path] = []
    
    all_df_linea = []
    all_df_estacion = []
    all_df_linea_estaciones = []
 
    for csv_path in tqdm(csv_files, desc="Procesando CSVs"):
        out = PARQUET_DIR / csv_path.with_suffix(".parquet").name
        try:
            pq, df_linea, df_estacion, df_linea_estaciones_out = process_raw_data(csv_path, out)
            parquet_files.append(pq)
            if df_linea is not None:
                all_df_linea.append(df_linea)
            if df_estacion is not None:
                all_df_estacion.append(df_estacion)
            if df_linea_estaciones_out is not None:
                all_df_linea_estaciones.append(df_linea_estaciones_out)
        except Exception as e:
            logger.error(f"Error procesando {csv_path.name}: {e}")
 
    if all_df_linea and all_df_linea_estaciones:
        _update_dimensions(all_df_linea, all_df_estacion, all_df_linea_estaciones)
 
    logger.success(
        f"========== PIPELINE COMPLETO: {len(parquet_files)} parquets =========="
    )
    return parquet_files

def _update_dimensions(all_df_linea: list[pl.DataFrame], all_df_estacion: list[pl.DataFrame], all_df_linea_estaciones: list[pl.DataFrame]):
    logger.info("Actualizando dimensiones (JSONs)")
    df_lineas_concat = pl.concat(all_df_linea)
    df_estaciones_concat = pl.concat(all_df_estacion)
    df_linea_estaciones_concat = pl.concat(all_df_linea_estaciones)

    df_lineas_agg = (
        df_lineas_concat.group_by(["codigo_linea", "nombre_linea"]).sum()
        .sort(["len"], descending=True)
        .unique(subset=["codigo_linea"], keep="first")
        .sort("codigo_linea")
    )
    df_estaciones_agg = (
        df_estaciones_concat.group_by(["codigo_estacion", "nombre_estacion"]).sum()
        .sort(["len"], descending=True)
        .unique(subset=["codigo_estacion"], keep="first")
        .sort("codigo_estacion")
    )
    df_linea_estaciones_agg = (
        df_linea_estaciones_concat.group_by(["codigo_linea", "codigo_estacion"]).sum()
        .group_by("codigo_linea")
        .agg(pl.col("codigo_estacion").unique())
    )

    dim_linea = {
        str(row["codigo_linea"]): row["nombre_linea"] 
        for row in df_lineas_agg.to_dicts()
    }
    dim_estacion = {
        str(row["codigo_estacion"]): row["nombre_estacion"] 
        for row in df_estaciones_agg.to_dicts()
    }
    linea_estaciones = {
        str(row["codigo_linea"]): row["codigo_estacion"] 
        for row in df_linea_estaciones_agg.to_dicts()
    }

    path_dim_linea = PROCESSED_DATA_DIR / "dim_linea.json"
    path_dim_estacion = PROCESSED_DATA_DIR / "dim_estacion.json"
    path_linea_estaciones = PROCESSED_DATA_DIR / "linea_estaciones.json"
    
    if path_dim_linea.exists():
        try:
            with open(path_dim_linea, "r", encoding="utf-8") as f:
                old_dim_linea = json.load(f)
                dim_linea = {**old_dim_linea, **dim_linea}
        except Exception as e:
            logger.warning(f"No se pudo leer {path_dim_linea.name}: {e}")
            
    if path_dim_estacion.exists():
        try:
            with open(path_dim_estacion, "r", encoding="utf-8") as f:
                old_dim_estacion = json.load(f)
                dim_estacion = {**old_dim_estacion, **dim_estacion}
        except Exception as e:
            logger.warning(f"No se pudo leer {path_dim_estacion.name}: {e}")

    if path_linea_estaciones.exists():
        try:
            with open(path_linea_estaciones, "r", encoding="utf-8") as f:
                old_linea_estaciones = json.load(f)
                linea_estaciones = {**old_linea_estaciones, **linea_estaciones}
        except Exception as e:
            logger.warning(f"No se pudo leer {path_linea_estaciones.name}: {e}")
            
    with open(path_dim_linea, "w", encoding="utf-8") as f:
        json.dump(dim_linea, f, ensure_ascii=False, indent=2)

    with open(path_dim_estacion, "w", encoding="utf-8") as f:
        json.dump(dim_estacion, f, ensure_ascii=False, indent=2)
        
    with open(path_linea_estaciones, "w", encoding="utf-8") as f:
        json.dump(linea_estaciones, f, ensure_ascii=False, indent=2)
        
    logger.info("Dimensiones guardadas en JSON exitosamente.")


# =======================
# HELPERS (privados)
# =======================

_NET_TIMEOUT = 30
_MAX_RETRIES = 3


def _urlopen_with_retry(url: str, timeout: int = _NET_TIMEOUT) -> str:
    """Abre una URL con reintentos y backoff exponencial. Devuelve el cuerpo como str."""
    for attempt in range(_MAX_RETRIES):
        try:
            with urlopen(url, timeout=timeout) as response:
                return response.read().decode("utf-8")
        except (URLError, HTTPError, OSError) as exc:
            if attempt < _MAX_RETRIES - 1:
                wait = 2 ** attempt
                logger.warning(f"Intento {attempt + 1}/{_MAX_RETRIES} fallido para {url}: {exc}. Reintentando en {wait}s...")
                time.sleep(wait)
            else:
                logger.error(f"No se pudo acceder a {url} tras {_MAX_RETRIES} intentos: {exc}")
                raise


def _urlretrieve_with_retry(url: str, dest: Path, timeout: int = _NET_TIMEOUT) -> None:
    """Descarga url a dest con reintentos y backoff exponencial."""
    for attempt in range(_MAX_RETRIES):
        try:
            urlretrieve(url, dest)
            return
        except (URLError, HTTPError, OSError) as exc:
            if dest.exists():
                dest.unlink()
            if attempt < _MAX_RETRIES - 1:
                wait = 2 ** attempt
                logger.warning(f"Intento {attempt + 1}/{_MAX_RETRIES} fallido descargando {url}: {exc}. Reintentando en {wait}s...")
                time.sleep(wait)
            else:
                logger.error(f"No se pudo descargar {url} tras {_MAX_RETRIES} intentos: {exc}")
                raise


def _list_remote_files() -> list[str]:
    """Consulta el HTML remoto y devuelve nombres de ZIP del año 2025."""
    logger.info(f"Consultando índice remoto: {INDEX_URL}")
    html = _urlopen_with_retry(INDEX_URL)

    all_zips = re.findall(r"(salidas\d{8}\.zip)", html)

    # Deduplicar manteniendo orden
    seen: set[str] = set()
    unique_zips: list[str] = []
    for name in all_zips:
        if name not in seen:
            seen.add(name)
            unique_zips.append(name)

    # Filtrar solo 2025
    zips_2025 = [z for z in unique_zips if z.startswith("salidas2025")]
    logger.info(f"Encontrados {len(zips_2025)} archivos ZIP del año 2025")
    return zips_2025


def _download_file(name: str) -> Path:
    """Descarga un ZIP si no existe localmente."""
    ZIP_DIR.mkdir(parents=True, exist_ok=True)
    dest = ZIP_DIR / name

    if dest.exists():
        logger.debug(f"Ya existe, saltando descarga: {name}")
        return dest

    url = BASE_URL + name
    logger.info(f"Descargando {url} → {dest}")
    _urlretrieve_with_retry(url, dest)
    return dest


def _extract_zip(zip_path: Path) -> list[Path]:
    """Extrae CSVs de un ZIP al directorio de CSVs crudos."""
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []

    try:
        zf_obj = zipfile.ZipFile(zip_path, "r")
    except zipfile.BadZipFile:
        logger.error(f"ZIP corrupto o inválido, eliminando: {zip_path.name}")
        zip_path.unlink(missing_ok=True)
        return []

    with zf_obj as zf:
        for member in zf.namelist():
            if member.lower().endswith(".csv"):
                out_path = CSV_DIR / Path(member).name
                if out_path.exists():
                    logger.debug(f"CSV ya extraído, saltando: {out_path.name}")
                    extracted.append(out_path)
                    continue
                zf.extract(member, CSV_DIR)
                actual = CSV_DIR / member
                if actual != out_path:
                    actual.rename(out_path)
                extracted.append(out_path)

    return extracted


# =======================
# CLI
# =======================


@app.command()
def main(force_geojson: bool = typer.Option(False, "--force-geojson", help="Forzar re-descarga del GeoJSON aunque ya exista")):
    """Ejecuta el pipeline completo de datos de TransMilenio 2025."""
    download_geojson(force=force_geojson)
    run_data_pipeline()


if __name__ == "__main__":
    app()
