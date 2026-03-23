import json
import re
import zipfile
from pathlib import Path
from urllib.request import urlopen, urlretrieve

import polars as pl
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
            return output_path, df_linea_counts, df_estacion_counts
            
        logger.debug(f"Parquet ya existe, saltando: {output_path.name}")
        return output_path, None, None

    logger.debug(f"Procesando: {csv_path.name}")

    # 1. Carga y selección de columnas ─────────────────────────────
    df = pl.read_csv(csv_path, try_parse_dates=False, infer_schema_length=5000)
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

    # 9. Selección final y persistencia ────────────────────────────
    df = df.select(FINAL_COLUMNS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)
    logger.info(f"Guardado: {output_path.name} ({len(df)} filas)")
    return output_path, df_linea_counts, df_estacion_counts


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

    # 1. Ingesta
    csv_files = ingest_raw_data()
    if not csv_files:
        return []

    # 2. Procesamiento
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    parquet_files: list[Path] = []
    
    all_df_linea = []
    all_df_estacion = []

    for csv_path in tqdm(csv_files, desc="Procesando CSVs"):
        out = PARQUET_DIR / csv_path.with_suffix(".parquet").name
        try:
            pq, df_linea, df_estacion = process_raw_data(csv_path, out)
            parquet_files.append(pq)
            if df_linea is not None:
                all_df_linea.append(df_linea)
            if df_estacion is not None:
                all_df_estacion.append(df_estacion)
        except Exception as e:
            logger.error(f"Error procesando {csv_path.name}: {e}")

    if all_df_linea and all_df_estacion:
        _update_dimensions(all_df_linea, all_df_estacion)

    logger.success(
        f"========== PIPELINE COMPLETO: {len(parquet_files)} parquets =========="
    )
    return parquet_files

def _update_dimensions(all_df_linea: list[pl.DataFrame], all_df_estacion: list[pl.DataFrame]):
    logger.info("Actualizando dimensiones (JSONs)")
    df_lineas_concat = pl.concat(all_df_linea)
    df_estaciones_concat = pl.concat(all_df_estacion)

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

    dim_linea = {
        str(row["codigo_linea"]): row["nombre_linea"] 
        for row in df_lineas_agg.to_dicts()
    }
    dim_estacion = {
        str(row["codigo_estacion"]): row["nombre_estacion"] 
        for row in df_estaciones_agg.to_dicts()
    }

    path_dim_linea = PROCESSED_DATA_DIR / "dim_linea.json"
    path_dim_estacion = PROCESSED_DATA_DIR / "dim_estacion.json"
    
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
            
    with open(path_dim_linea, "w", encoding="utf-8") as f:
        json.dump(dim_linea, f, ensure_ascii=False, indent=2)
        
    with open(path_dim_estacion, "w", encoding="utf-8") as f:
        json.dump(dim_estacion, f, ensure_ascii=False, indent=2)
        
    logger.info("Dimensiones guardadas en JSON exitosamente.")


# =======================
# HELPERS (privados)
# =======================


def _list_remote_files() -> list[str]:
    """Consulta el HTML remoto y devuelve nombres de ZIP del año 2025."""
    logger.info(f"Consultando índice remoto: {INDEX_URL}")
    with urlopen(INDEX_URL) as response:
        html = response.read().decode("utf-8")

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
    urlretrieve(url, dest)
    return dest


def _extract_zip(zip_path: Path) -> list[Path]:
    """Extrae CSVs de un ZIP al directorio de CSVs crudos."""
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []

    with zipfile.ZipFile(zip_path, "r") as zf:
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
def main():
    """Ejecuta el pipeline completo de datos de TransMilenio 2025."""
    run_data_pipeline()


if __name__ == "__main__":
    app()
