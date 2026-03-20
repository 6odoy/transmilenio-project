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
    "Acceso_Estacion",
    "Entradas_E",
    "Salidas_S",
]
GROUP_KEYS = ["Fecha_Transaccion", "Tiempo", "Estacion"]


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


def process_raw_data(csv_path: Path, output_path: Path) -> Path:
    """
    Limpia, agrega, transforma y persiste un CSV individual como parquet.
    - Elimina columnas innecesarias (Dispositivo, Acceso_Estacion, Linea)
    - Crea columna timestamp (Fecha + Tiempo)
    - Agrupa por fecha, hora y estación sumando entradas/salidas
    - Extrae fecha y hora del timestamp
    - Extrae código y nombre de la estación
    - Calcula total de validaciones (entradas + salidas)
    - Guarda el resultado en parquet
    """
    if output_path.exists():
        logger.debug(f"Parquet ya existe, saltando: {output_path.name}")
        return output_path

    # --- Limpieza ---
    logger.debug(f"Procesando: {csv_path.name}")
    df = pl.read_csv(csv_path, try_parse_dates=False, infer_schema_length=5000)

    # Seleccionar solo columnas útiles (elimina 'Dispositivo')
    df = df.select(COLUMNS_TO_KEEP)

    # Tipar columnas numéricas
    df = df.with_columns(
        pl.col("Entradas_E").cast(pl.Int64),
        pl.col("Salidas_S").cast(pl.Int64),
    )

    # Crear timestamp combinando Fecha_Transaccion + Tiempo
    df = df.with_columns(
        pl.concat_str(
            [pl.col("Fecha_Transaccion"), pl.col("Tiempo")],
            separator=" ",
        )
        .str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False)
        .alias("timestamp")
    )

    # --- Agregación por fecha + hora + estación ---
    df = (
        df.group_by(GROUP_KEYS)
        .agg(
            pl.col("Linea").first(),
            pl.col("Acceso_Estacion").first(),
            pl.col("Entradas_E").sum(),
            pl.col("Salidas_S").sum(),
            pl.col("timestamp").first(),
        )
        .sort(GROUP_KEYS)
    )

    # --- Transformación final ---
    df = (
        df
        .drop("Fecha_Transaccion", "Tiempo")
        .drop("Acceso_Estacion", "Linea")
        .rename({
            "Entradas_E": "entradas",
            "Salidas_S": "salidas",
        })
        .with_columns(
            pl.col("timestamp").dt.date().alias("fecha"),
            pl.col("timestamp").dt.time().alias("hora"),
            pl.col("Estacion").str.extract(r"\((\d+)\)", group_index=1).alias("codigo").cast(pl.Int64),
            pl.col("Estacion").str.extract(r"\)(.+)", group_index=1).alias("nombre"),
            (pl.col("entradas") + pl.col("salidas")).alias("total"),
        )
        .drop("timestamp", "Estacion")
        .select(["fecha", "hora", "codigo", "nombre", "entradas", "salidas", "total"])
    )

    # --- Persistencia ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)
    logger.info(f"Guardado: {output_path.name} ({len(df)} filas)")
    return output_path


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

    for csv_path in tqdm(csv_files, desc="Procesando CSVs"):
        out = PARQUET_DIR / csv_path.with_suffix(".parquet").name
        try:
            pq = process_raw_data(csv_path, out)
            parquet_files.append(pq)
        except Exception as e:
            logger.error(f"Error procesando {csv_path.name}: {e}")

    logger.success(
        f"========== PIPELINE COMPLETO: {len(parquet_files)} parquets =========="
    )
    return parquet_files


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
