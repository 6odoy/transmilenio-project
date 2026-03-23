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
    "codigo_linea", "nombre_linea",
    "codigo_estacion", "nombre_estacion",
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


def process_raw_data(csv_path: Path, output_path: Path) -> Path:
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
        logger.debug(f"Parquet ya existe, saltando: {output_path.name}")
        return output_path

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

    # 7. Consolidación de variantes de San Mateo ──────────────────
    #    Todas las variantes cuyo nombre termina en "San Mateo" se
    #    consolidan bajo "SAN MATEO - C.C. UNISUR", sumando conteos.
    san_mateo_variantes = df.filter(
        pl.col("nombre_estacion").str.contains(r".*San Mateo$")
    )
    san_mateo_unisur = df.filter(
        pl.col("nombre_estacion") == "SAN MATEO - C.C. UNISUR"
    )
    san_mateo_agg = san_mateo_variantes.group_by("timestamp").agg(
        pl.col("entradas").sum(),
        pl.col("salidas").sum(),
        pl.col("total").sum(),
    )
    resultado = (
        san_mateo_unisur
        .join(san_mateo_agg, on="timestamp", how="left", suffix="_var")
        .with_columns(
            (pl.col("entradas") + pl.col("entradas_var")).alias("entradas"),
            (pl.col("salidas") + pl.col("salidas_var")).alias("salidas"),
            (pl.col("total") + pl.col("total_var")).alias("total"),
        )
        .drop("entradas_var", "salidas_var", "total_var")
    )
    df = df.filter(
        ~pl.col("nombre_estacion").str.contains(r".*San Mateo$")
        & (pl.col("nombre_estacion") != "SAN MATEO - C.C. UNISUR")
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

    # 9. Selección final y persistencia ────────────────────────────
    df = df.select(FINAL_COLUMNS)

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
