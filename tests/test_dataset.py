"""
Tests básicos para el pipeline de datos de TransMilenio.
Cubre: validación de columnas, corrección de retornos y helpers de red.
"""

import zipfile
import pytest
import polars as pl
from pathlib import Path
from unittest.mock import patch, MagicMock
from urllib.error import URLError

from transmilenio_project.dataset import (
    COLUMNS_TO_KEEP,
    FINAL_COLUMNS,
    process_raw_data,
    _urlopen_with_retry,
    _urlretrieve_with_retry,
    _extract_zip,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def sample_csv(tmp_path: Path) -> Path:
    """CSV mínimo con todas las columnas requeridas."""
    content = (
        "Fecha_Transaccion,Tiempo,Linea,Estacion,Entradas_E,Salidas_S\n"
        "2025-01-15,07:00:00,(36)Zona A - Caracas,(9001)Portal Norte,100,80\n"
        "2025-01-15,08:00:00,(36)Zona A - Caracas,(9001)Portal Norte,200,150\n"
    )
    csv_file = tmp_path / "salidas20250115.csv"
    csv_file.write_text(content, encoding="utf-8")
    return csv_file


@pytest.fixture()
def sample_csv_missing_col(tmp_path: Path) -> Path:
    """CSV al que le falta la columna Entradas_E."""
    content = (
        "Fecha_Transaccion,Tiempo,Linea,Estacion,Salidas_S\n"
        "2025-01-15,07:00:00,(36)Zona A,(9001)Portal Norte,80\n"
    )
    csv_file = tmp_path / "bad.csv"
    csv_file.write_text(content, encoding="utf-8")
    return csv_file


# ──────────────────────────────────────────────────────────────────────────────
# process_raw_data — retornos
# ──────────────────────────────────────────────────────────────────────────────

def test_process_raw_data_returns_four_values(sample_csv, tmp_path):
    """Siempre debe desempacar 4 valores, ya sea procesando o usando caché."""
    out = tmp_path / "out.parquet"
    result = process_raw_data(sample_csv, out)
    assert len(result) == 4, "process_raw_data debe retornar exactamente 4 valores"


def test_process_raw_data_cache_returns_four_values(sample_csv, tmp_path):
    """Cuando el parquet ya existe debe retornar 4 valores (no 3)."""
    out = tmp_path / "out.parquet"
    process_raw_data(sample_csv, out)   # primera vez: crea el parquet
    result = process_raw_data(sample_csv, out)  # segunda vez: usa caché
    assert len(result) == 4


def test_process_raw_data_creates_parquet(sample_csv, tmp_path):
    out = tmp_path / "out.parquet"
    process_raw_data(sample_csv, out)
    assert out.exists()


def test_process_raw_data_parquet_schema(sample_csv, tmp_path):
    """El parquet resultante debe tener exactamente las columnas de FINAL_COLUMNS."""
    out = tmp_path / "out.parquet"
    process_raw_data(sample_csv, out)
    schema = pl.read_parquet_schema(out)
    assert set(schema.keys()) == set(FINAL_COLUMNS)


# ──────────────────────────────────────────────────────────────────────────────
# process_raw_data — validación de columnas
# ──────────────────────────────────────────────────────────────────────────────

def test_process_raw_data_missing_column_raises(sample_csv_missing_col, tmp_path):
    """Debe lanzar ValueError si faltan columnas requeridas."""
    out = tmp_path / "out.parquet"
    with pytest.raises(ValueError, match="Columnas faltantes"):
        process_raw_data(sample_csv_missing_col, out)


# ──────────────────────────────────────────────────────────────────────────────
# COLUMNS_TO_KEEP / FINAL_COLUMNS — invariantes
# ──────────────────────────────────────────────────────────────────────────────

def test_columns_to_keep_not_empty():
    assert len(COLUMNS_TO_KEEP) > 0


def test_final_columns_not_empty():
    assert len(FINAL_COLUMNS) > 0


# ──────────────────────────────────────────────────────────────────────────────
# _urlopen_with_retry — manejo de fallos
# ──────────────────────────────────────────────────────────────────────────────

def test_urlopen_retry_raises_after_max_attempts():
    """Debe re-lanzar la excepción tras agotar los reintentos."""
    with patch("transmilenio_project.dataset.urlopen", side_effect=URLError("timeout")):
        with patch("transmilenio_project.dataset.time.sleep"):
            with pytest.raises(URLError):
                _urlopen_with_retry("http://example.com")


def test_urlopen_retry_succeeds_on_second_attempt():
    """Debe tener éxito si el segundo intento funciona."""
    mock_response = MagicMock()
    mock_response.__enter__ = lambda s: s
    mock_response.__exit__ = MagicMock(return_value=False)
    mock_response.read.return_value = b"<html>ok</html>"

    side_effects = [URLError("timeout"), mock_response]
    with patch("transmilenio_project.dataset.urlopen", side_effect=side_effects):
        with patch("transmilenio_project.dataset.time.sleep"):
            result = _urlopen_with_retry("http://example.com")
    assert result == "<html>ok</html>"


# ──────────────────────────────────────────────────────────────────────────────
# _urlretrieve_with_retry — limpieza de archivo parcial
# ──────────────────────────────────────────────────────────────────────────────

def test_urlretrieve_cleans_partial_file_on_failure(tmp_path):
    """Si la descarga falla, no debe quedar un archivo parcial."""
    dest = tmp_path / "file.zip"

    def bad_retrieve(url, path):
        Path(path).write_bytes(b"partial")
        raise URLError("connection reset")

    with patch("transmilenio_project.dataset.urlretrieve", side_effect=bad_retrieve):
        with patch("transmilenio_project.dataset.time.sleep"):
            with pytest.raises(URLError):
                _urlretrieve_with_retry("http://example.com/file.zip", dest)

    assert not dest.exists(), "El archivo parcial debe eliminarse tras el fallo"


# ──────────────────────────────────────────────────────────────────────────────
# _extract_zip — ZIP corrupto
# ──────────────────────────────────────────────────────────────────────────────

def test_extract_zip_bad_zip_returns_empty(tmp_path):
    """Un ZIP corrupto debe retornar lista vacía, no lanzar excepción."""
    bad_zip = tmp_path / "bad.zip"
    bad_zip.write_bytes(b"this is not a zip")
    result = _extract_zip(bad_zip)
    assert result == []


def test_extract_zip_valid(tmp_path):
    """Un ZIP válido con CSV debe extraerlo correctamente."""
    zip_path = tmp_path / "valid.zip"
    csv_content = b"Fecha_Transaccion,Tiempo\n2025-01-01,07:00:00\n"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("data.csv", csv_content)

    extracted = _extract_zip(zip_path)
    assert len(extracted) == 1
    assert extracted[0].suffix == ".csv"
