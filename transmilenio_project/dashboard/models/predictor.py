"""
TransMilenio — Módulo de Predicción para el Dashboard
======================================================
Carga los artefactos serializados (modelo Random Forest, encoder, scaler) y
expone funciones que el dashboard puede invocar directamente para predecir
la Afluencia (Entradas) a partir de datos nuevos del usuario.

Uso desde el dashboard:

    from models.predictor import predecir_afluencia, obtener_metricas

    resultado = predecir_afluencia(
        mes=4,
        dia=15,
        hora=7,
        minuto=30,
        segundo=0,
        letra_zona="A",         # Zona/Letra de línea
        lat=4.6097,
        lon=-74.0817,
    )
    print(resultado)
    # {
    #   "prediccion_normalizada": 0.0532,
    #   "prediccion_entradas": 52,         # desnormalizado
    #   "modelo_utilizado": "Random Forest (max_depth=5)",
    #   "exito": True,
    #   "mensaje": ""
    # }
"""

import os
import numpy as np
import pandas as pd
import joblib

# ==============================================================================
# RUTAS
# ==============================================================================
_ARTEFACTOS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "artefactos"
)

# ==============================================================================
# CONSTANTES (deben coincidir con entrenar_modelo.py)
# ==============================================================================
LETRAS_ZONA = list("ABCDEFGHJKLT")
LETRAS_COLS = [f"Letra_{z}" for z in LETRAS_ZONA]
FEATURE_COLS = [
    "Mes", "Día", "Hora", "Minuto", "Segundo",
    *LETRAS_COLS,
    "lat", "lon",
]

# Rango máximo original de Entradas para desnormalizar (fallback)
_MAX_ENTRADAS_ORIGINAL = 980

# ==============================================================================
# CACHÉ DE ARTEFACTOS (se cargan una vez y se reutilizan)
# ==============================================================================
_cache = {}


def _cargar_artefacto(nombre_archivo):
    """Carga un artefacto joblib desde el directorio de artefactos."""
    ruta = os.path.join(_ARTEFACTOS_DIR, nombre_archivo)
    if nombre_archivo not in _cache:
        if not os.path.exists(ruta):
            raise FileNotFoundError(
                f"No se encontró el artefacto '{nombre_archivo}' en {_ARTEFACTOS_DIR}. "
                "Ejecute primero:  python -m models.entrenar_modelo"
            )
        _cache[nombre_archivo] = joblib.load(ruta)
    return _cache[nombre_archivo]


def cargar_modelo():
    """
    Carga el modelo Random Forest serializado.

    Returns:
        sklearn RandomForestRegressor ajustado.
    """
    return _cargar_artefacto("random_forest.joblib")


def cargar_scaler():
    """Devuelve el MinMaxScaler ajustado."""
    return _cargar_artefacto("scaler.joblib")


def cargar_encoder():
    """Devuelve el OneHotEncoder ajustado."""
    return _cargar_artefacto("encoder.joblib")


def obtener_metricas() -> dict:
    """
    Retorna las métricas de evaluación guardadas durante el entrenamiento.
    Estructura:
        {
            "rf": {"rmse": float, "r2": float}
        }
    """
    return _cargar_artefacto("metrics.joblib")


# ==============================================================================
# FUNCIÓN DE PREDICCIÓN
# ==============================================================================
def predecir_afluencia(
    mes: int = 4,
    dia: int = 1,
    hora: int = 7,
    minuto: int = 0,
    segundo: int = 0,
    letra_zona: str = "A",
    lat: float = 4.6097,
    lon: float = -74.0817,
) -> dict:
    """
    Predice la afluencia de pasajeros (Entradas) para un escenario dado
    utilizando el modelo Random Forest.

    Args:
        mes:          Mes (1-12).
        dia:          Día del mes (1-31).
        hora:         Hora del día (0-23).
        minuto:       Minuto (0-59).
        segundo:      Segundo (0-59).
        letra_zona:   Letra de la zona/línea (A, B, C, D, E, F, G, H, J, K, L, T).
        lat:          Latitud de la estación.
        lon:          Longitud de la estación.

    Returns:
        dict con claves:
            "prediccion_normalizada" : valor entre 0 y 1 (espacio normalizado)
            "prediccion_entradas"    : valor aproximado desnormalizado (entero)
            "modelo_utilizado"       : nombre legible del modelo
            "exito"                  : bool
            "mensaje"                : str con detalles en caso de error
    """
    nombre_modelo = "Random Forest (max_depth=5)"

    try:
        # Validar letra
        letra_zona = letra_zona.upper()
        if letra_zona not in LETRAS_ZONA:
            return {
                "prediccion_normalizada": None,
                "prediccion_entradas": None,
                "modelo_utilizado": nombre_modelo,
                "exito": False,
                "mensaje": f"Zona '{letra_zona}' no válida. Use: {LETRAS_ZONA}",
            }

        # Construir vector de features con One-Hot manual
        one_hot = {f"Letra_{z}": (1.0 if z == letra_zona else 0.0) for z in LETRAS_ZONA}

        fila = {
            "Mes":     float(mes),
            "Día":     float(dia),
            "Hora":    float(hora),
            "Minuto":  float(minuto),
            "Segundo": float(segundo),
            **one_hot,
            "lat":     float(lat),
            "lon":     float(lon),
        }

        # Construir DataFrame con el orden exacto de features
        df_input = pd.DataFrame([fila], columns=FEATURE_COLS)

        # Normalizar con el scaler guardado
        scaler = cargar_scaler()

        # El scaler fue ajustado sobre TODAS las columnas (incluida Entradas).
        # Para transformar solo features, necesitamos añadir un placeholder para
        # Entradas (el scaler espera 20 columnas: features + Entradas).
        cols_scaler = list(scaler.feature_names_in_) if hasattr(scaler, "feature_names_in_") else None

        if cols_scaler:
            # Reconstruir la fila con todas las columnas del scaler
            fila_completa = {}
            for col in cols_scaler:
                if col == "Entradas":
                    fila_completa[col] = 0.0  # placeholder
                elif col in fila:
                    fila_completa[col] = fila[col]
                else:
                    fila_completa[col] = 0.0  # columnas extra no presentes

            df_full = pd.DataFrame([fila_completa], columns=cols_scaler)
            df_scaled = pd.DataFrame(
                scaler.transform(df_full), columns=cols_scaler
            )
            # Extraer solo las features que el modelo necesita
            X = df_scaled[FEATURE_COLS]
        else:
            # Fallback: usar el input sin normalizar
            X = df_input

        # Predecir con Random Forest
        modelo_obj = cargar_modelo()
        pred_normalizada = float(modelo_obj.predict(X)[0])

        # Desnormalizar la predicción usando los parámetros del scaler
        if cols_scaler and "Entradas" in cols_scaler:
            idx_entradas = list(cols_scaler).index("Entradas")
            min_val = scaler.data_min_[idx_entradas]
            max_val = scaler.data_max_[idx_entradas]
            pred_entradas = int(round(pred_normalizada * (max_val - min_val) + min_val))
        else:
            pred_entradas = int(round(pred_normalizada * _MAX_ENTRADAS_ORIGINAL))

        return {
            "prediccion_normalizada": round(pred_normalizada, 6),
            "prediccion_entradas": max(pred_entradas, 0),
            "modelo_utilizado": nombre_modelo,
            "exito": True,
            "mensaje": "",
        }

    except FileNotFoundError as e:
        return {
            "prediccion_normalizada": None,
            "prediccion_entradas": None,
            "modelo_utilizado": nombre_modelo,
            "exito": False,
            "mensaje": str(e),
        }
    except Exception as e:
        return {
            "prediccion_normalizada": None,
            "prediccion_entradas": None,
            "modelo_utilizado": nombre_modelo,
            "exito": False,
            "mensaje": f"Error inesperado: {str(e)}",
        }


def obtener_estaciones_disponibles() -> list:
    """
    Retorna una lista de letras de zona disponibles para el modelo.
    Útil para poblar dropdowns en el dashboard.
    """
    zonas_nombres = {
        "A": "Zona A — Caracas",
        "B": "Zona B — AutoNorte",
        "C": "Zona C — Americas / Calle 26",
        "D": "Zona D — Suba",
        "E": "Zona E — NQS",
        "F": "Zona F — Eje Ambiental",
        "G": "Zona G — Calle 80",
        "H": "Zona H — Usme / NQS Sur",
        "J": "Zona J — AutoSur",
        "K": "Zona K — Caracas Sur",
        "L": "Zona L — Av. 68",
        "T": "Zona T — Troncal Soacha",
    }
    return [
        {"label": zonas_nombres.get(z, z), "value": z}
        for z in LETRAS_ZONA
    ]


# ==============================================================================
# Para pruebas rápidas
# ==============================================================================
if __name__ == "__main__":
    print("Probando predicción con Random Forest...")
    res = predecir_afluencia(
        mes=4, dia=15, hora=7, minuto=30, segundo=0,
        letra_zona="A", lat=4.6097, lon=-74.0817
    )
    for k, v in res.items():
        print(f"  {k}: {v}")
