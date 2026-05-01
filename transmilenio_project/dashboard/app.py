"""
TransMilenio — Dashboard Operativo de Análisis y Simulación
=============================================================
Aplicación principal construida con Dash + Plotly.
Incluye módulo de predicción con Random Forest integrado.
Configuración centralizada en config.py.
"""

import dash
import json
import os
import time
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, State, no_update, callback_context
import dash_bootstrap_components as dbc

from config import TM_ROJO, TM_AMARILLO, THEMES, LANGUAGES, BASE_DIR
from models.predictor import (
    predecir_afluencia,
    obtener_estaciones_disponibles,
)

# ==============================================================================
# INICIALIZACIÓN DE LA APLICACIÓN
# ==============================================================================
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.FLATLY,
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap",
        "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css"
    ],
    suppress_callback_exceptions=True,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"},
        {"name": "description", "content": "Dashboard operativo de TransMilenio — Monitoreo, análisis de afluencia y simulación."}
    ]
)
app.title = "TransMilenio | Dashboard Operativo"
server = app.server  # Exponer para despliegue (Gunicorn, etc.)


# ==============================================================================
# COMPONENTES REUTILIZABLES
# ==============================================================================
def build_navbar(t, lang_code):
    """Construye la barra de navegación superior con logos, título y controles."""
    return dbc.Navbar(
        dbc.Container([
            # ── Logos + Título ──
            html.Div(
                dbc.Row([
                    dbc.Col(html.Div([
                        html.Img(src="/assets/ext_logo.svg", height="40px"),
                        html.Div(style={
                            "borderLeft": "2px solid rgba(255, 255, 255, 0.5)",
                            "height": "30px", "margin": "0 15px"
                        }),
                        html.Img(src="/assets/tm_logo.svg", height="40px")
                    ], className="d-flex align-items-center me-3")),
                    dbc.Col(dbc.NavbarBrand(
                        t["nav_title"],
                        className="ms-2 fs-4 fw-bold text-white",
                        style={"letterSpacing": "0.5px"}
                    )),
                ], align="center", className="g-0")
            ),
            # ── Controles: Idioma + Tema ──
            dbc.Row([
                dbc.Col([
                    dbc.Select(
                        id="lang-select",
                        options=[
                            {"label": "🇪🇸 Español", "value": "es"},
                            {"label": "🇬🇧 English", "value": "en"},
                            {"label": "🇮🇹 Italiano", "value": "it"},
                        ],
                        value=lang_code,
                        size="sm",
                        className="me-4 shadow-sm fw-medium",
                        style={"cursor": "pointer", "width": "125px"}
                    )
                ], className="d-flex align-items-center"),
                dbc.Col([
                    html.I(className="bi bi-sun-fill text-white me-2", style={"fontSize": "1.2rem"}),
                    dbc.Switch(id="theme-switch", value=False, className="d-inline-block", style={"cursor": "pointer"}),
                    html.I(className="bi bi-moon-stars-fill text-white ms-2", style={"fontSize": "1.1rem"}),
                ], className="d-flex align-items-center"),
            ], className="ms-auto", align="center")
        ], fluid=True),
        color=TM_ROJO, dark=True, className="shadow-sm",
        style={"borderBottom": f"4px solid {TM_AMARILLO}", "zIndex": "10"}
    )


def build_kpi_card(title, value, desc, trend_class, c):
    """Genera una tarjeta KPI individual con estilo premium."""
    return dbc.Card(
        dbc.CardBody([
            html.H6(
                title,
                className="text-uppercase fw-bold mb-2",
                style={"fontSize": "0.75rem", "letterSpacing": "1px", "color": c["text_muted"]}
            ),
            html.H3(value, className="mb-1 fw-bold", style={"color": c["text_main"]}),
            html.Small([
                html.Span("● ", className=trend_class),
                desc
            ], style={"color": c["text_muted"], "fontWeight": "500"})
        ]),
        className="shadow-sm border-0 h-100 kpi-card",
        style={"borderRadius": "12px", "backgroundColor": c["card_bg"]}
    )


def build_placeholder_card(title, subtitle, icon, placeholder_text, height, c):
    """Genera una tarjeta con espacio reservado (placeholder) para componentes futuros."""
    return dbc.Card(
        dbc.CardBody([
            html.Div([
                html.H5(title, className="fw-bold mb-0", style={"color": c["text_main"]}),
                html.Small(subtitle, style={"color": c["text_muted"]})
            ], className="mb-4"),
            html.Div(
                [
                    html.I(className=f"bi {icon} text-muted mb-2", style={"fontSize": "2.5rem"}),
                    html.Span(placeholder_text)
                ],
                className="d-flex flex-column align-items-center justify-content-center fw-medium rounded text-center px-4",
                style={
                    "height": height,
                    "backgroundColor": c["chart_box"],
                    "border": f"2px dashed {c['border']}",
                    "color": c["text_muted"]
                }
            )
        ]),
        className="shadow-sm border-0 mb-4 kpi-card",
        style={"borderRadius": "12px", "backgroundColor": c["card_bg"]}
    )


def build_footer(t, c):
    """Construye el footer institucional."""
    return html.Footer(
        dbc.Container([
            html.Hr(style={"borderColor": c["border"], "borderWidth": "2px"}),
            dbc.Row([
                dbc.Col(
                    html.P([
                        html.Strong(t["footer1"], style={"color": c["text_main"]}),
                        t["footer2"]
                    ], className="small mb-0", style={"color": c["text_muted"]}),
                    md=6
                ),
                dbc.Col(
                    html.P(
                        t["footer3"],
                        className="small text-md-end mb-0 font-monospace",
                        style={"color": c["text_muted"]}
                    ),
                    md=6
                ),
            ])
        ], fluid=True, className="mt-4 pb-4 px-4")
    )


# ==============================================================================
# HELPERS — FORMULARIO DE PREDICCIÓN
# ==============================================================================
def _build_form_group(label_text, input_component, c):
    """Envuelve un input en un form group estilizado."""
    return html.Div([
        html.Label(
            label_text,
            className="form-label fw-semibold mb-1",
            style={"fontSize": "0.8rem", "letterSpacing": "0.5px", "color": c["text_muted"]}
        ),
        input_component
    ], className="mb-3")


def _input_style(c):
    """Estilo compartido para inputs del formulario."""
    return {
        "backgroundColor": c["chart_box"],
        "color": c["text_main"],
        "border": f"1px solid {c['border']}",
        "borderRadius": "8px",
        "fontSize": "0.9rem",
    }


# ==============================================================================
# CARGA DE KPIs REALES — con caché TTL de 5 minutos
# ==============================================================================
_KPI_CACHE: dict = {"data": None, "ts": 0.0}
_TABLE_CACHE: dict = {"data": None, "ts": 0.0}
_CACHE_TTL = 300  # segundos


def load_real_kpis():
    """Carga los KPIs desde el artefacto generado, con caché de 5 minutos."""
    now = time.time()
    if _KPI_CACHE["data"] is not None and now - _KPI_CACHE["ts"] < _CACHE_TTL:
        return _KPI_CACHE["data"]

    kpi_file = os.path.join(BASE_DIR, "models", "artefactos", "kpis.json")
    result = None
    try:
        if os.path.exists(kpi_file):
            with open(kpi_file, "r", encoding="utf-8") as f:
                result = json.load(f)
    except Exception:
        pass

    if result is None:
        result = {
            "kpi1_val": "1.250.000",
            "kpi2_val": "Zona A — Caracas",
            "kpi3_val": "17:00 - 18:00",
            "kpi4_val": "Ricaurte",
            "trend1": "text-success",
            "trend2": "text-success",
            "trend3": "text-warning",
            "trend4": "text-success"
        }

    _KPI_CACHE["data"] = result
    _KPI_CACHE["ts"] = now
    return result


# ==============================================================================
# PÁGINAS
# ==============================================================================

def load_table_data():
    """Carga los datos de la tabla de troncales, con caché de 5 minutos."""
    now = time.time()
    if _TABLE_CACHE["data"] is not None and now - _TABLE_CACHE["ts"] < _CACHE_TTL:
        return _TABLE_CACHE["data"]

    tabla_file = os.path.join(BASE_DIR, "models", "artefactos", "tabla_troncales.json")
    result = []
    try:
        if os.path.exists(tabla_file):
            with open(tabla_file, "r", encoding="utf-8") as f:
                result = json.load(f)
    except Exception:
        pass

    _TABLE_CACHE["data"] = result
    _TABLE_CACHE["ts"] = now
    return result

def build_dynamic_table(t, c):
    """Construye la tabla dinámica con datos cargados."""
    table_data = load_table_data()
    
    if not table_data:
        return build_placeholder_card(t["table_title"], t["table_desc"], "bi-table", t["table_box"], "400px", c)
        
    # Render table header
    header = [
        html.Thead(
            html.Tr([
                html.Th(t.get("th_linea", "Línea / Troncal"), style={"color": c["text_main"], "borderBottom": f"2px solid {c['border']}"}),
                html.Th(t.get("th_estacion", "Estación Principal"), style={"color": c["text_main"], "borderBottom": f"2px solid {c['border']}"}),
                html.Th(t.get("th_ingresos", "Validaciones Totales"), className="text-end", style={"color": c["text_main"], "borderBottom": f"2px solid {c['border']}"})
            ])
        )
    ]
    
    # Render table body
    rows = []
    import re
    def limpiar_nombre(nombre):
        if not isinstance(nombre, str): return nombre
        # Elimina códigos tipo (05000) al inicio
        nombre = re.sub(r"^\(\d+\)", "", nombre)
        # Elimina códigos tipo 05000 - al inicio
        nombre = re.sub(r"^\d+\s*-\s*", "", nombre)
        return nombre.strip()

    for row in table_data:
        estacion = limpiar_nombre(row.get("Top_Estacion", ""))
        rows.append(html.Tr([
            html.Td(row.get("Linea", ""), className="fw-medium", style={"color": c["text_main"], "borderColor": c["border"]}),
            html.Td(estacion, className="text-muted", style={"borderColor": c["border"]}),
            html.Td(row.get("Entradas", ""), className="text-end fw-bold", style={"color": TM_ROJO, "borderColor": c["border"]})
        ]))
    
    body = [html.Tbody(rows)]
    
    return dbc.Card(
        dbc.CardBody([
            html.Div([
                html.H5(t["table_title"], className="fw-bold mb-0", style={"color": c["text_main"]}),
                html.Small(t["table_desc"], style={"color": c["text_muted"]})
            ], className="mb-3"),
            html.Div(
                dbc.Table(
                    header + body, 
                    hover=True, 
                    responsive=True, 
                    className="mb-0", 
                    size="sm", 
                    style={
                        "fontSize": "0.9rem",
                        "color": c["text_main"],
                        "backgroundColor": c["card_bg"]
                    },
                    color="dark" if c["bg_base"] == "#121212" else None
                ),
                style={
                    "maxHeight": "385px", 
                    "overflowY": "auto", 
                    "borderRadius": "8px", 
                    "border": f"1px solid {c['border']}",
                    "backgroundColor": c["card_bg"]
                }
            )
        ]),
        className="shadow-sm border-0 mb-4 kpi-card",
        style={"borderRadius": "12px", "backgroundColor": c["card_bg"]}
    )
def _load_geojson_stations():
    """Carga las estaciones desde el archivo GeoJSON y retorna listas de lat, lon, nombres y colores."""
    from config import GEOJSON_PATH, TM_ROJO
    lats, lons, names, colors = [], [], [], []
    
    color_file = os.path.join(BASE_DIR, "models", "artefactos", "estacion_color.json")
    estacion_colors = {}
    try:
        if os.path.exists(color_file):
            with open(color_file, "r", encoding="utf-8") as f:
                estacion_colors = json.load(f)
    except Exception:
        pass

    try:
        if os.path.exists(GEOJSON_PATH):
            with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            for feat in data.get("features", []):
                geom = feat.get("geometry", {})
                props = feat.get("properties", {})
                
                nodo = str(props.get("codigo_nodo_estacion", ""))
                
                if geom.get("type") == "Point":
                    coords = geom.get("coordinates", [])
                    if len(coords) >= 2:
                        lons.append(coords[0])
                        lats.append(coords[1])
                        names.append(props.get("nombre_estacion", "Estación"))
                        
                        color_asignado = estacion_colors.get(nodo, {}).get("color", TM_ROJO)
                        colors.append(color_asignado)
    except Exception:
        pass
    return lats, lons, names, colors

def _build_station_map(t, c):
    """Construye un mapa Plotly Scattermapbox con las estaciones de TransMilenio."""
    lats, lons, names, colors = _load_geojson_stations()

    if not lats:
        # Fallback si no hay datos
        return html.Div(
            [
                html.I(className="bi bi-map text-muted mb-2", style={"fontSize": "2.5rem"}),
                html.Span(t.get("map_box", "Datos geoespaciales no disponibles"))
            ],
            className="d-flex flex-column align-items-center justify-content-center fw-medium rounded text-center px-4",
            style={
                "height": "420px", "backgroundColor": c["chart_box"],
                "border": f"2px dashed {c['border']}", "color": c["text_muted"]
            }
        )

    is_dark = c["bg_base"] == "#121212"

    fig = go.Figure(go.Scattermap(
        lat=lats,
        lon=lons,
        mode="markers",
        marker=go.scattermap.Marker(
            size=9,
            color=colors if colors else TM_ROJO,
            opacity=0.9,
        ),
        text=names,
        hovertemplate="<b>%{text}</b><br>Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<extra></extra>",
        hoverlabel=dict(
            font_size=12,
            font_family="Inter, sans-serif",
            font_color="white",
            bgcolor=colors if colors else TM_ROJO,
        ),
    ))

    fig.update_layout(
        map=dict(
            style="carto-darkmatter" if is_dark else "carto-positron",
            center=dict(lat=4.648, lon=-74.1),
            zoom=10.8,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=420,
        showlegend=False,
    )

    return dcc.Graph(
        figure=fig,
        config={"displayModeBar": False, "scrollZoom": True},
        style={"borderRadius": "8px", "overflow": "hidden"}
    )


def build_page_dashboard(t, c):
    """Página 1: Dashboard general con KPIs, mapa y tabla."""
    # Cargar datos pre-calculados
    r_kpi = load_real_kpis()

    kpi_row = dbc.Row([
        dbc.Col(build_kpi_card(t["kpi1_title"], r_kpi["kpi1_val"], t["kpi1_desc"], r_kpi["trend1"], c), md=3, sm=6, className="mb-4"),
        dbc.Col(build_kpi_card(t["kpi2_title"], r_kpi["kpi2_val"], t["kpi2_desc"], r_kpi["trend2"], c), md=3, sm=6, className="mb-4"),
        dbc.Col(build_kpi_card(t["kpi3_title"], r_kpi["kpi3_val"], t["kpi3_desc"], r_kpi["trend3"], c), md=3, sm=6, className="mb-4"),
        dbc.Col(build_kpi_card(t["kpi4_title"], r_kpi["kpi4_val"], t["kpi4_desc"], r_kpi["trend4"], c), md=3, sm=6, className="mb-4"),
    ])

    # Mapa Plotly nativo con estaciones de TransMilenio
    map_content = _build_station_map(t, c)

    return html.Div([
        # ── Header con botón de navegación ──
        dbc.Row([
            dbc.Col(html.Div([
                html.H2(t["hero_title"], className="fw-bolder mb-1 mt-4", style={"color": c["text_main"]}),
                html.P(t["hero_desc"], className="mb-4 fs-6", style={"color": c["text_muted"]})
            ]), md=8),
            dbc.Col(
                html.Div([
                    dbc.Button(
                        t["btn_goto_pred"],
                        href="/prediccion",
                        color="primary",
                        className="fw-bold px-4 py-2 mt-4 shadow-sm text-end",
                        style={"backgroundColor": TM_ROJO, "borderColor": TM_ROJO, "borderRadius": "8px"}
                    )
                ], className="d-flex justify-content-md-end align-items-center h-100 text-end"),
                md=4
            )
        ]),

        # ── KPIs ──
        kpi_row,

        # ── Mapa + Tabla ──
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.Div([
                            html.H5(t["map_title"], className="fw-bold mb-0", style={"color": c["text_main"]}),
                            html.Small(t["map_desc"], style={"color": c["text_muted"]})
                        ], className="mb-3"),
                        map_content
                    ]),
                    className="shadow-sm border-0 mb-4 kpi-card",
                    style={"borderRadius": "12px", "backgroundColor": c["card_bg"]}
                ),
                md=6
            ),
            dbc.Col(
                build_dynamic_table(t, c),
                md=6
            )
        ])
    ])


def build_page_prediction(t, c):
    """Página 2: Módulo de predicción interactivo con Random Forest."""

    zonas = obtener_estaciones_disponibles()
    ist = _input_style(c)

    # ── Formulario de entrada ──
    formulario = dbc.Card(
        dbc.CardBody([
            html.Div([
                html.H5([
                    html.I(className="bi bi-sliders me-2"),
                    t["input_title"]
                ], className="fw-bold mb-0", style={"color": c["text_main"]}),
                ], className="mb-4"),

            # Zona / Línea
            _build_form_group(t["lbl_zona"], dbc.Select(
                id="pred-zona",
                options=zonas,
                value="L",
                style=ist,
                className="shadow-sm"
            ), c),

            # Fecha y hora en grid compacto
            dbc.Row([
                dbc.Col(_build_form_group(t["lbl_mes"], dbc.Input(
                    id="pred-mes", type="number", min=1, max=12, value=4,
                    style=ist, className="shadow-sm"
                ), c), md=4),
                dbc.Col(_build_form_group(t["lbl_dia"], dbc.Input(
                    id="pred-dia", type="number", min=1, max=31, value=15,
                    style=ist, className="shadow-sm"
                ), c), md=4),
                dbc.Col(_build_form_group(t["lbl_hora"], dbc.Input(
                    id="pred-hora", type="number", min=0, max=23, value=7,
                    style=ist, className="shadow-sm"
                ), c), md=4),
            ]),
            dbc.Row([
                dbc.Col(_build_form_group(t["lbl_minuto"], dbc.Input(
                    id="pred-minuto", type="number", min=0, max=59, value=30,
                    style=ist, className="shadow-sm"
                ), c), md=6),
                dbc.Col(_build_form_group(t["lbl_segundo"], dbc.Input(
                    id="pred-segundo", type="number", min=0, max=59, value=0,
                    style=ist, className="shadow-sm"
                ), c), md=6),
            ]),

            # Coordenadas
            dbc.Row([
                dbc.Col(_build_form_group(t["lbl_lat"], dbc.Input(
                    id="pred-lat", type="number", value=4.6580, step=0.0001,
                    style=ist, className="shadow-sm"
                ), c), md=6),
                dbc.Col(_build_form_group(t["lbl_lon"], dbc.Input(
                    id="pred-lon", type="number", value=-74.0940, step=0.0001,
                    style=ist, className="shadow-sm"
                ), c), md=6),
            ]),

            # Botones
            html.Div([
                dbc.Button([
                    html.I(className="bi bi-play-fill me-2"),
                    t["btn_predecir"]
                ],
                    id="btn-predecir",
                    color="danger",
                    className="fw-bold px-4 py-2 shadow-sm me-2",
                    style={"backgroundColor": TM_ROJO, "borderColor": TM_ROJO, "borderRadius": "8px"}
                ),
                dbc.Button([
                    html.I(className="bi bi-arrow-counterclockwise me-2"),
                    t["btn_limpiar"]
                ],
                    id="btn-limpiar",
                    color="secondary",
                    outline=True,
                    className="fw-bold px-4 py-2",
                    style={"borderRadius": "8px", "color": c["text_main"], "borderColor": c["border"]}
                ),
            ], className="d-flex mt-2")
        ]),
        className="shadow-sm border-0 mb-4 kpi-card",
        style={"borderRadius": "12px", "backgroundColor": c["card_bg"]}
    )

    # ── Panel de resultado ──
    resultado_panel = dbc.Card(
        dbc.CardBody([
            html.Div([
                html.H5([
                    html.I(className="bi bi-graph-up-arrow me-2"),
                    t["res_titulo"]
                ], className="fw-bold mb-0", style={"color": c["text_main"]}),
            ], className="mb-3"),
            html.Div(
                id="pred-resultado-container",
                children=_build_resultado_esperando(t, c)
            )
        ]),
        className="shadow-sm border-0 mb-4 kpi-card",
        style={
            "borderRadius": "12px",
            "backgroundColor": c["card_bg"],
            "borderTop": f"4px solid {TM_AMARILLO} !important"
        }
    )

    # ── Mapa interactivo de la troncal ──
    mapa_folium_file = os.path.join(BASE_DIR, "assets", "mapa_estaciones.html")
    if os.path.exists(mapa_folium_file):
        map_content = html.Iframe(
            src="/assets/mapa_estaciones.html",
            width="100%",
            height="650px",
            style={"border": "none", "borderRadius": "8px", "backgroundColor": c["chart_box"]}
        )
    else:
        map_content = html.Div(
            [
                html.I(className="bi bi-map text-muted mb-2", style={"fontSize": "2.5rem"}),
                html.Span(t.get("map_box", "Cargando mapa..."))
            ],
            className="d-flex flex-column align-items-center justify-content-center fw-medium rounded text-center px-4",
            style={
                "height": "650px", "backgroundColor": c["chart_box"],
                "border": f"2px dashed {c['border']}", "color": c["text_muted"]
            }
        )

    # Añadimos un Graph oculto para no romper el callback `update_lat_lon` existente
    mapa_oculto_para_callback = dcc.Graph(
        id="mapa-interactivo-prediccion",
        figure=go.Figure(),
        style={"display": "none"}
    )

    mapa_panel = dbc.Card(
        dbc.CardBody([
            html.Div([
                html.H5(t["pred_map_title"], className="fw-bold mb-0", style={"color": c["text_main"]}),
                html.Small(t["pred_map_desc"], style={"color": c["text_muted"]})
            ], className="mb-3"),
            map_content,
            mapa_oculto_para_callback
        ]),
        className="shadow-sm border-0 mb-4 kpi-card",
        style={"borderRadius": "12px", "backgroundColor": c["card_bg"]}
    )


    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Button(
                    t["btn_back_dash"],
                    href="/",
                    color="secondary",
                    outline=True,
                    className="fw-bold mb-4 mt-4 shadow-sm",
                    style={"borderRadius": "8px", "color": c["text_main"], "borderColor": c["border"]}
                ),
                html.H2(t["pred_title"], className="fw-bolder mb-1", style={"color": c["text_main"]}),
                html.P(t["pred_desc"], className="mb-4 fs-6", style={"color": c["text_muted"]})
            ])
        ]),
        dbc.Row([
            dbc.Col([
                formulario,
                resultado_panel,
            ], md=5, lg=4),

            # ── Panel derecho: Mapa de la troncal ──
            dbc.Col(
                mapa_panel,
                md=7, lg=8
            )
        ])
    ])


# ==============================================================================
# HELPERS — RESULTADO DE PREDICCIÓN
# ==============================================================================
def _build_resultado_esperando(t, c):
    """Estado inicial: esperando que el usuario ejecute la predicción."""
    return html.Div([
        html.I(className="bi bi-hourglass text-muted mb-2", style={"fontSize": "2rem"}),
        html.P(t["res_esperando"], className="fw-medium mb-0", style={"color": c["text_muted"]})
    ], className="d-flex flex-column align-items-center justify-content-center text-center py-4",
        style={
            "backgroundColor": c["chart_box"],
            "borderRadius": "8px",
            "border": f"1px dashed {c['border']}"
        }
    )


def _build_resultado_exito(resultado, t, c):
    """Muestra el resultado exitoso de la predicción."""
    return html.Div([
        # Valor principal
        html.Div([
            html.Span(
                t["res_entradas"],
                className="text-uppercase fw-bold d-block mb-1",
                style={"fontSize": "0.7rem", "letterSpacing": "0.5px", "color": c["text_muted"]}
            ),
            html.H2(
                f'{resultado["prediccion_entradas"]:,}',
                className="fw-bolder mb-0",
                style={"color": TM_ROJO, "fontSize": "2.5rem"}
            ),
            html.Small(
                f'{t["res_normalizada"]}: {resultado["prediccion_normalizada"]}',
                className="font-monospace",
                style={"color": c["text_muted"], "fontSize": "0.8rem"}
            ),
        ], className="text-center p-4 rounded mb-3", style={
            "backgroundColor": "rgba(193, 0, 31, 0.04)",
            "border": f"1px solid {TM_ROJO}30",
            "borderRadius": "10px"
        }),
    ])


def _build_resultado_error(resultado, t, c):
    """Muestra un error en la predicción."""
    return html.Div([
        html.I(className="bi bi-exclamation-triangle-fill text-warning mb-2", style={"fontSize": "2rem"}),
        html.P(t["res_error"], className="fw-bold mb-1", style={"color": c["text_main"]}),
        html.Small(resultado["mensaje"], className="font-monospace", style={"color": c["text_muted"]})
    ], className="d-flex flex-column align-items-center justify-content-center text-center py-4",
        style={
            "backgroundColor": "rgba(255, 209, 0, 0.06)",
            "borderRadius": "8px",
            "border": f"1px solid {TM_AMARILLO}60"
        }
    )


# ==============================================================================
# GENERADOR DE LAYOUT DINÁMICO
# ==============================================================================
def create_layout(lang_code, theme_mode, pathname="/"):
    """Ensambla el layout completo según idioma, tema y ruta actual."""
    t = LANGUAGES.get(lang_code, LANGUAGES["es"])
    c = THEMES[theme_mode]

    navbar = build_navbar(t, lang_code)
    footer = build_footer(t, c)

    # Routing
    if pathname == "/prediccion":
        current_content = build_page_prediction(t, c)
    else:
        current_content = build_page_dashboard(t, c)

    return html.Div([
        navbar,
        dbc.Container(current_content, fluid=True, className="px-md-5 px-3"),
        footer
    ], style={
        "backgroundColor": c["bg_base"],
        "minHeight": "100vh",
        "fontFamily": "'Inter', 'Roboto', sans-serif"
    })


# ==============================================================================
# ROOT LAYOUT Y CALLBACKS
# ==============================================================================
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    dcc.Store(id="current-lang", data="es"),
    dcc.Store(id="current-theme", data="light"),
    html.Div(id="app-wrapper", children=create_layout("es", "light", "/"))
])


@app.callback(
    Output("app-wrapper", "children"),
    Input("lang-select", "value"),
    Input("theme-switch", "value"),
    Input("url", "pathname"),
    prevent_initial_call=False
)
def update_dashboard(lang, is_dark, pathname):
    """Callback maestro: reconstruye el layout al cambiar idioma, tema o ruta."""
    theme = "dark" if is_dark else "light"
    lang = lang if lang else "es"
    pathname = pathname if pathname else "/"
    return create_layout(lang, theme, pathname)


# ==============================================================================
# CALLBACK — PREDICCIÓN
# ==============================================================================
@app.callback(
    Output("pred-resultado-container", "children"),
    Input("btn-predecir", "n_clicks"),
    Input("btn-limpiar", "n_clicks"),
    State("pred-zona", "value"),
    State("pred-mes", "value"),
    State("pred-dia", "value"),
    State("pred-hora", "value"),
    State("pred-minuto", "value"),
    State("pred-segundo", "value"),
    State("pred-lat", "value"),
    State("pred-lon", "value"),
    State("lang-select", "value"),
    State("theme-switch", "value"),
    prevent_initial_call=True
)
def ejecutar_prediccion(
    n_pred, n_limpiar,
    zona, mes, dia, hora, minuto, segundo, lat, lon,
    lang, is_dark
):
    """Callback de predicción: ejecuta el modelo o restablece el panel."""
    t = LANGUAGES.get(lang or "es", LANGUAGES["es"])
    c = THEMES["dark" if is_dark else "light"]

    triggered = callback_context.triggered_id

    # ── Restablecer ──
    if triggered == "btn-limpiar":
        return _build_resultado_esperando(t, c)

    # ── Predecir ──
    if triggered == "btn-predecir":
        try:
            resultado = predecir_afluencia(
                mes=int(mes or 4),
                dia=int(dia or 1),
                hora=int(hora or 7),
                minuto=int(minuto or 0),
                segundo=int(segundo or 0),
                letra_zona=str(zona or "A"),
                lat=float(lat or 4.6097),
                lon=float(lon or -74.0817),
            )

            if resultado["exito"]:
                return _build_resultado_exito(resultado, t, c)
            else:
                return _build_resultado_error(resultado, t, c)

        except Exception as e:
            error_result = {
                "exito": False,
                "mensaje": f"Error inesperado: {str(e)}"
            }
            return _build_resultado_error(error_result, t, c)

    return no_update

@app.callback(
    [Output("pred-lat", "value"), Output("pred-lon", "value")],
    Input("mapa-interactivo-prediccion", "clickData"),
    prevent_initial_call=True
)
def update_lat_lon(clickData):
    """Actualiza los inputs de coordenadas al hacer clic en el mapa."""
    if clickData and "points" in clickData:
        point = clickData["points"][0]
        return point.get("lat", no_update), point.get("lon", no_update)
    return no_update, no_update


# ==============================================================================
# PUNTO DE ENTRADA
# ==============================================================================
if __name__ == "__main__":
    app.run(debug=True, port=8050)
