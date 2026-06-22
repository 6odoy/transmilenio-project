"""
TransMilenio — Dashboard Operativo de Análisis y Simulación
=============================================================
Aplicación principal construida con Dash + Plotly.
Incluye módulo de predicción con Random Forest integrado.
Configuración centralizada en config.py.
"""

import json
import os
import time

from charts_causal import (
    build_campin_event_study,
    build_campin_fdv_comparacion,
    build_campin_por_evento,
    build_campin_spillover,
    build_ciclovia_mapa,
    build_ciclovia_ratio,
    build_ciclovia_resultados,
    build_clima_coeficientes,
    build_clima_simulacion,
    build_combustible_especificaciones,
    build_combustible_serie,
    build_efectos_individuales,
    build_event_study_main,
    build_fdv_tipo_dia,
    build_sc_losses,
    build_sc_rmspe,
    build_sc_weights,
    build_subgrupos,
    build_timeline_festivos,
)
import dash
from dash import Input, Output, dcc, html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from config import BASE_DIR, LANGUAGES, THEMES, TM_AMARILLO, TM_ROJO

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
        if not isinstance(nombre, str):
            return nombre
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
    """Carga estaciones del GeoJSON. Retorna (lats, lons, names, colors, lines_data, customdata).

    lines_data: dict {linea: {"color": str, "lats": [...], "lons": [...]}}
    customdata: list of [nombre_linea, promedio_diario_fmt, total_entradas_fmt] per station
    """
    from config import GEOJSON_PATH, TM_ROJO
    lats, lons, names, colors, customdata = [], [], [], [], []

    color_file = os.path.join(BASE_DIR, "models", "artefactos", "estacion_color.json")
    estacion_colors = {}
    try:
        if os.path.exists(color_file):
            with open(color_file, "r", encoding="utf-8") as f:
                estacion_colors = json.load(f)
    except Exception:
        pass

    stats_file = os.path.join(BASE_DIR, "models", "artefactos", "estacion_stats.json")
    estacion_stats = {}
    try:
        if os.path.exists(stats_file):
            with open(stats_file, "r", encoding="utf-8") as f:
                estacion_stats = json.load(f)
    except Exception:
        pass

    _le_artefactos = os.path.join(BASE_DIR, "models", "artefactos", "linea_estaciones.json")
    _le_repo = os.path.join(BASE_DIR, "..", "..", "data", "processed", "linea_estaciones.json")
    linea_estaciones_file = _le_artefactos if os.path.exists(_le_artefactos) else _le_repo
    linea_estaciones = {}
    try:
        if os.path.exists(linea_estaciones_file):
            with open(linea_estaciones_file, "r", encoding="utf-8") as f:
                linea_estaciones = json.load(f)
    except Exception:
        pass

    # Índice nodo → coordenadas
    nodo_coords: dict = {}
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
                        nodo_coords[nodo] = (coords[1], coords[0])  # (lat, lon)
                        lons.append(coords[0])
                        lats.append(coords[1])
                        names.append(props.get("nombre_estacion", "Estación"))
                        color_asignado = estacion_colors.get(nodo, {}).get("color", TM_ROJO)
                        colors.append(color_asignado)

                        s = estacion_stats.get(nodo, {})
                        prom = s.get("promedio_diario")
                        total = s.get("total_entradas")
                        prom_fmt = f"{prom:,}".replace(",", ".") if prom is not None else "N/D"
                        total_fmt = f"{total:,}".replace(",", ".") if total is not None else "N/D"
                        customdata.append([s.get("nombre_linea", ""), prom_fmt, total_fmt])
    except Exception:
        pass

    # Construye trazos por línea respetando el orden del JSON (lista enlazada)
    lines_data: dict = {}
    for linea, nodos in linea_estaciones.items():
        puntos = []
        for n in nodos:
            coords = nodo_coords.get(str(n))
            if coords:
                puntos.append(coords)
        if len(puntos) < 2:
            continue
        color = estacion_colors.get(str(nodos[0]), {}).get("color", TM_ROJO)
        lines_data[linea] = {
            "color": color,
            "lats": [p[0] for p in puntos],
            "lons": [p[1] for p in puntos],
        }

    return lats, lons, names, colors, lines_data, customdata

def _build_station_map(t, c):
    """Construye el mapa de estaciones con líneas suaves por troncal."""
    lats, lons, names, colors, lines_data, customdata = _load_geojson_stations()

    if not lats:
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

    # ── Trazos de líneas (uno por troncal) ──
    traces = []
    for linea, ld in lines_data.items():
        traces.append(go.Scattermap(
            lat=ld["lats"],
            lon=ld["lons"],
            mode="lines",
            line=dict(width=2.5, color=ld["color"]),
            opacity=0.55,
            hoverinfo="skip",
            showlegend=False,
        ))

    # ── Puntos de estaciones encima ──
    traces.append(go.Scattermap(
        lat=lats,
        lon=lons,
        mode="markers",
        marker=go.scattermap.Marker(
            size=9,
            color=colors if colors else TM_ROJO,
            opacity=0.9,
        ),
        text=names,
        customdata=customdata if customdata else None,
        hovertemplate=(
            "<b>%{text}</b><br>"
            "%{customdata[0]}<br>"
            "Prom. diario: %{customdata[1]} entradas<br>"
            "Total: %{customdata[2]} entradas"
            "<extra></extra>"
        ) if customdata else "<b>%{text}</b><extra></extra>",
        hoverlabel=dict(
            font_size=12,
            font_family="Inter, sans-serif",
            font_color="white",
            bgcolor=colors if colors else TM_ROJO,
        ),
        showlegend=False,
    ))

    fig = go.Figure(traces)

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


def build_page_causal(t, c, lang_code="es"):
    """Página 2: Módulo de inferencia causal con resultados por análisis."""

    # ── Helpers locales ──────────────────────────────────────────────────────
    def fig_card(src, caption):
        return dbc.Card(
            dbc.CardBody([
                html.Img(src=src, style={"width": "100%", "height": "auto", "borderRadius": "6px"}),
                html.Small(caption, className="d-block text-center mt-2",
                           style={"color": c["text_muted"], "fontSize": "0.78rem", "lineHeight": "1.4"})
            ]),
            className="shadow-sm border-0 mb-3",
            style={"borderRadius": "10px", "backgroundColor": c["card_bg"]}
        )

    def finding_banner(badge_key, badge_color, title_key, method_key, finding_key, border_hex):
        return dbc.Card(
            dbc.CardBody(
                dbc.Row([
                    dbc.Col([
                        dbc.Badge(t[badge_key], color=badge_color,
                                  className="mb-2 px-3 py-2",
                                  style={"fontSize": "0.8rem", "fontWeight": "600"}),
                        html.H5(t[title_key], className="fw-bold mb-1 mt-1",
                                style={"color": c["text_main"]}),
                        html.Code(t[method_key],
                                  style={"fontSize": "0.78rem", "color": c["text_muted"],
                                         "whiteSpace": "pre-wrap"}),
                    ], md=5, className="mb-2"),
                    dbc.Col(
                        html.Div([
                            html.I(className="bi bi-lightbulb-fill me-2",
                                   style={"color": TM_AMARILLO, "fontSize": "1rem",
                                          "flexShrink": "0"}),
                            html.Span(t[finding_key],
                                      style={"color": c["text_main"], "fontSize": "0.88rem",
                                             "lineHeight": "1.5"})
                        ], className="d-flex align-items-start p-3 h-100",
                           style={"backgroundColor": "rgba(255,209,0,0.07)",
                                  "border": f"1px solid {TM_AMARILLO}55",
                                  "borderRadius": "8px"}),
                        md=7, className="mb-2"
                    ),
                ], className="g-2 align-items-stretch")
            ),
            className="shadow-sm border-0 mb-3",
            style={"borderRadius": "12px", "backgroundColor": c["card_bg"],
                   "borderLeft": f"4px solid {border_hex}"}
        )

    # ── Tab 1: Festivos ──────────────────────────────────────────────────────
    _graph_cfg = {"displayModeBar": "hover", "scrollZoom": False,
                  "modeBarButtonsToRemove": ["lasso2d", "select2d"]}

    def graph_card(fig, height=430):
        return dbc.Card(
            dbc.CardBody(
                dcc.Graph(figure=fig, config=_graph_cfg,
                          style={"height": f"{height}px"}, responsive=True)
            ),
            className="shadow-sm border-0 mb-3",
            style={"borderRadius": "10px", "backgroundColor": c["card_bg"]}
        )

    tab_festivos = html.Div([
        finding_banner("badge_confirmado", "success",
                       "festivos_titulo", "festivos_metodo", "festivos_hallazgo", "#198754"),
        dbc.Row([
            dbc.Col(graph_card(build_event_study_main(c, lang_code)), md=6),
            dbc.Col(graph_card(build_efectos_individuales(c, lang_code), height=530), md=6),
        ]),
        dbc.Row([
            dbc.Col(graph_card(build_subgrupos(c, lang_code)), md=6),
            dbc.Col(graph_card(build_timeline_festivos(c, lang_code)), md=6),
        ]),
    ], className="pt-3")

    # ── Tab 2: Clima ─────────────────────────────────────────────────────────
    tab_clima = html.Div([
        finding_banner("badge_refutado", "danger",
                       "clima_titulo", "clima_metodo", "clima_hallazgo", TM_ROJO),
        dbc.Row([
            dbc.Col(graph_card(build_clima_coeficientes(c, lang_code)), md=6),
            dbc.Col(graph_card(build_clima_simulacion(c, lang_code)), md=6),
        ]),
    ], className="pt-3")

    # ── Tab 3: Ciclovía ──────────────────────────────────────────────────────
    tab_ciclovia = html.Div([
        finding_banner("badge_no_confirmado", "warning",
                       "ciclovia_titulo", "ciclovia_metodo", "ciclovia_hallazgo", TM_AMARILLO),
        graph_card(build_ciclovia_mapa(c, lang_code), height=480),
        dbc.Row([
            dbc.Col(graph_card(build_ciclovia_ratio(c, lang_code)), md=6),
            dbc.Col(graph_card(build_ciclovia_resultados(c, lang_code)), md=6),
        ]),
    ], className="pt-3")

    # ── Tab 4: Conciertos ────────────────────────────────────────────────────
    tab_campin = html.Div([
        finding_banner("badge_confirmado", "success",
                       "campin_titulo", "campin_metodo", "campin_hallazgo", "#198754"),
        dbc.Row([
            dbc.Col(graph_card(build_campin_event_study(c, lang_code)), md=7),
            dbc.Col(graph_card(build_campin_por_evento(c, lang_code), height=430), md=5),
        ]),
        dbc.Row([
            dbc.Col(graph_card(build_campin_spillover(c, lang_code)), md=6),
            dbc.Col(graph_card(build_campin_fdv_comparacion(c, lang_code)), md=6),
        ]),
        graph_card(build_fdv_tipo_dia(c, lang_code), height=380),
    ], className="pt-3")

    # ── Tab 5: Combustible ───────────────────────────────────────────────────
    tab_combustible = html.Div([
        finding_banner("badge_inconcluso", "warning",
                       "combustible_titulo", "combustible_metodo", "combustible_hallazgo",
                       TM_AMARILLO),
        dbc.Row([
            dbc.Col(graph_card(build_combustible_serie(c, lang_code)), md=7),
            dbc.Col(graph_card(build_combustible_especificaciones(c, lang_code)), md=5),
        ]),
    ], className="pt-3")

    # ── Tab 6: Control Sintético ───────────────────────────────────────────
    tab_synth = html.Div([
        finding_banner("badge_confirmado", "success",
                       "synth_titulo", "synth_metodo", "synth_hallazgo", "#198754"),
        dbc.Row([
            dbc.Col(graph_card(build_sc_losses(c, lang_code), height=430), md=7),
            dbc.Col(graph_card(build_sc_weights(c, lang_code), height=430), md=5),
        ]),
        graph_card(build_sc_rmspe(c, lang_code), height=400),
    ], className="pt-3")

    # ── Ensamble de tabs ─────────────────────────────────────────────────────
    tabs = dbc.Tabs([
        dbc.Tab(tab_festivos,     label=t["tab_festivos"],     tab_id="tab-festivos"),
        dbc.Tab(tab_clima,        label=t["tab_clima"],        tab_id="tab-clima"),
        dbc.Tab(tab_ciclovia,     label=t["tab_ciclovia"],     tab_id="tab-ciclovia"),
        dbc.Tab(tab_campin,       label=t["tab_campin"],       tab_id="tab-campin"),
        dbc.Tab(tab_combustible,  label=t["tab_combustible"],  tab_id="tab-combustible"),
        dbc.Tab(tab_synth,        label=t["tab_synth"],        tab_id="tab-synth"),
    ], active_tab="tab-festivos", className="mb-0",
       style={"borderBottom": f"2px solid {TM_ROJO}"})

    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Button(t["btn_back_dash"], href="/", color="secondary", outline=True,
                           className="fw-bold mb-4 mt-4 shadow-sm",
                           style={"borderRadius": "8px", "color": c["text_main"],
                                  "borderColor": c["border"]}),
                html.H2(t["pred_title"], className="fw-bolder mb-1",
                        style={"color": c["text_main"]}),
                html.P(t["pred_desc"], className="mb-3 fs-6",
                       style={"color": c["text_muted"]})
            ])
        ]),
        tabs,
    ])



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
        current_content = build_page_causal(t, c, lang_code)
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
# PUNTO DE ENTRADA
# ==============================================================================
if __name__ == "__main__":
    app.run(debug=True, port=8050)
