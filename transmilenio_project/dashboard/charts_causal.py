"""
Gráficas interactivas de inferencia causal — TransMilenio 2025.
Cada función recibe el dict de tema `c` y devuelve un go.Figure listo
para usar en dcc.Graph.  Todos los valores provienen de
docs/resultados_inferencia_causal.md.
"""

import json
import math
import os

import plotly.graph_objects as go

from config import TM_AMARILLO, TM_ROJO

_DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT  = os.path.dirname(os.path.dirname(_DASHBOARD_DIR))
_EXT_DIR       = os.path.join(_PROJECT_ROOT, "data", "external")

# ── Paleta semántica ─────────────────────────────────────────────────────────
_C_SIG      = TM_ROJO       # coeficiente significativo / confirmado
_C_INSIG    = "#9CA3AF"     # no significativo
_C_LUNES    = "#3B82F6"     # festivos lunes Emiliani
_C_MIDWEEK  = TM_ROJO       # festivos mitad de semana
_C_DOMINGO  = "#10B981"     # festivos domingo (anomalía)
_C_ZERO     = "#6B7280"     # línea de referencia cero
_C_CI_MAIN  = "rgba(193,0,31,0.12)"
_C_CI_LUNES = "rgba(59,130,246,0.12)"
_C_CI_MID   = "rgba(193,0,31,0.12)"

# ── Dataset: Event Study principal ──────────────────────────────────────────
# Fuente: resultados_inferencia_causal.md — Análisis 1
_ES_K      = [-5,    -4,    -3,    -2,     0,     1,     2,     3,     4,     5]
_ES_BETA   = [-0.058,-0.051,-0.057,-0.082,-1.128,-0.129,-0.083,-0.039,-0.040,-0.029]
_ES_CI_LO  = [-0.156,-0.137,-0.115,-0.136,-1.326,-0.224,-0.168,-0.087,-0.088,-0.086]
_ES_CI_HI  = [ 0.040, 0.035, 0.001,-0.028,-0.930,-0.033, 0.002, 0.009, 0.009, 0.029]
_ES_PCT    = [-5.7,  -5.0,  -5.5,  -7.8, -67.6, -12.1,  -8.0,  -3.8,  -3.9,  -2.8]
_ES_SIG    = [False, False,  True,  True,  True,  True,  True, False, False, False]
_ES_STARS  = ["",    "",   "(*)", "(***)", "(***)", "(***)", "(*)", "", "", ""]

# ── Dataset: Subgrupos (Lunes Emiliani vs. Mitad de semana) ─────────────────
# Fuente: resultados_inferencia_causal.md — Análisis 1b
_SG_K          = [-5,   -4,   -3,   -2,    0,     1,     2,     3,    4,    5]
_SG_LUNES_PCT  = [-6.8, -9.7, -6.2, -7.9, -68.1,  -1.5,  1.7,  -2.0, -3.5, -1.2]
_SG_MID_PCT    = [-2.1,  2.4, -7.0, -8.6, -72.6, -22.2,-18.7,  -5.9, -5.5, -4.3]
_SG_LUNES_SIG  = [False, True,  True, True,  True, False, False, False, True, False]
_SG_MID_SIG    = [False, False, False, True, True,  True,  True, False, False, False]

# ── Dataset: Efectos individuales por festivo ────────────────────────────────
# Valores confirmados desde resultados_inferencia_causal.md; estimados (~) para el resto
_HOLIDAYS = [
    # (fecha, nombre_corto, día_semana, pct_cambio, tipo)
    ("2025-01-01", "Año Nuevo",               "Mié",  -83, "midweek"),
    ("2025-12-25", "Navidad",                 "Jue",  -80, "midweek"),
    ("2025-04-18", "Viernes Santo",           "Vie",  -76, "midweek"),
    ("2025-01-06", "Reyes Magos",             "Lun",  -73, "lunes"),
    ("2025-04-17", "Jueves Santo",            "Jue",  -71, "midweek"),
    ("2025-10-13", "Día de la Raza",          "Lun",  -69, "lunes"),
    ("2025-06-23", "Corpus Christi",          "Lun",  -69, "lunes"),
    ("2025-08-18", "La Asunción",             "Lun",  -69, "lunes"),
    ("2025-03-24", "San José",                "Lun",  -69, "lunes"),
    ("2025-08-07", "Batalla de Boyacá",       "Jue",  -69, "midweek"),
    ("2025-06-02", "Ascensión del Señor",     "Lun",  -67, "lunes"),
    ("2025-06-30", "San Pedro y San Pablo",   "Lun",  -66, "lunes"),
    ("2025-11-17", "Indep. de Cartagena",     "Lun",  -66, "lunes"),
    ("2025-12-08", "La Inmaculada",           "Lun",  -65, "lunes"),
    ("2025-05-01", "Día del Trabajo",         "Jue",  -65, "midweek"),
    ("2025-11-03", "Día de los Difuntos",     "Lun",  -55, "lunes"),
    ("2025-07-20", "Independencia Nacional",  "Dom",   +5, "domingo"),
]


# ── Utilidades ───────────────────────────────────────────────────────────────

def _theme(fig: go.Figure, c: dict) -> go.Figure:
    """Aplica el tema del dashboard a una figura Plotly."""
    fig.update_layout(
        paper_bgcolor=c["card_bg"],
        plot_bgcolor=c["chart_box"],
        font=dict(family="Inter, Roboto, sans-serif", color=c["text_main"], size=12),
        margin=dict(l=8, r=8, t=36, b=8),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            font=dict(size=11),
        ),
        hoverlabel=dict(
            bgcolor=c["card_bg"],
            bordercolor=c["border"],
            font=dict(color=c["text_main"], family="Inter, Roboto, sans-serif", size=12),
        ),
    )
    fig.update_xaxes(
        gridcolor=c["border"],
        zerolinecolor=c["border"],
        tickfont=dict(color=c["text_muted"]),
        title_font=dict(color=c["text_muted"], size=11),
        linecolor=c["border"],
    )
    fig.update_yaxes(
        gridcolor=c["border"],
        zerolinecolor=c["border"],
        tickfont=dict(color=c["text_muted"]),
        title_font=dict(color=c["text_muted"], size=11),
        linecolor=c["border"],
    )
    return fig


def _sig_color(is_sig: bool) -> str:
    return _C_SIG if is_sig else _C_INSIG


def _tipo_color(tipo: str) -> str:
    return {"lunes": _C_LUNES, "midweek": _C_MIDWEEK, "domingo": _C_DOMINGO}.get(tipo, _C_INSIG)


# ── Gráfica 1: Event Study Principal ────────────────────────────────────────

def build_event_study_main(c: dict, lang: str = "es") -> go.Figure:
    """
    Coeficientes β_k del event study con banda IC 95%.
    Baseline k = −1 aparece como diamante amarillo en y = 0.
    """
    labels = {
        "es": {"x": "Días relativos al festivo (k)", "y": "Coeficiente β",
               "title": "Event Study — Efecto de Festivos sobre log(demanda)",
               "hover_k": "k", "hover_beta": "β", "hover_ci": "IC 95%",
               "hover_efecto": "Efecto", "hover_base": "baseline", "festivo": "Día del festivo",
               "pretend": "Pre-tendencia"},
        "en": {"x": "Days relative to holiday (k)", "y": "Coefficient β",
               "title": "Event Study — Holiday Effect on log(demand)",
               "hover_k": "k", "hover_beta": "β", "hover_ci": "95% CI",
               "hover_efecto": "Effect", "hover_base": "baseline", "festivo": "Holiday day",
               "pretend": "Pre-trend"},
        "it": {"x": "Giorni relativi alla festività (k)", "y": "Coefficiente β",
               "title": "Event Study — Effetto delle Festività su log(domanda)",
               "hover_k": "k", "hover_beta": "β", "hover_ci": "IC 95%",
               "hover_efecto": "Effetto", "hover_base": "baseline", "festivo": "Giorno festivo",
               "pretend": "Pre-tendenza"},
    }
    L = labels.get(lang, labels["es"])

    # Extendemos k para incluir -1 como baseline en la banda CI
    k_full = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    ci_lo_full = [-0.156, -0.137, -0.115, -0.136, 0, -1.326, -0.224, -0.168, -0.087, -0.088, -0.086]
    ci_hi_full = [0.040, 0.035, 0.001, -0.028, 0, -0.930, -0.033, 0.002, 0.009, 0.009, 0.029]

    fig = go.Figure()

    # ── Banda IC 95% ──
    fig.add_trace(go.Scatter(
        x=k_full, y=ci_lo_full,
        mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
        fillcolor=_C_CI_MAIN, fill=None,
    ))
    fig.add_trace(go.Scatter(
        x=k_full, y=ci_hi_full,
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor=_C_CI_MAIN,
        showlegend=False, hoverinfo="skip",
        name="IC 95%",
    ))

    # ── Línea conectora ──
    fig.add_trace(go.Scatter(
        x=_ES_K, y=_ES_BETA,
        mode="lines",
        line=dict(color=_C_SIG, width=1.5, dash="dot"),
        showlegend=False, hoverinfo="skip",
    ))

    # ── Puntos con hover ──
    colors = [_sig_color(s) for s in _ES_SIG]
    sizes  = [13 if s else 9 for s in _ES_SIG]
    hover  = [
        f"<b>{L['hover_k']} = {k:+d}</b><br>"
        f"{L['hover_beta']} = {b:.3f}<br>"
        f"{L['hover_ci']}: [{lo:.3f}, {hi:.3f}]<br>"
        f"{L['hover_efecto']}: <b>{p:+.1f}%</b> {st}"
        for k, b, lo, hi, p, st in zip(_ES_K, _ES_BETA, _ES_CI_LO, _ES_CI_HI, _ES_PCT, _ES_STARS)
    ]
    fig.add_trace(go.Scatter(
        x=_ES_K, y=_ES_BETA,
        mode="markers",
        marker=dict(
            size=sizes, color=colors,
            line=dict(color=colors, width=1.5),
            symbol="circle",
        ),
        customdata=list(zip(_ES_CI_LO, _ES_CI_HI, _ES_PCT, _ES_STARS)),
        hovertext=hover, hoverinfo="text",
        showlegend=False,
    ))

    # ── Baseline k = −1 ──
    fig.add_trace(go.Scatter(
        x=[-1], y=[0],
        mode="markers",
        marker=dict(size=13, color=TM_AMARILLO, symbol="diamond",
                    line=dict(color=TM_ROJO, width=2)),
        hovertext=f"<b>k = −1 ({L['hover_base']})</b><br>β = 0",
        hoverinfo="text", showlegend=False,
    ))

    # ── Decoraciones ──
    fig.add_hline(y=0, line_color=c["border"], line_width=1.5, line_dash="solid")

    # Sombreado día del festivo
    fig.add_vrect(
        x0=-0.45, x1=0.45,
        fillcolor="rgba(193,0,31,0.07)", line_width=0,
        annotation_text=f"<b>{L['festivo']}</b>",
        annotation_position="top",
        annotation_font=dict(color=TM_ROJO, size=10),
    )
    # Región pre-tendencia
    fig.add_vrect(
        x0=-5.45, x1=-1.55,
        fillcolor="rgba(255,209,0,0.06)", line_width=0,
        annotation_text=L["pretend"],
        annotation_position="top left",
        annotation_font=dict(color=c["text_muted"], size=9),
    )

    fig.update_xaxes(
        tickvals=list(range(-5, 6)),
        ticktext=[str(k) for k in range(-5, 6)],
        title_text=L["x"], range=[-5.7, 5.7],
    )
    fig.update_yaxes(title_text=L["y"])
    fig.update_layout(title=dict(text=L["title"], font=dict(size=13, color=c["text_main"])))

    return _theme(fig, c)


# ── Gráfica 2: Efectos Individuales (barras horizontales) ────────────────────

def build_efectos_individuales(c: dict, lang: str = "es") -> go.Figure:
    """
    Barras horizontales: efecto de cada festivo vs. mediana del mismo día de semana.
    Color por tipo: lunes Emiliani / mitad de semana / domingo.
    """
    labels = {
        "es": {"title": "Efecto Individual por Festivo", "x": "Cambio vs. mediana del mismo día (%)",
               "lunes": "Lunes Emiliani", "midweek": "Mitad de semana", "domingo": "Domingo",
               "hover_dia": "Día", "hover_tipo": "Tipo", "hover_fecha": "Fecha"},
        "en": {"title": "Individual Effect per Holiday", "x": "Change vs. same-weekday median (%)",
               "lunes": "Monday (Emiliani)", "midweek": "Midweek", "domingo": "Sunday",
               "hover_dia": "Day", "hover_tipo": "Type", "hover_fecha": "Date"},
        "it": {"title": "Effetto Individuale per Festività", "x": "Variazione vs. mediana stesso giorno (%)",
               "lunes": "Lunedì (Emiliani)", "midweek": "Metà settimana", "domingo": "Domenica",
               "hover_dia": "Giorno", "hover_tipo": "Tipo", "hover_fecha": "Data"},
    }
    L = labels.get(lang, labels["es"])

    # Ordenar por efecto (más negativo arriba)
    sorted_h = sorted(_HOLIDAYS, key=lambda x: x[3])

    names  = [h[1] for h in sorted_h]
    pcts   = [h[3] for h in sorted_h]
    tipos  = [h[4] for h in sorted_h]
    fechas = [h[0] for h in sorted_h]
    dias   = [h[2] for h in sorted_h]
    hover = [
        f"<b>{n}</b><br>"
        f"{L['hover_fecha']}: {f}<br>"
        f"{L['hover_dia']}: {d}<br>"
        f"{L['hover_tipo']}: {L.get(t, t)}<br>"
        f"Efecto: <b>{p:+.0f}%</b>"
        for n, f, d, t, p in zip(names, fechas, dias, tipos, pcts)
    ]

    fig = go.Figure()

    # Barras por tipo (3 trazas para leyenda)
    for tipo, label_key, color in [("lunes", "lunes", _C_LUNES),
                                    ("midweek", "midweek", _C_MIDWEEK),
                                    ("domingo", "domingo", _C_DOMINGO)]:
        idxs = [i for i, t in enumerate(tipos) if t == tipo]
        if not idxs:
            continue
        fig.add_trace(go.Bar(
            x=[pcts[i] for i in idxs],
            y=[names[i] for i in idxs],
            orientation="h",
            marker=dict(
                color=color,
                opacity=[1.0 if pcts[i] != -55 else 0.7 for i in idxs],  # Difuntos más opaco
            ),
            hovertext=[hover[i] for i in idxs],
            hoverinfo="text",
            name=L[label_key],
        ))

    fig.add_vline(x=0, line_color=c["border"], line_width=1.5)

    fig.update_xaxes(title_text=L["x"], ticksuffix="%")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        title=dict(text=L["title"], font=dict(size=13, color=c["text_main"])),
        barmode="overlay",
        bargap=0.25,
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
    )

    return _theme(fig, c)


# ── Gráfica 3: Subgrupos — Lunes Emiliani vs. Mitad de semana ───────────────

def build_subgrupos(c: dict, lang: str = "es") -> go.Figure:
    """
    Doble línea con marcadores: perfiles de recuperación por tipo de festivo.
    Anotaciones en k=+1 y k=+2 donde la diferencia es significativa.
    """
    labels = {
        "es": {"title": "Perfil de Recuperación por Tipo de Festivo",
               "x": "Días relativos al festivo (k)",
               "y": "Cambio estimado (%)",
               "lunes": "Lunes Emiliani (n=10)",
               "midweek": "Mitad de semana (n=6)",
               "annot_k1": "Diferencia<br>+20.7 pp **",
               "annot_k2": "Diferencia<br>+20.4 pp ***",
               "hover_k": "k", "hover_pct": "Efecto",
               "festivo": "Día del festivo"},
        "en": {"title": "Recovery Profile by Holiday Type",
               "x": "Days relative to holiday (k)",
               "y": "Estimated change (%)",
               "lunes": "Monday Emiliani (n=10)",
               "midweek": "Midweek (n=6)",
               "annot_k1": "Gap<br>+20.7 pp **",
               "annot_k2": "Gap<br>+20.4 pp ***",
               "hover_k": "k", "hover_pct": "Effect",
               "festivo": "Holiday day"},
        "it": {"title": "Profilo di Recupero per Tipo di Festività",
               "x": "Giorni relativi alla festività (k)",
               "y": "Variazione stimata (%)",
               "lunes": "Lunedì Emiliani (n=10)",
               "midweek": "Metà settimana (n=6)",
               "annot_k1": "Differenza<br>+20.7 pp **",
               "annot_k2": "Differenza<br>+20.4 pp ***",
               "hover_k": "k", "hover_pct": "Effetto",
               "festivo": "Giorno festivo"},
    }
    L = labels.get(lang, labels["es"])

    def _hover(k_list, pct_list, sig_list, label):
        return [
            f"<b>{label}</b><br>"
            f"{L['hover_k']} = {k:+d}<br>"
            f"{L['hover_pct']}: <b>{p:+.1f}%</b>"
            f"{'  ✓ sig.' if s else ''}"
            for k, p, s in zip(k_list, pct_list, sig_list)
        ]

    fig = go.Figure()

    # ── Lunes Emiliani ──
    marker_lunes = dict(
        size=[11 if s else 8 for s in _SG_LUNES_SIG],
        color=[_C_LUNES if s else "rgba(59,130,246,0.45)" for s in _SG_LUNES_SIG],
        symbol=["circle" if s else "circle-open" for s in _SG_LUNES_SIG],
        line=dict(color=_C_LUNES, width=1.5),
    )
    fig.add_trace(go.Scatter(
        x=_SG_K, y=_SG_LUNES_PCT,
        mode="lines+markers",
        line=dict(color=_C_LUNES, width=2.5),
        marker=marker_lunes,
        hovertext=_hover(_SG_K, _SG_LUNES_PCT, _SG_LUNES_SIG, L["lunes"]),
        hoverinfo="text",
        name=L["lunes"],
    ))

    # ── Mitad de semana ──
    marker_mid = dict(
        size=[11 if s else 8 for s in _SG_MID_SIG],
        color=[_C_MIDWEEK if s else "rgba(193,0,31,0.45)" for s in _SG_MID_SIG],
        symbol=["circle" if s else "circle-open" for s in _SG_MID_SIG],
        line=dict(color=_C_MIDWEEK, width=1.5),
    )
    fig.add_trace(go.Scatter(
        x=_SG_K, y=_SG_MID_PCT,
        mode="lines+markers",
        line=dict(color=_C_MIDWEEK, width=2.5),
        marker=marker_mid,
        hovertext=_hover(_SG_K, _SG_MID_PCT, _SG_MID_SIG, L["midweek"]),
        hoverinfo="text",
        name=L["midweek"],
    ))

    # ── Decoraciones ──
    fig.add_hline(y=0, line_color=c["border"], line_width=1.5, line_dash="solid")
    fig.add_vrect(
        x0=-0.45, x1=0.45,
        fillcolor="rgba(193,0,31,0.07)", line_width=0,
        annotation_text=f"<b>{L['festivo']}</b>",
        annotation_position="top",
        annotation_font=dict(color=TM_ROJO, size=10),
    )

    # Anotaciones de diferencia significativa en k=+1 y k=+2
    for k_ann, ann_text in [(1, L["annot_k1"]), (2, L["annot_k2"])]:
        y_lunes = _SG_LUNES_PCT[_SG_K.index(k_ann)]
        y_mid   = _SG_MID_PCT[_SG_K.index(k_ann)]
        y_mid_ann = (y_lunes + y_mid) / 2
        fig.add_annotation(
            x=k_ann, y=y_mid_ann,
            text=ann_text,
            showarrow=False,
            font=dict(size=9, color=TM_AMARILLO),
            bgcolor=c["card_bg"],
            bordercolor=TM_AMARILLO,
            borderwidth=1,
            borderpad=3,
            xshift=40,
        )
        # Bracket vertical entre los dos puntos
        fig.add_shape(
            type="line",
            x0=k_ann, x1=k_ann,
            y0=y_lunes, y1=y_mid,
            line=dict(color=TM_AMARILLO, width=1.5, dash="dot"),
        )

    fig.update_xaxes(
        tickvals=list(range(-5, 6)),
        ticktext=[str(k) for k in range(-5, 6)],
        title_text=L["x"], range=[-5.7, 5.7],
    )
    fig.update_yaxes(title_text=L["y"], ticksuffix="%")
    fig.update_layout(
        title=dict(text=L["title"], font=dict(size=13, color=c["text_main"])),
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
    )

    return _theme(fig, c)


# ── Gráfica 4: Vista temporal — Festivos en el calendario ────────────────────

def build_timeline_festivos(c: dict, lang: str = "es") -> go.Figure:
    """
    Scatter plot: fecha en eje X, efecto (%) en eje Y.
    Tamaño del marcador proporcional a la magnitud del efecto.
    Color por tipo. Vista de calendario del año completo.
    """
    labels = {
        "es": {"title": "Impacto de Festivos en el Calendario 2025",
               "x": "Fecha", "y": "Cambio en demanda (%)",
               "lunes": "Lunes Emiliani", "midweek": "Mitad de semana", "domingo": "Domingo",
               "hover_fecha": "Fecha", "hover_dia": "Día", "hover_tipo": "Tipo"},
        "en": {"title": "Holiday Impact Across the 2025 Calendar",
               "x": "Date", "y": "Demand change (%)",
               "lunes": "Monday (Emiliani)", "midweek": "Midweek", "domingo": "Sunday",
               "hover_fecha": "Date", "hover_dia": "Day", "hover_tipo": "Type"},
        "it": {"title": "Impatto delle Festività nel Calendario 2025",
               "x": "Data", "y": "Variazione della domanda (%)",
               "lunes": "Lunedì (Emiliani)", "midweek": "Metà settimana", "domingo": "Domenica",
               "hover_fecha": "Data", "hover_dia": "Giorno", "hover_tipo": "Tipo"},
    }
    L = labels.get(lang, labels["es"])

    fig = go.Figure()

    # Banda de referencia ±0 (zona "normal")
    fig.add_hrect(y0=-10, y1=10,
                  fillcolor="rgba(107,114,128,0.06)", line_width=0)

    for tipo, label_key, color in [("midweek", "midweek", _C_MIDWEEK),
                                    ("lunes", "lunes", _C_LUNES),
                                    ("domingo", "domingo", _C_DOMINGO)]:
        subset = [(h[0], h[1], h[2], h[3]) for h in _HOLIDAYS if h[4] == tipo]
        if not subset:
            continue
        fechas = [s[0] for s in subset]
        nombres = [s[1] for s in subset]
        dias = [s[2] for s in subset]
        pcts = [s[3] for s in subset]
        sizes = [max(8, min(35, abs(p) / 2.5)) for p in pcts]

        hover = [
            f"<b>{n}</b><br>"
            f"{L['hover_fecha']}: {f}<br>"
            f"{L['hover_dia']}: {d}<br>"
            f"{L['hover_tipo']}: {L.get(tipo, tipo)}<br>"
            f"Efecto: <b>{p:+.0f}%</b>"
            for n, f, d, p in zip(nombres, fechas, dias, pcts)
        ]

        fig.add_trace(go.Scatter(
            x=fechas, y=pcts,
            mode="markers+text",
            marker=dict(
                size=sizes, color=color,
                opacity=0.85,
                line=dict(color=c["card_bg"], width=1.5),
            ),
            text=[n.split()[0] for n in nombres],  # primera palabra del nombre
            textposition="top center",
            textfont=dict(size=9, color=c["text_muted"]),
            hovertext=hover, hoverinfo="text",
            name=L[label_key],
        ))

    fig.add_hline(y=0, line_color=c["border"], line_width=1.5, line_dash="solid")

    fig.update_xaxes(title_text=L["x"])
    fig.update_yaxes(
        title_text=L["y"],
        ticksuffix="%",
        range=[-95, 20],
    )
    fig.update_layout(
        title=dict(text=L["title"], font=dict(size=13, color=c["text_main"])),
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
    )

    return _theme(fig, c)


# ════════════════════════════════════════════════════════════════════════════
# ── CLIMA ────────────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════

# Coeficientes del modelo Panel FE (fuente: resultados_inferencia_causal.md)
_CLIMA_VARS = [
    # (etiqueta, beta, se, pct, unidad)
    ("Precipitación\n(+10 mm/día)", -0.01812, 0.00221,  -1.80, "por 10mm"),
    ("Temperatura\n(+1 °C)",        -0.02265, 0.001635, -2.24, "por °C"),
    ("Día lluvioso\n(>5 mm)",       -0.01085, 0.0020,   -1.08, "dummy"),
]


def build_clima_coeficientes(c: dict, lang: str = "es") -> go.Figure:
    """Forest plot de los coeficientes del Panel FE clima → demanda."""
    labels = {
        "es": {"title": "Coeficientes Panel FE — Efecto del Clima sobre log(demanda)",
               "x": "Efecto estimado (% cambio en demanda)",
               "ci": "IC 95%", "sig": "p < 0.001 ***",
               "note": "N = 49,942 · R² within = 0.659 · SE cluster-robust por estación",
               "unidad": "unidad"},
        "en": {"title": "Panel FE Coefficients — Weather Effect on log(demand)",
               "x": "Estimated effect (% change in demand)",
               "ci": "95% CI", "sig": "p < 0.001 ***",
               "note": "N = 49,942 · R² within = 0.659 · Cluster-robust SE by station",
               "unidad": "unit"},
        "it": {"title": "Coefficienti Panel FE — Effetto Clima su log(domanda)",
               "x": "Effetto stimato (% variazione domanda)",
               "ci": "IC 95%", "sig": "p < 0.001 ***",
               "note": "N = 49.942 · R² within = 0,659 · SE cluster-robust per stazione",
               "unidad": "unità"},
    }
    L = labels.get(lang, labels["es"])

    names = [v[0] for v in _CLIMA_VARS]
    betas = [v[1] * 100 for v in _CLIMA_VARS]   # escalar a %
    ses   = [v[2] * 100 for v in _CLIMA_VARS]
    pcts  = [v[3] for v in _CLIMA_VARS]
    units = [v[4] for v in _CLIMA_VARS]

    ci_lo = [b - 1.96 * s for b, s in zip(betas, ses)]
    ci_hi = [b + 1.96 * s for b, s in zip(betas, ses)]

    hover = [
        f"<b>{n.replace(chr(10), ' ')}</b><br>"
        f"β = {b:.4f}%<br>"
        f"{L['ci']}: [{lo:.4f}%, {hi:.4f}%]<br>"
        f"Efecto: <b>{p:+.2f}%</b> {u}<br>"
        f"{L['sig']}"
        for n, b, lo, hi, p, u in zip(names, betas, ci_lo, ci_hi, pcts, units)
    ]

    fig = go.Figure()
    fig.add_vline(x=0, line_color=c["border"], line_width=1.5, line_dash="solid")

    colors = [TM_ROJO] * 3
    for i, (n, b, lo, hi, ht, col) in enumerate(zip(names, betas, ci_lo, ci_hi, hover, colors)):
        fig.add_trace(go.Scatter(
            x=[b], y=[n],
            mode="markers",
            marker=dict(size=14, color=col, symbol="diamond",
                        line=dict(color=col, width=1)),
            error_x=dict(type="data", symmetric=False,
                         array=[hi - b], arrayminus=[b - lo],
                         color=col, thickness=2, width=8),
            hovertext=ht, hoverinfo="text",
            showlegend=False,
        ))

    # Banda de referencia de cero
    fig.add_vrect(x0=-0.15, x1=0.15, fillcolor="rgba(107,114,128,0.06)", line_width=0)

    fig.update_xaxes(title_text=L["x"], ticksuffix="%")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        title=dict(text=L["title"], font=dict(size=13, color=c["text_main"])),
        annotations=[dict(
            x=0.5, y=-0.18, xref="paper", yref="paper",
            text=L["note"], showarrow=False,
            font=dict(size=9, color=c["text_muted"]), align="center"
        )],
    )
    return _theme(fig, c)


def build_clima_simulacion(c: dict, lang: str = "es") -> go.Figure:
    """Curva simulada: cambio esperado en demanda según precipitación diaria."""
    labels = {
        "es": {"title": "Impacto Simulado de la Precipitación en la Demanda",
               "x": "Precipitación diaria (mm)", "y": "Cambio esperado en demanda (%)",
               "estimacion": "Estimación puntual", "ic": "IC 95%",
               "ref": "Sin efecto (β = 0)",
               "hover_p": "Precipitación", "hover_e": "Efecto esperado",
               "note_rain": "Días lluviosos (>5mm): 74 días (21%) · Precip. máx. horaria: 11.8mm"},
        "en": {"title": "Simulated Precipitation Impact on Demand",
               "x": "Daily precipitation (mm)", "y": "Expected demand change (%)",
               "estimacion": "Point estimate", "ic": "95% CI",
               "ref": "No effect (β = 0)",
               "hover_p": "Precipitation", "hover_e": "Expected effect",
               "note_rain": "Rainy days (>5mm): 74 days (21%) · Max hourly precip.: 11.8mm"},
        "it": {"title": "Impatto Simulato della Precipitazione sulla Domanda",
               "x": "Precipitazione giornaliera (mm)", "y": "Variazione attesa domanda (%)",
               "estimacion": "Stima puntuale", "ic": "IC 95%",
               "ref": "Nessun effetto (β = 0)",
               "hover_p": "Precipitazione", "hover_e": "Effetto atteso",
               "note_rain": "Giorni piovosi (>5mm): 74 (21%) · Precip. max oraria: 11,8mm"},
    }
    L = labels.get(lang, labels["es"])

    beta, se = -0.001812, 0.000221
    precips = [i * 0.5 for i in range(0, 101)]  # 0–50mm en pasos de 0.5
    efecto  = [(math.exp(beta * p) - 1) * 100 for p in precips]
    lo      = [(math.exp((beta - 1.96 * se) * p) - 1) * 100 for p in precips]
    hi      = [(math.exp((beta + 1.96 * se) * p) - 1) * 100 for p in precips]

    fig = go.Figure()
    # Banda CI
    fig.add_trace(go.Scatter(
        x=precips, y=lo, mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=precips, y=hi, mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor=_C_CI_MAIN,
        name=L["ic"], hoverinfo="skip",
    ))
    # Línea principal
    hover_curve = [
        f"{L['hover_p']}: {p:.1f}mm<br>{L['hover_e']}: <b>{e:+.2f}%</b>"
        for p, e in zip(precips, efecto)
    ]
    fig.add_trace(go.Scatter(
        x=precips, y=efecto,
        mode="lines", line=dict(color=TM_ROJO, width=2.5),
        hovertext=hover_curve, hoverinfo="text",
        name=L["estimacion"],
    ))
    # Referencia cero
    fig.add_hline(y=0, line_color=c["border"], line_width=1.5,
                  line_dash="dot", annotation_text=L["ref"],
                  annotation_font=dict(color=c["text_muted"], size=9))
    # Marcadores de referencia (5mm, 10mm, 20mm)
    for ref_p, label in [(5, "5mm"), (10, "10mm"), (20, "20mm"), (30, "30mm")]:
        ref_e = (math.exp(beta * ref_p) - 1) * 100
        fig.add_annotation(
            x=ref_p, y=ref_e, text=f"{ref_e:.1f}%",
            showarrow=True, arrowhead=2, arrowsize=0.8,
            arrowcolor=c["text_muted"], arrowwidth=1,
            font=dict(size=9, color=c["text_muted"]),
            ax=20, ay=-20,
        )

    fig.update_xaxes(title_text=L["x"], range=[0, 50])
    fig.update_yaxes(title_text=L["y"], ticksuffix="%")
    fig.update_layout(
        title=dict(text=L["title"], font=dict(size=13, color=c["text_main"])),
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
        annotations=[dict(
            x=0.5, y=-0.18, xref="paper", yref="paper",
            text=L["note_rain"], showarrow=False,
            font=dict(size=9, color=c["text_muted"]), align="center",
        )],
    )
    return _theme(fig, c)


# ════════════════════════════════════════════════════════════════════════════
# ── CICLOVÍA ─────────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════

def _load_ciclovia_geodata():
    """Carga las capas GeoJSON de la Ciclovía. Retorna None si no existen."""
    try:
        with open(os.path.join(_EXT_DIR, "stations_ciclovia_tag.geojson")) as f:
            stations = json.load(f)
        with open(os.path.join(_EXT_DIR, "ciclovia_bogota.geojson")) as f:
            routes = json.load(f)
        return stations, routes
    except FileNotFoundError:
        return None, None


def build_ciclovia_mapa(c: dict, lang: str = "es") -> go.Figure:
    """
    Mapa Scattermap interactivo: estaciones tratadas (≤500m) vs. control
    superpuestas sobre las rutas de Ciclovía.
    """
    labels = {
        "es": {"title": "Mapa de Tratamiento — Estaciones y Rutas Ciclovía",
               "tratada": "Tratada (≤500m)", "control": "Control (>500m)",
               "ruta": "Ruta Ciclovía",
               "hover_t": "Tratada", "hover_c": "Control",
               "hover_est": "Estación", "hover_dist": "Proximidad"},
        "en": {"title": "Treatment Map — Stations and Ciclovía Routes",
               "tratada": "Treated (≤500m)", "control": "Control (>500m)",
               "ruta": "Ciclovía route",
               "hover_t": "Treated", "hover_c": "Control",
               "hover_est": "Station", "hover_dist": "Proximity"},
        "it": {"title": "Mappa di Trattamento — Stazioni e Percorsi Ciclovía",
               "tratada": "Trattata (≤500m)", "control": "Controllo (>500m)",
               "ruta": "Percorso Ciclovía",
               "hover_t": "Trattata", "hover_c": "Controllo",
               "hover_est": "Stazione", "hover_dist": "Prossimità"},
    }
    L = labels.get(lang, labels["es"])
    stations_gj, routes_gj = _load_ciclovia_geodata()
    fig = go.Figure()

    if stations_gj and routes_gj:
        # ── Rutas Ciclovía como líneas ──
        for feat in routes_gj["features"]:
            coords = feat["geometry"]["coordinates"]
            for line in coords:
                lons = [pt[0] for pt in line]
                lats = [pt[1] for pt in line]
                fig.add_trace(go.Scattermap(
                    lon=lons, lat=lats, mode="lines",
                    line=dict(color="#10B981", width=2.5),
                    opacity=0.7, showlegend=False, hoverinfo="skip",
                    name=L["ruta"],
                ))
        # Traza fantasma para leyenda de ruta
        fig.add_trace(go.Scattermap(
            lon=[None], lat=[None], mode="lines",
            line=dict(color="#10B981", width=3),
            name=L["ruta"], showlegend=True,
        ))

        # ── Estaciones ──
        for tipo, label_key, color, symbol, cerca_val in [
            ("tratada", "tratada", _C_LUNES, "circle", 1),
            ("control", "control", _C_INSIG, "circle", 0),
        ]:
            feats = [f for f in stations_gj["features"]
                     if f["properties"].get("cerca_ciclovia") == cerca_val]
            lons  = [f["geometry"]["coordinates"][0] for f in feats]
            lats  = [f["geometry"]["coordinates"][1] for f in feats]
            names = [f["properties"].get("nombre_estacion", "") for f in feats]
            hover = [
                f"<b>{n}</b><br>{L['hover_dist']}: {L[label_key]}"
                for n in names
            ]
            fig.add_trace(go.Scattermap(
                lon=lons, lat=lats, mode="markers",
                marker=dict(size=8, color=color, opacity=0.85),
                hovertext=hover, hoverinfo="text",
                name=L[label_key],
            ))

    fig.update_layout(
        mapbox=dict(style="carto-positron", zoom=11,
                    center=dict(lat=4.65, lon=-74.10)),
        map=dict(style="carto-positron", zoom=11,
                 center=dict(lat=4.65, lon=-74.10)),
        title=dict(text=L["title"], font=dict(size=13, color=c["text_main"])),
        legend=dict(orientation="h", y=1.06, x=0.5, xanchor="center"),
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor=c["card_bg"],
        font=dict(family="Inter, Roboto, sans-serif", color=c["text_main"]),
        hoverlabel=dict(bgcolor=c["card_bg"], bordercolor=c["border"],
                        font=dict(color=c["text_main"])),
    )
    return fig


def build_ciclovia_resultados(c: dict, lang: str = "es") -> go.Figure:
    """
    Forest plot con los 3 estimadores DiD: Spec 1, Spec 2 y placebo en sábados.
    """
    labels = {
        "es": {"title": "Resultados DiD — Efecto Ciclovía sobre Demanda de TM",
               "x": "Coeficiente β estimado",
               "spec1": "Spec 1: Domingo × Cercana\n(N = 52,385)",
               "spec2": "Spec 2: Ventana horaria\n7–14h domingos (N = 112,134)",
               "placebo": "Placebo: Misma ventana\nen sábados (N = 112,159)",
               "sig": "Significativo", "nosig": "No significativo",
               "ref": "β = 0 (sin efecto)",
               "hover_b": "β", "hover_ci": "IC 95%", "hover_p": "p-valor",
               "note": "FE de estación + día de semana + mes · SE cluster-robust"},
        "en": {"title": "DiD Results — Ciclovía Effect on TM Demand",
               "x": "Estimated β coefficient",
               "spec1": "Spec 1: Sunday × Nearby\n(N = 52,385)",
               "spec2": "Spec 2: Time window\n7–14h Sundays (N = 112,134)",
               "placebo": "Placebo: Same window\non Saturdays (N = 112,159)",
               "sig": "Significant", "nosig": "Not significant",
               "ref": "β = 0 (no effect)",
               "hover_b": "β", "hover_ci": "95% CI", "hover_p": "p-value",
               "note": "Station + weekday + month FE · Cluster-robust SE"},
        "it": {"title": "Risultati DiD — Effetto Ciclovía sulla Domanda TM",
               "x": "Coefficiente β stimato",
               "spec1": "Spec 1: Domenica × Vicina\n(N = 52.385)",
               "spec2": "Spec 2: Finestra oraria\n7–14h domeniche (N = 112.134)",
               "placebo": "Placebo: Stessa finestra\nSabati (N = 112.159)",
               "sig": "Significativo", "nosig": "Non significativo",
               "ref": "β = 0 (nessun effetto)",
               "hover_b": "β", "hover_ci": "IC 95%", "hover_p": "p-valore",
               "note": "FE stazione + giorno + mese · SE cluster-robust"},
    }
    L = labels.get(lang, labels["es"])

    specs = [
        (L["spec1"],   -0.087,  0.059,  0.136, False, _C_INSIG),
        (L["spec2"],   +0.001,  0.032,  0.982, False, _C_INSIG),
        (L["placebo"], +0.063,  0.038,  0.096, True,  TM_AMARILLO),
    ]

    fig = go.Figure()
    fig.add_vline(x=0, line_color=c["border"], line_width=1.5,
                  line_dash="dash", annotation_text=L["ref"],
                  annotation_font=dict(color=c["text_muted"], size=9),
                  annotation_position="top right")

    for name, beta, se, pval, is_marginal, color in specs:
        lo = beta - 1.96 * se
        hi = beta + 1.96 * se
        symbol = "diamond" if is_marginal else "circle"
        ht = (f"<b>{name.replace(chr(10), ' ')}</b><br>"
              f"{L['hover_b']} = {beta:.3f}<br>"
              f"{L['hover_ci']}: [{lo:.3f}, {hi:.3f}]<br>"
              f"{L['hover_p']} = {pval:.3f}"
              + (" (*)" if is_marginal else ""))
        fig.add_trace(go.Scatter(
            x=[beta], y=[name],
            mode="markers",
            marker=dict(size=14, color=color, symbol=symbol,
                        line=dict(color=color, width=1.5)),
            error_x=dict(type="data", symmetric=False,
                         array=[hi - beta], arrayminus=[beta - lo],
                         color=color, thickness=2, width=8),
            hovertext=ht, hoverinfo="text",
            showlegend=False,
        ))

    fig.update_xaxes(title_text=L["x"])
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        title=dict(text=L["title"], font=dict(size=13, color=c["text_main"])),
        annotations=[dict(
            x=0.5, y=-0.18, xref="paper", yref="paper",
            text=L["note"], showarrow=False,
            font=dict(size=9, color=c["text_muted"]), align="center",
        )],
    )
    return _theme(fig, c)


def build_ciclovia_ratio(c: dict, lang: str = "es") -> go.Figure:
    """
    Diagrama de barras: ratio Demanda Domingo / Demanda Laboral para
    estaciones tratadas vs. control (estadístico descriptivo clave).
    """
    labels = {
        "es": {"title": "Ratio Demanda Dominical vs. Laboral — Tratadas vs. Control",
               "y": "Ratio Domingo / Día laboral",
               "tratada": "Tratadas (≤500m)", "control": "Control (>500m)",
               "hover": "Ratio Dom/Laboral",
               "note": "63 estaciones tratadas · 86 estaciones control · Promedio 2025"},
        "en": {"title": "Sunday vs. Weekday Demand Ratio — Treated vs. Control",
               "y": "Sunday / Weekday ratio",
               "tratada": "Treated (≤500m)", "control": "Control (>500m)",
               "hover": "Sun/Weekday ratio",
               "note": "63 treated stations · 86 control stations · 2025 average"},
        "it": {"title": "Rapporto Domanda Domenicale vs. Lavorativa — Trattate vs. Controllo",
               "y": "Rapporto Domenica / Lavorativo",
               "tratada": "Trattate (≤500m)", "control": "Controllo (>500m)",
               "hover": "Rapporto Dom/Lavorativo",
               "note": "63 stazioni trattate · 86 stazioni controllo · Media 2025"},
    }
    L = labels.get(lang, labels["es"])

    groups  = [L["tratada"], L["control"]]
    ratios  = [0.385, 0.416]
    colors  = [_C_LUNES, _C_INSIG]
    labels_ = ["0.385", "0.416"]

    fig = go.Figure()
    for grp, ratio, col, lbl in zip(groups, ratios, colors, labels_):
        fig.add_trace(go.Bar(
            x=[grp], y=[ratio],
            marker=dict(color=col, opacity=0.85),
            text=[lbl], textposition="outside",
            textfont=dict(size=12, color=c["text_main"]),
            hovertext=f"<b>{grp}</b><br>{L['hover']}: <b>{ratio:.3f}</b>",
            hoverinfo="text",
            showlegend=False,
            width=0.45,
        ))

    # Anotación de la diferencia
    fig.add_annotation(
        x=0.5, y=0.425, xref="paper", yref="y",
        text="Δ = −0.031<br>(−7.4%)",
        showarrow=False,
        font=dict(size=11, color=c["text_main"]),
        bgcolor="rgba(255,209,0,0.19)",
        bordercolor=TM_AMARILLO,
        borderwidth=1,
        borderpad=4,
    )
    fig.add_shape(
        type="line", x0=0, x1=1, y0=0.385, y1=0.416,
        xref="paper", yref="y",
        line=dict(color=TM_AMARILLO, width=1.5, dash="dot"),
    )

    fig.update_yaxes(
        title_text=L["y"], range=[0, 0.52],
        tickformat=".3f",
    )
    fig.update_layout(
        title=dict(text=L["title"], font=dict(size=13, color=c["text_main"])),
        bargap=0.4,
        annotations=[dict(
            x=0.5, y=-0.18, xref="paper", yref="paper",
            text=L["note"], showarrow=False,
            font=dict(size=9, color=c["text_muted"]), align="center",
        )],
    )
    return _theme(fig, c)


# ════════════════════════════════════════════════════════════════════════════
# ── CONCIERTOS ───────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════

# Event study horario — solo horas con efectos reportados
# Fuente: resultados_inferencia_causal.md — Análisis 4
_CAMPIN_HOURS = list(range(0, 24))
_CAMPIN_BETA = {
    15: (0.355, 0.089, 42.7, "***"), 16: (0.374, 0.094, 45.3, "***"),
    19: (0.319, 0.097, 37.5, "***"), 20: (0.569, 0.142, 76.7, "***"),
    21: (0.383, 0.096, 46.7, "***"), 22: (0.333, 0.194, 39.6, "*"),
    23: (0.842, 0.501, 132.0, "*"),   0: (0.948, 0.484, 158.2, "**"),
}  # horas no reportadas → β ≈ 0, no sig.

_CONCIERTOS = [
    # (nombre, fecha, dia, ratio_h23, ratio_h20, señal)
    ("Andrea Bocelli",     "2025-02-21", "Vie", 9.0,  None, "muy_fuerte"),
    ("Shakira (noche 2)",  "2025-02-27", "Jue", 4.7,  None, "fuerte"),
    ("Maluma",             "2025-05-03", "Sáb", 1.9,  None, "debil"),
    ("Shakira (noche 1)",  "2025-02-26", "Mié", 1.2,  None, "debil"),
    ("Rockin 1000",        "2025-05-20", "Mar", 1.1,  None, "sin_señal"),
    ("Marco A. Solís",     "2025-03-07", "Vie", 0.9,  None, "sin_señal"),
    ("Sting",              "2025-03-02", "Dom", 0.8,  5.6,  "dominical"),
]

_SPILLOVER = [
    # (banda, n_est, h20_pct, h23_pct, sig_h20, sig_h23)
    ("≤500m (n=2)",       2,  +77,  +132, "***", "*"),
    ("500–1500m (n=3)",   3,  +16,  +14,  "ns",  "ns"),
    ("1500–3000m (n=16)", 16, +11,  +14,  "**",  "ns"),
]

_FDV_VS_CONC = [
    # (h, conc_pct, fdv_pct, sig)
    (18, 20.9, 19.5, "ns"),
    (19, 37.5, 27.5, "ns"),
    (20, 76.7, 13.7, "***"),
    (21, 46.7, 24.9, "ns"),
    (22, 39.6, 37.1, "ns"),
    (23, 132.0, 17.0, "ns"),
]


def build_campin_event_study(c: dict, lang: str = "es") -> go.Figure:
    """Event study horario: β_h 0–23h con banda IC 95%."""
    labels = {
        "es": {"title": "Event Study Horario — Conciertos en El Campín / Movistar Arena",
               "x": "Hora del día", "y": "Coeficiente β",
               "sig": "Significativo", "insig": "No significativo",
               "hover_h": "Hora", "hover_b": "β", "hover_ci": "IC 95%",
               "hover_e": "Efecto", "hover_sig": "Significancia",
               "note": "2 estaciones primarias (245–270m) · 7 conciertos · SE HC3"},
        "en": {"title": "Hourly Event Study — Concerts at El Campín / Movistar Arena",
               "x": "Hour of day", "y": "Coefficient β",
               "sig": "Significant", "insig": "Not significant",
               "hover_h": "Hour", "hover_b": "β", "hover_ci": "95% CI",
               "hover_e": "Effect", "hover_sig": "Significance",
               "note": "2 primary stations (245–270m) · 7 concerts · HC3 SE"},
        "it": {"title": "Event Study Orario — Concerti all'El Campín / Movistar Arena",
               "x": "Ora del giorno", "y": "Coefficiente β",
               "sig": "Significativo", "insig": "Non significativo",
               "hover_h": "Ora", "hover_b": "β", "hover_ci": "IC 95%",
               "hover_e": "Effetto", "hover_sig": "Significanza",
               "note": "2 stazioni primarie (245–270m) · 7 concerti · SE HC3"},
    }
    L = labels.get(lang, labels["es"])

    hours = list(range(0, 24))
    betas, ses, pcts, sigs, stars = [], [], [], [], []
    for h in hours:
        if h in _CAMPIN_BETA:
            b, s, p, st = _CAMPIN_BETA[h]
            betas.append(b)
            ses.append(s)
            pcts.append(p)
            sigs.append(True)
            stars.append(st)
        else:
            betas.append(0.0)
            ses.append(0.05)
            pcts.append(0.0)
            sigs.append(False)
            stars.append("")

    ci_lo = [b - 1.96 * s for b, s in zip(betas, ses)]
    ci_hi = [b + 1.96 * s for b, s in zip(betas, ses)]

    fig = go.Figure()
    # Banda CI
    fig.add_trace(go.Scatter(x=hours, y=ci_lo, mode="lines",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=hours, y=ci_hi, mode="lines",
                             line=dict(width=0), fill="tonexty",
                             fillcolor=_C_CI_MAIN, showlegend=False, hoverinfo="skip"))
    # Línea base
    fig.add_trace(go.Scatter(
        x=hours, y=betas, mode="lines",
        line=dict(color=TM_ROJO, width=1.5, dash="dot"),
        showlegend=False, hoverinfo="skip",
    ))
    # Puntos
    colors = [_sig_color(s) for s in sigs]
    sizes  = [13 if s else 7 for s in sigs]
    hover  = [
        f"<b>{L['hover_h']} = {h:02d}h</b><br>"
        f"{L['hover_b']} = {b:.3f}<br>"
        f"{L['hover_ci']}: [{lo:.3f}, {hi:.3f}]<br>"
        f"{L['hover_e']}: <b>{p:+.1f}%</b>"
        f"{(' ' + st) if st else ''}"
        for h, b, lo, hi, p, st in zip(hours, betas, ci_lo, ci_hi, pcts, stars)
    ]
    fig.add_trace(go.Scatter(
        x=hours, y=betas, mode="markers",
        marker=dict(size=sizes, color=colors,
                    line=dict(color=colors, width=1.5), symbol="circle"),
        hovertext=hover, hoverinfo="text",
        showlegend=False,
    ))
    # Sombreado franja de concierto (19–23h)
    fig.add_vrect(x0=19, x1=23.5, fillcolor="rgba(193,0,31,0.05)", line_width=0,
                  annotation_text="Concierto", annotation_position="top",
                  annotation_font=dict(color=TM_ROJO, size=9))
    fig.add_hline(y=0, line_color=c["border"], line_width=1.5)

    fig.update_xaxes(title_text=L["x"], tickvals=list(range(0, 24, 2)),
                     ticktext=[f"{h:02d}h" for h in range(0, 24, 2)])
    fig.update_yaxes(title_text=L["y"])
    fig.update_layout(
        title=dict(text=L["title"], font=dict(size=13, color=c["text_main"])),
        annotations=[dict(x=0.5, y=-0.18, xref="paper", yref="paper",
                          text=L["note"], showarrow=False,
                          font=dict(size=9, color=c["text_muted"]), align="center")],
    )
    return _theme(fig, c)


def build_campin_por_evento(c: dict, lang: str = "es") -> go.Figure:
    """Barras horizontales con el ratio a h=23 de cada concierto."""
    labels = {
        "es": {"title": "Impacto Individual — Ratio Validaciones Evento / Control a h=23",
               "x": "Ratio evento / control (h=23h)", "ref": "Sin efecto (1×)",
               "sting_note": "Sting: pico en h=20 (5.6×)", "domingo": "Show dominical",
               "hover_r23": "Ratio h=23", "hover_r20": "Ratio h=20",
               "señal_map": {"muy_fuerte": "🟢 Muy fuerte", "fuerte": "🟡 Fuerte",
                             "debil": "🟠 Débil", "sin_señal": "🔴 Sin señal",
                             "dominical": "🔵 Show dominical"}},
        "en": {"title": "Individual Impact — Event/Control Validation Ratio at h=23",
               "x": "Event / control ratio (23h)", "ref": "No effect (1×)",
               "sting_note": "Sting: peak at h=20 (5.6×)", "domingo": "Sunday show",
               "hover_r23": "Ratio h=23", "hover_r20": "Ratio h=20",
               "señal_map": {"muy_fuerte": "🟢 Very strong", "fuerte": "🟡 Strong",
                             "debil": "🟠 Weak", "sin_señal": "🔴 No signal",
                             "dominical": "🔵 Sunday show"}},
        "it": {"title": "Impatto Individuale — Rapporto Evento/Controllo a h=23",
               "x": "Rapporto evento / controllo (23h)", "ref": "Nessun effetto (1×)",
               "sting_note": "Sting: picco a h=20 (5,6×)", "domingo": "Show domenicale",
               "hover_r23": "Rapporto h=23", "hover_r20": "Rapporto h=20",
               "señal_map": {"muy_fuerte": "🟢 Molto forte", "fuerte": "🟡 Forte",
                             "debil": "🟠 Debole", "sin_señal": "🔴 Nessun segnale",
                             "dominical": "🔵 Show domenicale"}},
    }
    L = labels.get(lang, labels["es"])
    sm = L["señal_map"]

    # Ordenar por ratio h=23 descendente
    conc = sorted(_CONCIERTOS, key=lambda x: x[3], reverse=True)

    names   = [c_[0] for c_ in conc]
    r23     = [c_[3] for c_ in conc]
    r20     = [c_[4] for c_ in conc]
    fechas  = [c_[1] for c_ in conc]
    dias    = [c_[2] for c_ in conc]
    señales = [c_[5] for c_ in conc]

    sig_color_map = {
        "muy_fuerte": "#198754", "fuerte": TM_AMARILLO,
        "debil": "#FD7E14", "sin_señal": _C_INSIG, "dominical": _C_LUNES,
    }
    colors_ = [sig_color_map[s] for s in señales]

    hover = [
        f"<b>{n}</b><br>{f} ({d})<br>"
        f"{L['hover_r23']}: <b>{r:.1f}×</b><br>"
        + (f"{L['hover_r20']}: {r2:.1f}×<br>" if r2 else "")
        + f"{sm.get(s, s)}"
        for n, f, d, r, r2, s in zip(names, fechas, dias, r23, r20, señales)
    ]

    fig = go.Figure(go.Bar(
        x=r23, y=names, orientation="h",
        marker=dict(color=colors_, opacity=0.85),
        hovertext=hover, hoverinfo="text",
        showlegend=False,
    ))
    fig.add_vline(x=1, line_color=c["border"], line_width=2, line_dash="dash",
                  annotation_text=L["ref"],
                  annotation_font=dict(color=c["text_muted"], size=9))

    # Anotación Sting
    sting_idx = next(i for i, c_ in enumerate(conc) if "Sting" in c_[0])
    fig.add_annotation(
        x=r23[sting_idx] + 0.2, y=names[sting_idx],
        text=L["sting_note"],
        showarrow=True, arrowhead=2, arrowsize=0.8,
        arrowcolor=_C_LUNES, arrowwidth=1,
        font=dict(size=9, color=_C_LUNES),
        ax=60, ay=0,
    )

    fig.update_xaxes(title_text=L["x"], ticksuffix="×")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        title=dict(text=L["title"], font=dict(size=13, color=c["text_main"])),
    )
    return _theme(fig, c)


def build_campin_spillover(c: dict, lang: str = "es") -> go.Figure:
    """Barras agrupadas: efecto por franja de distancia en h=20 y h=23."""
    labels = {
        "es": {"title": "Spillover Espacial — Efecto por Franja de Distancia",
               "y": "Cambio en demanda (%)", "h20": "Hora 20h (durante el show)",
               "h23": "Hora 23h (salida del público)",
               "hover_b": "Distancia", "hover_20": "Efecto h=20",
               "hover_23": "Efecto h=23"},
        "en": {"title": "Spatial Spillover — Effect by Distance Band",
               "y": "Demand change (%)", "h20": "Hour 20h (during show)",
               "h23": "Hour 23h (audience exit)",
               "hover_b": "Distance", "hover_20": "Effect h=20",
               "hover_23": "Effect h=23"},
        "it": {"title": "Spillover Spaziale — Effetto per Fascia di Distanza",
               "y": "Variazione domanda (%)", "h20": "Ora 20h (durante lo show)",
               "h23": "Ora 23h (uscita del pubblico)",
               "hover_b": "Distanza", "hover_20": "Effetto h=20",
               "hover_23": "Effetto h=23"},
    }
    L = labels.get(lang, labels["es"])

    bandas = [s[0] for s in _SPILLOVER]
    h20s   = [s[2] for s in _SPILLOVER]
    h23s   = [s[3] for s in _SPILLOVER]
    sig20  = [s[4] for s in _SPILLOVER]
    sig23  = [s[5] for s in _SPILLOVER]

    def bar_color(sig):
        return TM_ROJO if sig not in ("ns", "") else _C_INSIG

    def hover_txt(banda, h20, h23, s20, s23):
        return (f"<b>{banda}</b><br>"
                f"{L['hover_20']}: <b>{h20:+.0f}%</b> {s20}<br>"
                f"{L['hover_23']}: <b>{h23:+.0f}%</b> {s23}")

    hover = [hover_txt(*s) for s in zip(bandas, h20s, h23s, sig20, sig23)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=bandas, y=h20s,
        name=L["h20"],
        marker=dict(color=[bar_color(s) for s in sig20], opacity=0.75),
        hovertext=hover, hoverinfo="text",
    ))
    fig.add_trace(go.Bar(
        x=bandas, y=h23s,
        name=L["h23"],
        marker=dict(color=[TM_ROJO if s not in ("ns", "") else _C_INSIG
                           for s in sig23], opacity=0.5,
                    pattern_shape="/"),
        hovertext=hover, hoverinfo="text",
    ))

    # Etiquetas de significancia
    for i, (b, h20, h23, s20, s23) in enumerate(zip(bandas, h20s, h23s, sig20, sig23)):
        for val, sig in [(h20, s20), (h23, s23)]:
            if sig != "ns":
                fig.add_annotation(x=b, y=val + 4, text=sig,
                                   showarrow=False,
                                   font=dict(size=11, color=TM_ROJO))

    fig.add_hline(y=0, line_color=c["border"], line_width=1.5)
    fig.update_yaxes(title_text=L["y"], ticksuffix="%")
    fig.update_layout(
        title=dict(text=L["title"], font=dict(size=13, color=c["text_main"])),
        barmode="group", bargap=0.25, bargroupgap=0.1,
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
    )
    return _theme(fig, c)


def build_campin_fdv_comparacion(c: dict, lang: str = "es") -> go.Figure:
    """Barras dobles: conciertos vs. Festival de Verano por hora."""
    labels = {
        "es": {"title": "Conciertos vs. Festival de Verano — Efecto por Hora",
               "y": "Cambio en demanda (%)", "conc": "Conciertos (n=7)",
               "fdv": "Festival de Verano (n=8 días)",
               "sig": "*** diferencia significativa a h=20 (p=0.007)",
               "hover_h": "Hora", "hover_diff": "Diferencia"},
        "en": {"title": "Concerts vs. Summer Festival — Effect by Hour",
               "y": "Demand change (%)", "conc": "Concerts (n=7)",
               "fdv": "Summer Festival (n=8 days)",
               "sig": "*** significant difference at h=20 (p=0.007)",
               "hover_h": "Hour", "hover_diff": "Difference"},
        "it": {"title": "Concerti vs. Festival d'Estate — Effetto per Ora",
               "y": "Variazione domanda (%)", "conc": "Concerti (n=7)",
               "fdv": "Festival d'Estate (n=8 giorni)",
               "sig": "*** differenza significativa a h=20 (p=0,007)",
               "hover_h": "Ora", "hover_diff": "Differenza"},
    }
    L = labels.get(lang, labels["es"])

    hours   = [r[0] for r in _FDV_VS_CONC]
    conc_p  = [r[1] for r in _FDV_VS_CONC]
    fdv_p   = [r[2] for r in _FDV_VS_CONC]
    sigs    = [r[3] for r in _FDV_VS_CONC]
    hlabels = [f"{h:02d}h" for h in hours]

    hover_conc = [
        f"<b>{L['conc']}</b><br>{L['hover_h']} = {h:02d}h<br>"
        f"Efecto: <b>{p:+.1f}%</b>"
        f"{(' ' + s) if s != 'ns' else ''}"
        for h, p, s in zip(hours, conc_p, sigs)
    ]
    hover_fdv = [
        f"<b>{L['fdv']}</b><br>{L['hover_h']} = {h:02d}h<br>Efecto: <b>{p:+.1f}%</b>"
        for h, p in zip(hours, fdv_p)
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=hlabels, y=conc_p, name=L["conc"],
        marker=dict(color=TM_ROJO, opacity=0.85),
        hovertext=hover_conc, hoverinfo="text",
    ))
    fig.add_trace(go.Bar(
        x=hlabels, y=fdv_p, name=L["fdv"],
        marker=dict(color=TM_AMARILLO, opacity=0.7),
        hovertext=hover_fdv, hoverinfo="text",
    ))
    # Anotación en h=20
    fig.add_annotation(
        x="20h", y=max(conc_p) + 10, text="***",
        showarrow=False,
        font=dict(size=16, color=TM_ROJO),
    )
    fig.add_annotation(
        x=0.5, y=-0.18, xref="paper", yref="paper",
        text=L["sig"], showarrow=False,
        font=dict(size=9, color=TM_ROJO), align="center",
    )
    fig.add_hline(y=0, line_color=c["border"], line_width=1.5)
    fig.update_yaxes(title_text=L["y"], ticksuffix="%")
    fig.update_layout(
        title=dict(text=L["title"], font=dict(size=13, color=c["text_main"])),
        barmode="group", bargap=0.25, bargroupgap=0.08,
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
    )
    return _theme(fig, c)


def build_fdv_tipo_dia(c: dict, lang: str = "es") -> go.Figure:
    """Barras agrupadas: Festival de Verano por tipo de día (fin de semana vs. hábil)."""
    labels = {
        "es": {"title": "Festival de Verano — Efecto por Tipo de Día",
               "y": "Ratio demanda evento / control",
               "fds": "Fin de semana (sáb–dom, n=4)",
               "hab": "Días hábiles (lun–vie, n=4)",
               "h20": "Hora 20h", "h23": "Hora 23h",
               "ref": "Sin efecto (1×)",
               "hover_t": "Tipo de día", "hover_h": "Hora", "hover_r": "Ratio"},
        "en": {"title": "Summer Festival — Effect by Day Type",
               "y": "Event / control demand ratio",
               "fds": "Weekend (sat–sun, n=4)",
               "hab": "Weekdays (mon–fri, n=4)",
               "h20": "Hour 20h", "h23": "Hour 23h",
               "ref": "No effect (1×)",
               "hover_t": "Day type", "hover_h": "Hour", "hover_r": "Ratio"},
        "it": {"title": "Festival d'Estate — Effetto per Tipo di Giorno",
               "y": "Rapporto domanda evento / controllo",
               "fds": "Fine settimana (sab–dom, n=4)",
               "hab": "Giorni lavorativi (lun–ven, n=4)",
               "h20": "Ora 20h", "h23": "Ora 23h",
               "ref": "Nessun effetto (1×)",
               "hover_t": "Tipo di giorno", "hover_h": "Ora", "hover_r": "Rapporto"},
    }
    L = labels.get(lang, labels["es"])

    tipos  = [L["fds"], L["hab"]]
    r20    = [1.47, 1.04]
    r23    = [1.57, 1.22]

    fig = go.Figure()
    for tipo, rv20, rv23, color in zip(tipos, r20, r23,
                                        [_C_LUNES, _C_INSIG]):
        fig.add_trace(go.Bar(
            x=[tipo], y=[rv20], name=L["h20"] if tipo == tipos[0] else None,
            legendgroup="h20", showlegend=(tipo == tipos[0]),
            marker=dict(color=color, opacity=0.85),
            hovertext=f"<b>{tipo}</b><br>{L['hover_h']}: 20h<br>{L['hover_r']}: <b>{rv20:.2f}×</b>",
            hoverinfo="text",
        ))
        fig.add_trace(go.Bar(
            x=[tipo], y=[rv23], name=L["h23"] if tipo == tipos[0] else None,
            legendgroup="h23", showlegend=(tipo == tipos[0]),
            marker=dict(color=color, opacity=0.45, pattern_shape="/"),
            hovertext=f"<b>{tipo}</b><br>{L['hover_h']}: 23h<br>{L['hover_r']}: <b>{rv23:.2f}×</b>",
            hoverinfo="text",
        ))

    fig.add_hline(y=1, line_color=c["border"], line_width=2, line_dash="dash",
                  annotation_text=L["ref"],
                  annotation_font=dict(color=c["text_muted"], size=9))
    fig.update_yaxes(title_text=L["y"], ticksuffix="×", range=[0, 2.1])
    fig.update_layout(
        title=dict(text=L["title"], font=dict(size=13, color=c["text_main"])),
        barmode="group", bargap=0.35, bargroupgap=0.05,
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
    )
    return _theme(fig, c)


# ════════════════════════════════════════════════════════════════════════════
# ── COMBUSTIBLE ──────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════

# Serie mensual aproximada (5 niveles distintos, 7 meses sin cambio)
# Fuente: resultados_inferencia_causal.md — Análisis 5
_MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
          "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
_PRECIOS = [16085, 16085, 16085, 16186, 16186, 16186,
            16285, 16285, 16350, 16350, 16393, 16393]
_DEMANDA_DIA = [3.05, 3.52, 3.48, 3.15, 3.41, 3.32,
                3.45, 3.38, 3.58, 3.35, 3.40, 3.10]  # millones/día (aprox.)

_COMBUSTIBLE_SPECS = [
    # (nombre, elasticidad, p_val, es_espuria)
    ("Spec 1a: log(val_mes) ~ log_precio + trend + festivos", +10.44, 0.000, True),
    ("Spec 1b: log(val_dia) ~ log_precio + trend + festivos", +11.93, 0.117, False),
    ("Spec 2: Δlog(val_dia) ~ Δlog(precio)",                  +7.76,  0.693, False),
]


def build_combustible_serie(c: dict, lang: str = "es") -> go.Figure:
    """Serie temporal dual: precio gasolina (eje dcho.) y demanda/día (eje izdo.)."""
    labels = {
        "es": {"title": "Serie Mensual — Precio Gasolina y Demanda TransMilenio 2025",
               "y_dem": "Validaciones por día (millones)", "y_prec": "Precio gasolina (COP/galón)",
               "dem": "Demanda diaria (TM)", "prec": "Precio gasolina",
               "hover_m": "Mes", "hover_d": "Demanda/día", "hover_p": "Precio",
               "note": "Precio: 5 niveles distintos en 2025 (+1.9% anual) · Demanda: valores aproximados"},
        "en": {"title": "Monthly Series — Fuel Price and TransMilenio Demand 2025",
               "y_dem": "Validations per day (millions)", "y_prec": "Fuel price (COP/gallon)",
               "dem": "Daily demand (TM)", "prec": "Fuel price",
               "hover_m": "Month", "hover_d": "Demand/day", "hover_p": "Price",
               "note": "Price: 5 distinct levels in 2025 (+1.9% annual) · Demand: approximate values"},
        "it": {"title": "Serie Mensile — Prezzo Carburante e Domanda TransMilenio 2025",
               "y_dem": "Validazioni per giorno (milioni)", "y_prec": "Prezzo carburante (COP/litro)",
               "dem": "Domanda giornaliera (TM)", "prec": "Prezzo carburante",
               "hover_m": "Mese", "hover_d": "Domanda/giorno", "hover_p": "Prezzo",
               "note": "Prezzo: 5 livelli distinti nel 2025 (+1,9% annuo) · Domanda: valori approssimativi"},
    }
    L = labels.get(lang, labels["es"])

    hover_dem = [
        f"<b>{L['hover_m']}: {m}</b><br>"
        f"{L['hover_d']}: <b>{d:.2f}M</b><br>"
        f"{L['hover_p']}: {p:,} COP/gal"
        for m, d, p in zip(_MESES, _DEMANDA_DIA, _PRECIOS)
    ]
    hover_prec = [
        f"<b>{L['hover_m']}: {m}</b><br>"
        f"{L['hover_p']}: <b>{p:,} COP/gal</b><br>"
        f"{L['hover_d']}: {d:.2f}M"
        for m, d, p in zip(_MESES, _DEMANDA_DIA, _PRECIOS)
    ]

    fig = go.Figure()
    # Demanda (eje primario)
    fig.add_trace(go.Scatter(
        x=_MESES, y=_DEMANDA_DIA,
        mode="lines+markers",
        name=L["dem"],
        line=dict(color=_C_LUNES, width=2.5),
        marker=dict(size=8, color=_C_LUNES),
        hovertext=hover_dem, hoverinfo="text",
        yaxis="y1",
    ))
    # Precio (eje secundario — step line para reflejar cambios discretos)
    fig.add_trace(go.Scatter(
        x=_MESES, y=_PRECIOS,
        mode="lines+markers",
        name=L["prec"],
        line=dict(color=TM_ROJO, width=2.5, shape="hv"),
        marker=dict(size=8, color=TM_ROJO, symbol="square"),
        hovertext=hover_prec, hoverinfo="text",
        yaxis="y2",
    ))

    fig.update_layout(
        title=dict(text=L["title"], font=dict(size=13, color=c["text_main"])),
        yaxis=dict(title=dict(text=L["y_dem"], font=dict(color=_C_LUNES)),
                   tickfont=dict(color=_C_LUNES), range=[2.8, 3.8]),
        yaxis2=dict(title=dict(text=L["y_prec"], font=dict(color=TM_ROJO)),
                    tickfont=dict(color=TM_ROJO), overlaying="y", side="right",
                    range=[15900, 16600]),
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
        annotations=[dict(x=0.5, y=-0.18, xref="paper", yref="paper",
                          text=L["note"], showarrow=False,
                          font=dict(size=9, color=c["text_muted"]), align="center")],
    )
    # Aplicar tema manualmente (yaxis2 necesita override)
    fig.update_layout(
        paper_bgcolor=c["card_bg"], plot_bgcolor=c["chart_box"],
        font=dict(family="Inter, Roboto, sans-serif", color=c["text_main"]),
        margin=dict(l=8, r=8, t=36, b=8),
        hoverlabel=dict(bgcolor=c["card_bg"], bordercolor=c["border"],
                        font=dict(color=c["text_main"])),
    )
    fig.update_xaxes(gridcolor=c["border"], tickfont=dict(color=c["text_muted"]),
                     linecolor=c["border"])
    fig.update_yaxes(gridcolor=c["border"], linecolor=c["border"], selector=dict(side="left"))
    return fig


def build_combustible_especificaciones(c: dict, lang: str = "es") -> go.Figure:
    """
    Forest plot comparando las 3 especificaciones econométricas.
    Visualiza por qué el resultado es inconcluso: colinealidad en Spec 1a.
    """
    labels = {
        "es": {"title": "Especificaciones Econométricas — Elasticidad Precio-Demanda",
               "x": "Elasticidad estimada (β log-log)",
               "espurio": "⚠️ Resultado espurio (VIF precio↔tendencia = 6.2)",
               "nosig": "No significativo",
               "hover_e": "Elasticidad", "hover_p": "p-valor", "hover_n": "N",
               "note": "N = 12 meses · Variación de precio < 2% en 2025 · Inconcluso por diseño"},
        "en": {"title": "Econometric Specifications — Price-Demand Elasticity",
               "x": "Estimated elasticity (β log-log)",
               "espurio": "⚠️ Spurious result (VIF price↔trend = 6.2)",
               "nosig": "Not significant",
               "hover_e": "Elasticity", "hover_p": "p-value", "hover_n": "N",
               "note": "N = 12 months · < 2% price variation in 2025 · Inconclusive by design"},
        "it": {"title": "Specificazioni Econometriche — Elasticità Prezzo-Domanda",
               "x": "Elasticità stimata (β log-log)",
               "espurio": "⚠️ Risultato spurio (VIF prezzo↔trend = 6,2)",
               "nosig": "Non significativo",
               "hover_e": "Elasticità", "hover_p": "p-valore", "hover_n": "N",
               "note": "N = 12 mesi · Variazione prezzo < 2% nel 2025 · Inconcludente per progetto"},
    }
    L = labels.get(lang, labels["es"])

    fig = go.Figure()
    fig.add_vline(x=0, line_color=c["border"], line_width=1.5, line_dash="dash")

    colors_spec = {True: TM_AMARILLO, False: _C_INSIG}
    symbols_spec = {True: "diamond-open", False: "circle"}

    for name, elast, pval, espurio in _COMBUSTIBLE_SPECS:
        color = colors_spec[espurio]
        symbol = symbols_spec[espurio]
        sig_label = L["espurio"] if espurio else f"p = {pval:.3f} — {L['nosig']}"
        ht = (f"<b>{name[:45]}...</b><br>"
              f"{L['hover_e']}: <b>{elast:.2f}</b><br>"
              f"{L['hover_p']} = {pval:.3f}<br>"
              f"{sig_label}")
        # Usar ±0 como CI (no disponibles) pero marcar visualmente
        fig.add_trace(go.Scatter(
            x=[elast], y=[name[:50]],
            mode="markers",
            marker=dict(size=16, color=color, symbol=symbol,
                        line=dict(color=color, width=2)),
            hovertext=ht, hoverinfo="text",
            showlegend=False,
        ))

    # Anotación de colinealidad
    fig.add_annotation(
        x=10.44, y=_COMBUSTIBLE_SPECS[0][0][:50],
        text="Espuria<br>r(precio,trend)=0.92",
        showarrow=True, arrowhead=2,
        arrowcolor=TM_AMARILLO, arrowwidth=1.5,
        font=dict(size=9, color=TM_AMARILLO),
        ax=40, ay=-30,
        bgcolor=c["card_bg"],
        bordercolor=TM_AMARILLO, borderwidth=1, borderpad=3,
    )

    fig.update_xaxes(title_text=L["x"])
    fig.update_yaxes(autorange="reversed",
                     tickfont=dict(size=9))
    fig.update_layout(
        title=dict(text=L["title"], font=dict(size=13, color=c["text_main"])),
        annotations=[dict(x=0.5, y=-0.18, xref="paper", yref="paper",
                          text=L["note"], showarrow=False,
                          font=dict(size=9, color=c["text_muted"]), align="center")],
    )
    return _theme(fig, c)


# ════════════════════════════════════════════════════════════════════════════
# ── CONTROL SINTÉTICO — CIUDAD BOLÍVAR ──────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════

# Datos extraídos del notebook 1.06-ae-ciudad-bolivar-synth-control.ipynb
# Tratada: línea 40 (Zona T Ciudad Bolívar)
# Suspensión: 2025-10-04 → 2025-10-19 (16 días)

_SC_WEIGHTS = [
    ("Zona F - Eje Ambiental", 0.0810),
    ("Zona H - Usme / NQS Sur", 0.5765),
    ("Zona K - Caracas Sur", 0.3425),
]

_SC_RMSPE_RATIOS = [
    ("Zona T Ciudad Bolívar", 15.90, True),
    ("Zona E - NQS Central", 3.21, False),
    ("Zona L - Av. 68", 2.45, False),
    ("Zona H - Usme / NQS Sur", 2.18, False),
    ("Zona D - Suba", 2.05, False),
    ("Zona K - Caracas Sur", 1.89, False),
    ("Zona G - Calle 80", 1.75, False),
    ("Zona B - AutoNorte", 1.62, False),
    ("Zona A - Caracas", 1.48, False),
    ("Zona J - AutoSur", 1.35, False),
    ("Zona F - Eje Ambiental", 1.22, False),
    ("Zona C - Americas / Calle 26", 1.10, False),
]

_SC_LOSS_DAYS = [
    ("Oct 04", -56800), ("Oct 05", -38200), ("Oct 06", -14500),
    ("Oct 07", -56100), ("Oct 08", -55900), ("Oct 09", -55700),
    ("Oct 10", -55400), ("Oct 11", -42100), ("Oct 12", -15800),
    ("Oct 13", -20100), ("Oct 14", -55300), ("Oct 15", -55100),
    ("Oct 16", -54800), ("Oct 17", -54600), ("Oct 18", -32400),
    ("Oct 19", -13200),
]

_SC_TOTAL_LOST = 750000
_SC_DAILY_AVG_LOST = 46900
_SC_PCT_LOST = 91.8
_SC_P_VALUE = 0.000


def build_sc_weights(c: dict, lang: str = "es") -> go.Figure:
    """Barras horizontales: composición del control sintético (pesos w_j)."""
    labels = {
        "es": {"title": "Composición del Control Sintético — Zona T Ciudad Bolívar",
               "x": "Peso en el sintético (w)",
               "hover_w": "Peso",
               "note": "Optimización SLSQP en símplex · Pre-período: ene-sep 2025 (276 días)"},
        "en": {"title": "Synthetic Control Composition — Zona T Ciudad Bolívar",
               "x": "Weight in synthetic (w)",
               "hover_w": "Weight",
               "note": "SLSQP optimization on simplex · Pre-period: Jan-Sep 2025 (276 days)"},
        "it": {"title": "Composizione del Controllo Sintetico — Zona T Ciudad Bolívar",
               "x": "Peso nel sintetico (w)",
               "hover_w": "Peso",
               "note": "Ottimizzazione SLSQP su simplesso · Pre-periodo: gen-set 2025 (276 giorni)"},
    }
    L = labels.get(lang, labels["es"])

    donors = sorted(_SC_WEIGHTS, key=lambda x: x[1])
    names = [d[0] for d in donors]
    weights = [d[1] for d in donors]

    hover = [
        f"<b>{n}</b><br>{L['hover_w']}: <b>{w:.4f}</b> ({w*100:.1f}%)"
        for n, w in zip(names, weights)
    ]

    fig = go.Figure(go.Bar(
        x=weights, y=names, orientation="h",
        marker=dict(color=[TM_ROJO if w > 0.3 else "#3B82F6" for w in weights],
                    opacity=0.85),
        hovertext=hover, hoverinfo="text",
        showlegend=False,
    ))

    for n, w in zip(names, weights):
        fig.add_annotation(x=w, y=n, text=f"{w:.1%}", showarrow=False,
                           xshift=25, font=dict(size=11, color=c["text_main"]))

    fig.update_xaxes(title_text=L["x"], tickformat=".0%", range=[0, 0.7])
    fig.update_layout(
        title=dict(text=L["title"], font=dict(size=13, color=c["text_main"])),
        annotations=[dict(x=0.5, y=-0.22, xref="paper", yref="paper",
                          text=L["note"], showarrow=False,
                          font=dict(size=9, color=c["text_muted"]), align="center")],
    )
    return _theme(fig, c)


def build_sc_rmspe(c: dict, lang: str = "es") -> go.Figure:
    """Barras horizontales: razón RMSPE post/pre por troncal (inferencia por aleatorización)."""
    labels = {
        "es": {"title": "Inferencia por Aleatorización — Razón RMSPE post/pre",
               "x": "Razón RMSPE post / pre",
               "hover_r": "Razón", "hover_p": "p-valor",
               "note": f"p-valor = {_SC_P_VALUE:.3f} · Zona T supera todos los placebos"},
        "en": {"title": "Randomization Inference — RMSPE post/pre Ratio",
               "x": "RMSPE post / pre ratio",
               "hover_r": "Ratio", "hover_p": "p-value",
               "note": f"p-value = {_SC_P_VALUE:.3f} · Zona T exceeds all placebos"},
        "it": {"title": "Inferenza per Randomizzazione — Rapporto RMSPE post/pre",
               "x": "Rapporto RMSPE post / pre",
               "hover_r": "Rapporto", "hover_p": "p-valore",
               "note": f"p-valore = {_SC_P_VALUE:.3f} · Zona T supera tutti i placebos"},
    }
    L = labels.get(lang, labels["es"])

    sorted_data = sorted(_SC_RMSPE_RATIOS, key=lambda x: x[1])
    names = [d[0] for d in sorted_data]
    ratios = [d[1] for d in sorted_data]
    is_treated = [d[2] for d in sorted_data]
    colors = [TM_ROJO if t else "#3B82F6" for t in is_treated]

    hover = [
        f"<b>{n}</b><br>{L['hover_r']}: <b>{r:.2f}</b>"
        + (f"<br>{L['hover_p']} = {_SC_P_VALUE:.3f}" if t else "")
        for n, r, t in zip(names, ratios, is_treated)
    ]

    fig = go.Figure(go.Bar(
        x=ratios, y=names, orientation="h",
        marker=dict(color=colors, opacity=0.85),
        hovertext=hover, hoverinfo="text",
        showlegend=False,
    ))

    treated_ratio = next(r for _, r, t in _SC_RMSPE_RATIOS if t)
    fig.add_vline(x=treated_ratio, line_color=TM_ROJO, line_width=1.5, line_dash="dot")

    fig.update_xaxes(title_text=L["x"])
    fig.update_layout(
        title=dict(text=L["title"], font=dict(size=13, color=c["text_main"])),
        annotations=[dict(x=0.5, y=-0.22, xref="paper", yref="paper",
                          text=L["note"], showarrow=False,
                          font=dict(size=9, color=c["text_muted"]), align="center")],
    )
    return _theme(fig, c)


def build_sc_losses(c: dict, lang: str = "es") -> go.Figure:
    """Barras verticales: pérdida diaria de validaciones durante la suspensión."""
    labels = {
        "es": {"title": "Pérdida Diaria de Validaciones — Suspensión Zona T",
               "x": "Fecha (octubre 2025)", "y": "Validaciones perdidas",
               "hover_d": "Fecha", "hover_l": "Pérdida",
               "note": f"Pérdida total: ~{_SC_TOTAL_LOST:,} · Media/día: ~{_SC_DAILY_AVG_LOST:,} · -{_SC_PCT_LOST:.1f}% vs. contrafactual"},
        "en": {"title": "Daily Validation Loss — Zona T Suspension",
               "x": "Date (October 2025)", "y": "Validations lost",
               "hover_d": "Date", "hover_l": "Loss",
               "note": f"Total loss: ~{_SC_TOTAL_LOST:,} · Daily avg: ~{_SC_DAILY_AVG_LOST:,} · -{_SC_PCT_LOST:.1f}% vs. counterfactual"},
        "it": {"title": "Perdita Giornaliera di Validazioni — Sospensione Zona T",
               "x": "Data (ottobre 2025)", "y": "Validazioni perse",
               "hover_d": "Data", "hover_l": "Perdita",
               "note": f"Perdita totale: ~{_SC_TOTAL_LOST:,} · Media/giorno: ~{_SC_DAILY_AVG_LOST:,} · -{_SC_PCT_LOST:.1f}% vs. controffattuale"},
    }
    L = labels.get(lang, labels["es"])

    dates = [d[0] for d in _SC_LOSS_DAYS]
    losses = [abs(d[1]) for d in _SC_LOSS_DAYS]

    hover = [
        f"<b>{L['hover_d']}: {d}</b><br>{L['hover_l']}: <b>-{val:,.0f}</b>"
        for d, val in zip(dates, losses)
    ]

    fig = go.Figure(go.Bar(
        x=dates, y=losses,
        marker=dict(color=TM_ROJO, opacity=0.75),
        hovertext=hover, hoverinfo="text",
        showlegend=False,
    ))

    fig.add_hline(y=_SC_DAILY_AVG_LOST, line_color=TM_AMARILLO, line_width=2,
                  line_dash="dash", annotation_text=f"Media: {_SC_DAILY_AVG_LOST/1e3:.1f}k",
                  annotation_font=dict(color=TM_AMARILLO, size=10),
                  annotation_position="top right")

    fig.update_xaxes(title_text=L["x"], tickangle=-45)
    fig.update_yaxes(title_text=L["y"], tickformat=",d")
    fig.update_layout(
        title=dict(text=L["title"], font=dict(size=13, color=c["text_main"])),
        annotations=[dict(x=0.5, y=-0.28, xref="paper", yref="paper",
                          text=L["note"], showarrow=False,
                          font=dict(size=9, color=c["text_muted"]), align="center")],
    )
    return _theme(fig, c)
