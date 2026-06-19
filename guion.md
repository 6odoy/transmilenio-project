# Guión de exposición — 20 minutos
## ¿Qué mueve la demanda de TransMilenio? Seis estudios causales sobre el sistema BRT más grande de América Latina
### 2026 Urban Data Science Summer School — Trento, Italia

**Expositores:** A (Andrea) · B (segundo expositor)
**Duración total:** 20 minutos + preguntas
**Distribución:** A abre, presenta el sistema y la arquitectura de datos; B lidera la inferencia causal; A cierra con el dashboard y conclusiones.

---

## BLOQUE 0 — Apertura [A · 1 min]

> *Slide: título + foto aérea de Bogotá con las troncales superpuestas*

**A:** Buenos días. Vamos a hablarles de TransMilenio: el sistema de Bus Rapid Transit de Bogotá, con 152 estaciones, 12 líneas troncales y —en 2025— 1 230 millones de validaciones. Es el BRT más grande de América Latina y uno de los más instrumentados del mundo: cada pasada de tarjeta queda registrada con estación, línea, hora y sentido.

La pregunta que nos hacemos es sencilla de formular y difícil de responder: **¿qué perturba esa demanda, con qué magnitud, y qué no tiene efecto identificable?** Responderla bien importa para operaciones, para planificación de nuevas troncales y para política de movilidad.

---

## BLOQUE 1 — El sistema y los datos [A · 3 min]

> *Slide: mapa del sistema + tabla de volúmenes*

**A:** El sistema tiene una estructura clara. Doce troncales corren de norte a sur o de oriente a occidente; en los extremos hay portales de intercambio modal con rutas alimentadoras. Cada troncal tiene entre 8 y 22 estaciones.

Los datos provienen de Google Cloud Storage: TransMilenio S.A. publica mensualmente un ZIP con todos los registros de validación. El archivo más reciente tiene cuatro columnas que nos interesan: fecha, hora, estación y número de entradas y salidas.

> *Slide: diagrama del pipeline ETL*

El primer reto fue técnico. Los archivos mensuales suman unos 4,6 millones de filas al año, y las estaciones tienen variantes históricas de código —en algunos casos la misma estación aparece con hasta tres identificadores distintos. Construimos un pipeline en Polars con evaluación diferida que normaliza los códigos, consolida variantes y persiste el resultado en Parquet particionado por mes. Ese Parquet es la base de todos los análisis que siguen.

> *Slide: fragmento de código del ETL o esquema de columnas*

**A:** Con esa base limpia, la agenda del día es esta: seis choques, seis métodos, seis respuestas. Le paso la palabra a [B] para que las recorra.

---

## BLOQUE 2 — Marco de identificación causal [B · 2 min]

> *Slide: tabla resumen de los seis estudios*

**B:** Gracias. Antes de entrar en cada estudio, conviene decir por qué usamos inferencia causal y no solo correlación.

El problema estándar es que la demanda de transporte público correlaciona con casi todo: llueve más en invierno, en invierno hay más festivos, los festivos coinciden con períodos vacacionales. Si no separamos los efectos, cualquier coeficiente mezcla múltiples causas.

La estrategia que adoptamos es la de la **credibilidad**: para cada choque buscamos una fuente de variación exógena —algo que cambie el tratamiento pero no esté correlacionado con factores de demanda no observados. Cuando el diseño no permite un contrafactual limpio, lo decimos explícitamente.

La tabla que ven en pantalla resume los seis estudios. Los efectos están todos en escala logarítmica para ser comparables entre sí.

| # | Choque | Método | Efecto central |
|---|--------|--------|----------------|
| 1 | Festivos nacionales | Event study | −68% el día festivo |
| 2 | Lluvia y temperatura | Panel FE | −1,1% por día lluvioso; −2,2% por +1 °C |
| 3 | Ciclovía dominical | DiD espacial | ~0% (nulo) |
| 4 | Conciertos en Campín | Event study horario | +45% a +844% según hora y tipo |
| 5 | Precio de la gasolina | Series de tiempo | Inconcluso |
| 6 | Suspensión troncal Ciudad Bolívar | Control sintético | −91,8% · 750k validaciones perdidas |

Vamos uno a uno con los más interesantes.

---

## BLOQUE 3 — Festivos: el choque más limpio [B · 2 min]

> *Slide: gráfico event_study_festivos.png*

**B:** El estudio de festivos es el más limpio metodológicamente porque los festivos colombianos están fijados por ley con años de anticipación. Son exógenos por construcción.

Usamos un event study de ±5 días alrededor de cada festivo, con efectos fijos de estación y día de semana:

$$\log(\text{total}_{it}) = \sum_{k=-5}^{+5} \beta_k \cdot \mathbf{1}[t - \text{festivo} = k] + \alpha_i + \delta_{\text{dow}} + \varepsilon_{it}$$

El resultado es inequívoco: el día festivo la demanda cae un **68%**. Lo interesante no es eso —es lo que ocurre a los lados. El día anterior al festivo se anticipa parte del viaje, y el día siguiente hay un rebote negativo de −12% que interpretamos como recuperación incompleta de la movilidad.

Cuando desagregamos por tipo de festivo, los religiosos tienen caídas más pronunciadas que los cívicos. Eso tiene sentido: en un festivo religioso las personas permanecen en casa; en un festivo cívico hay más eventos que generan desplazamientos alternativos.

---

## BLOQUE 4 — Ciclovía: un resultado nulo informativo [B · 2 min]

> *Slide: ciclovia_mapa_tratamiento.png + ciclovia_parallel_trends.png*

**B:** La Ciclovía es el experimento natural más tentador de Bogotá. Cada domingo, el IDRD cierra 127 km de vías al tráfico motorizado entre las 7am y las 2pm. La pregunta obvia es: ¿esas personas se bajan del carro y se suben al bus?

El diseño es una diferencia en diferencias espacial. Clasificamos las estaciones en tratadas —a menos de 500 metros de una ruta de Ciclovía— y control. La fuente de variación es el domingo con Ciclovía activa frente a otros días.

$$\log(\text{total}_{it}) = \beta \cdot (\text{sunday}_t \times \text{cerca}_i) + \alpha_i + \delta_{\text{dow}} + \delta_{\text{mes}} + \varepsilon_{it}$$

El coeficiente $\beta$ es −8,4%, con un intervalo de confianza que cruza cero holgadamente (p = 0,14). Mismo resultado con la especificación horaria dentro del domingo.

**La conclusión es que la Ciclovía no desplaza demanda hacia TransMilenio.** Los usuarios de Ciclovía no son sustitutos del bus; son principalmente ciclistas recreativos o peatones que en domingo de todas formas no usarían el sistema. Este resultado nulo es relevante para la planificación: no esperen que los domingos sin carro llenen el BRT.

---

## BLOQUE 5 — Campín: choques puntuales de demanda [B · 3 min]

> *Slide: campin_deteccion_eventos.png + campin_conciertos_vs_grandes_eventos.png*

**B:** El Estadio El Campín y el Movistar Arena están a menos de 300 metros de dos estaciones troncales. En 2025 hubo conciertos de Shakira, Andrea Bocelli, Sting, Maluma —entre otros— y el Festival de Verano en agosto.

El event study aquí es horario. En lugar de estimar el efecto día a día, estimamos hora a hora:

$$\log(\text{total}_{ih}) = \beta_h \cdot \text{evento}_d + \alpha_i + \varepsilon_{ih}$$

Los resultados son los más heterogéneos del proyecto. Para los **conciertos en arena cerrada** (Movistar Arena, capacidad ~14 000 personas), el pico es a las 3–4pm con un +45%. Para los **grandes eventos de estadio** (El Campín, capacidad ~36 000), el pico es a las 11pm con +844%.

La diferencia se explica por el horario: los conciertos de arena terminan temprano y la gente sale en transición; los eventos de estadio terminan pasada la medianoche cuando no hay alternativas de transporte privado. Interesante desde el ángulo operativo: en esas noches el sistema absorbe una demanda que triplica cualquier pico ordinario.

Detectamos también un efecto spillover: las estaciones a entre 500m y 1km del estadio muestran +20% a +30%, lo que sugiere que la congestión en las estaciones primarias expulsa usuarios hacia las más cercanas.

---

## BLOQUE 6 — Ciudad Bolívar: el contrafactual más claro [B · 3 min]

> *Slide: ciudad_bolivar_synth.png + ciudad_bolivar_placebos.png*

**B:** El estudio más robusto del proyecto es también el más dramático. El 4 de octubre de 2025, la Troncal Zona T —que sirve las laderas de Ciudad Bolívar, uno de los sectores más densamente poblados y de menores ingresos de Bogotá— suspendió operaciones. La causa fue una emergencia geotécnica en el corredor. La suspensión duró hasta el 19 de octubre.

En esos 16 días, las validaciones diarias en esa troncal cayeron de ~58 000 a menos de 600. Una caída del **91,8%**.

Aplicamos el método de control sintético de Abadie, Diamond y Hainmueller (2010). La idea es construir un contrafactual —¿cómo habría evolucionado la demanda si no hubiera habido suspensión?— como combinación ponderada del resto de las 11 troncales, eligiendo los pesos para que el período pre-tratamiento se ajuste lo mejor posible.

> *Slide: ciudad_bolivar_gap.png*

La brecha entre la troncal observada y su sintético durante el período de suspensión es de ~750 000 validaciones perdidas. Eso equivale a unos 47 000 viajes diarios que no pudieron realizarse o debieron redirigirse —en un sector donde el 78% de la población depende del transporte público.

La validez del estimador la probamos con **placebos en espacio**: aplicamos el mismo estimador a cada una de las otras 11 troncales, que no recibieron tratamiento. Si el estimador fuera ruido, algunas deberían mostrar brechas comparables. El ratio RMSPE post/pre para Ciudad Bolívar es más de 10 veces el de cualquier placebo. El resultado es estadísticamente inusual.

---

## BLOQUE 7 — Dashboard operacional [A · 2 min]

> *Slide: captura del dashboard en producción*

**A:** Paralelo al análisis causal construimos una herramienta operativa: un dashboard en Dash/Plotly desplegado en Vercel con soporte multilenguaje —español, inglés e italiano, por las necesidades de este evento.

El dashboard tiene dos páginas. La primera muestra KPIs en tiempo real: demanda diaria promedio, línea de mayor afluencia, hora pico del sistema y estación más congestionada. El mapa interactivo permite explorar la distribución geográfica de las estaciones con métricas por troncal.

> *Slide: página del simulador de demanda*

La segunda página es un simulador de demanda. Dada una fecha, hora, zona y coordenadas, un modelo de Random Forest entrenado sobre todo el 2025 predice la afluencia esperada. Este componente está orientado a la planificación de la nueva Troncal Avenida 68, que aún no tiene datos históricos.

El stack técnico es completamente reproducible: Polars para el procesamiento, scikit-learn para el modelo, Conda para el entorno, Ruff para calidad de código.

---

## BLOQUE 8 — Síntesis y conclusiones [A · 2 min]

> *Slide: synthesis_effect_sizes.png + synthesis_demand_scale.png*

**A:** Si tuviéramos que quedar con tres hallazgos, serían estos:

**Primero, los factores institucionales dominan la variación.** Los festivos por sí solos explican caídas de casi 70%. Nada en el análisis climático o de eventos se acerca a esa magnitud. Para un operador de transporte, el calendario festivo es el principal predictor de demanda atípica.

**Segundo, los eventos masivos nocturnos crean picos de demanda sin precedente.** Un partido de estadio a las 11pm puede triplicar la demanda horaria en las estaciones adyacentes. La operación actual no está dimensionada para eso, y los datos lo muestran.

**Tercero, la infraestructura es irremplazable.** La suspensión de Ciudad Bolívar demostró que no hay sustituto para una troncal en los sectores periféricos. Las alternativas informales absorbieron algo, pero 750 000 validaciones no aparecieron en ningún otro punto del sistema.

Lo que queda pendiente: el estudio de combustible resultó inconcluso por insuficiencia de datos (N=12 meses, variación de precio menor al 2%). Ese análisis requeriría series más largas o variación inter-ciudad. Y el modelo predictivo mejorará sustancialmente cuando tengamos datos de la Avenida 68.

Gracias. Quedamos disponibles para preguntas.

---

## Notas de producción

**Tiempo por bloque:**

| Bloque | Contenido | Min | Quién |
|--------|-----------|-----|-------|
| 0 | Apertura | 1 | A |
| 1 | Sistema y datos / ETL | 3 | A |
| 2 | Marco causal | 2 | B |
| 3 | Festivos | 2 | B |
| 4 | Ciclovía (DiD) | 2 | B |
| 5 | Campín (event study) | 3 | B |
| 6 | Ciudad Bolívar (sintético) | 3 | B |
| 7 | Dashboard | 2 | A |
| 8 | Síntesis y cierre | 2 | A |
| **Total** | | **20** | |

**Figuras clave a incluir en slides** (todas en `reports/figures/`):

- `synthesis_effect_sizes.png` — panel comparativo de todos los efectos (apertura y cierre)
- `event_study_festivos.png` — coeficientes ±5 días alrededor del festivo
- `ciclovia_mapa_tratamiento.png` — mapa de estaciones tratadas vs control
- `ciclovia_parallel_trends.png` — tendencias paralelas pre-tratamiento
- `campin_deteccion_eventos.png` — calendario de eventos detectados
- `campin_conciertos_vs_grandes_eventos.png` — comparación conciertos vs estadio
- `ciudad_bolivar_synth.png` — troncal observada vs sintético
- `ciudad_bolivar_placebos.png` — distribución de placebos
- `ciudad_bolivar_gap.png` — brecha acumulada
- `synthesis_demand_scale.png` — escala comparativa de los choques

**Transiciones sugeridas:**

- Bloque 1 → 2: "Con esa base limpia, la agenda del día es esta..."
- Bloque 2 → 3: "Vamos uno a uno. Empezamos con el más limpio metodológicamente..."
- Bloque 5 → 6: "El más espectacular en magnitud viene ahora..."
- Bloque 6 → 7: "Ese análisis cierra la parte causal. [A] les muestra cómo lo presentamos en producción."
- Bloque 7 → 8: "Y eso nos lleva a las conclusiones..."

**Preguntas anticipadas:**

- *¿Por qué Polars y no pandas?* — Evaluación diferida sobre 4,6M de filas; el tiempo de procesamiento cae de ~90s a ~8s.
- *¿El modelo Random Forest generaliza a la Av. 68?* — Solo con features de calendario y geografía; la calibración requerirá datos reales de las primeras semanas de operación.
- *¿Los placebos del sintético son paramétricos o de permutación?* — Permutación en espacio (aplicar el mismo estimador a cada unidad de control); no asumen distribución.
- *¿La ciclovía tiene efecto en otros modos?* — No lo medimos; ese análisis requeriría datos de bicicletas y peatones, que no están en el registro de validaciones.
