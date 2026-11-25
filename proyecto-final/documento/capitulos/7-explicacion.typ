= Tipificación y explicación de errores (POS tagging)

Esta etapa convierte la salida de BART en lo que distingue al tutor de un corrector: el tipo de cada error y su explicación. En términos de Ashmore et al. forma parte del despliegue. Es la integración del modelo en un sistema que sea _adecuado al propósito_ (explicar, no solo corregir) y _tolerado_ por el usuario, porque dice cuándo no está seguro. Todo vive en `src/explicar.py` y el notebook 06 lo recorre con la oración de ejemplo, de la que ahora tomamos las dos frases completas:

#block(inset: (x: 10pt, y: 7pt), fill: luma(248), stroke: 0.5pt + luma(210), radius: 2pt, width: 100%)[
  #set text(size: 9.6pt)
  *Original:* As you know I am in charge for organise the conference in my college. Do not worry if Richard did not left you all the information. \
  *BART-large:* As you know I am in charge of organising the conference in my college. Do not worry if Richard did not give you all the information. \
  *Referencia:* As you know, I am in charge of organising the conference in my college. Do not worry if Richard did not leave you all the information.
]

== Paso 1: POS tagging de las dos oraciones

ERRANT analiza la original y la corregida con spaCy (`en_core_web_sm`). Para cada palabra obtiene la categoría universal (`pos`), la etiqueta fina del Penn Treebank (`tag`, que distingue `VB` de `VBG`) y el lema (@tab-pos).

#figure(
  table(
    columns: (auto,) + (1fr,) * 6,
    align: (left,) + (center,) * 6,
    table.header[][in][charge][*for* / *of*][*organise* / *organising*][the][conference],
    [`pos`], [ADP], [NOUN], [ADP / ADP], [NOUN / VERB], [DET], [NOUN],
    [`tag`], [IN], [NN], [IN / IN], [NN / VBG], [DT], [NN],
    [lema], [in], [charge], [for / of], [organise / organise], [the], [conference],
  ),
  caption: [Etiquetas de spaCy en la parte que cambia (original / corregida).],
) <tab-pos>

El error confunde al etiquetador: después de _for_, lee _organise_ como sustantivo (`NN`). Por eso ERRANT no tipa con una sola oración. Compara las dos, y lo decisivo es el *lema*: es el mismo en ambas.

== Paso 2: alineación y tipo

ERRANT alinea las secuencias como un _diff_, pero con costos que prefieren emparejar palabras con el mismo lema o la misma categoría. Después tipa cada diferencia con reglas sobre esas etiquetas:

- _for → of_: las dos son `ADP`, así que el tipo es `R:PREP`.
- _organise → organising_: mismo lema y el corregido es `VERB/VBG`. Es el mismo verbo en otra forma: `R:VERB:FORM`.
- _left → give_: dos verbos de lema distinto: `R:VERB`, elección de palabra.

Medimos qué tan estable es este paso (notebook 02, análisis P2). Lejos del error, solo el 0,8 % de las palabras que no se editan cambian de POS entre la oración original y la corregida; junto al error, el 2,9 %. `en_core_web_lg` mejora esa cifra en apenas 0,04 puntos y es más lento, así que nos quedamos con `sm`, el mismo modelo de la evaluación oficial.

== Paso 3: plantilla en español

El tipo elige una de las *28 plantillas* de `explicar.PLANTILLAS`. Si el tipo no tiene plantilla propia, se usa la genérica de su operación: _falta_, _sobra_ o _cambia_. Las plantillas vuelven a usar el POS tagging para el detalle. Por ejemplo, la de forma verbal lee la etiqueta fina del verbo corregido para nombrar la forma y mira la palabra anterior:

```python
def _forma_verbal(e):
    c = _verbo(e.c_toks)
    forma = FORMAS.get(c.tag_, "otra forma")       # VBG → "la forma en -ing"
    texto = f"Aquí va {forma} del verbo: «{e.c_str}» en lugar de «{e.o_str}»."
    anterior = _anterior(e.c_toks)                  # la palabra antes del verbo
    if c.tag_ == "VBG" and anterior is not None and anterior.pos_ == "ADP":
        texto += (f" Después de una preposición («{anterior.text}»)"
                  " el verbo va en -ing.")
    elif c.tag_ == "VB" and anterior is not None and (anterior.lower_ == "to"
                                                     or anterior.tag_ == "MD"):
        texto += (f" Después de «{anterior.text}» va la forma base,"
                  " sin -s, -ing ni -ed.")
    return texto
```

Los tipos en que los hispanohablantes se equivocan más llevan el contraste con el español. `M:PRON` explica: «Falta el sujeto «It». En inglés el sujeto no se omite aunque el verbo ya lo indique («es grande» → «it is big»)». `R:PREP` explica que las preposiciones no se traducen palabra por palabra. Para la oración de ejemplo, el sistema produce:

#block(inset: (x: 10pt, y: 7pt), fill: luma(248), stroke: 0.5pt + luma(210), radius: 2pt, width: 100%)[
  #set text(size: 9.6pt)
  [media] `R:PREP` _for → of_ --- La preposición correcta es «of», no «for». Las preposiciones no se traducen palabra por palabra del español: conviene aprenderlas junto con la palabra que las rige. \
  [alta] `R:VERB:FORM` _organise → organising_ --- Aquí va la forma en -ing del verbo: «organising» en lugar de «organise». Después de una preposición («of») el verbo va en -ing. \
  [baja] `R:VERB` _left → give_ --- Sugerencia de vocabulario: «give» (verbo) expresa mejor la idea que «left».
]

== Confianza y cobertura

Cada explicación lleva una *confianza*, tomada del F0.5 por tipo de la verificación (sección 6.3). Es *alta* para los tipos con F0.5 ≥ 0,63 (ortografía, concordancia, forma verbal, número, morfología, mayúsculas, artículos) y *baja* para la elección de palabras y las reescrituras (F0.5 ≤ 0,32). Los demás tipos quedan en *media*. La app muestra la confianza baja como _Sugerencia_, no como error. En la oración de ejemplo, la confianza baja acierta. El modelo cambió _left_ por _give_, cuando la referencia pide _leave_: el error era la forma verbal (_did not left_), no la palabra. La sugerencia es gramatical, pero cambia el sentido, y el estudiante la ve marcada como dudosa.

Medimos la cobertura sobre lo que el sistema realmente va a explicar: las 2.890 ediciones que ERRANT extrae de las salidas de BART-large en FCE test (1.405 oraciones cambiadas, 51 tipos distintos). *El 91,4 % recibe plantilla propia* y el 8,6 % restante, una genérica. Por confianza, el 43 % de esas ediciones salen como alta, el 39 % como media y el 18 % como baja.

#figure(
  image("../../figuras/19_cobertura_plantillas.png", width: 58%),
  caption: [Ediciones de BART-large en FCE test por tipo, con y sin plantilla propia.],
) <fig-cobertura>
