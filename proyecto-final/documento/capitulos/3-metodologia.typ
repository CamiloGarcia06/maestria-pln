= Metodología y pipeline de PLN

== El ciclo de vida de Ashmore et al.

#cite(<ashmore2021>, form: "prose") dividen el ciclo de vida de un sistema de aprendizaje de máquina en cuatro etapas. Para cada una proponen _desiderata_, propiedades que el resultado de la etapa debe cumplir para que el sistema sea confiable. Usamos esas etapas como esqueleto del proyecto y de este documento. Cada análisis que hicimos tenía que terminar en una decisión para alguna etapa, no en una estadística suelta.

#figure(
  table(
    columns: (auto, 1fr, 1.25fr, auto),
    align: (left, left, left, center),
    table.header[Etapa][Desiderata][Qué la cumple en el proyecto][Cap.],
    [Gestión de datos], [relevantes, completos, balanceados, precisos],
      [corpus de estudiantes en examen; análisis de los hispanohablantes; filtros de `UNK` y duplicados; ERRANT reproduce la anotación], [4],
    [Aprendizaje], [con buen desempeño, robustos, reutilizables, interpretables],
      [afinado de BART con selección en dev por F0.5; cuatro corridas comparadas; probabilidad por pieza], [5],
    [Verificación], [completa, relevante para el contexto, comprensible],
      [test intocado; resultados aparte para hispanohablantes; F0.5 por tipo de error; techo de la evaluación], [6],
    [Despliegue], [adecuado al propósito, tolerado, adaptable],
      [explicación en español con nivel de confianza; app web de \~120 ms por oración; modelo intercambiable], [7--8],
  ),
  caption: [Etapas y desiderata de @ashmore2021 aplicados al tutor.],
) <tab-ashmore>

== El pipeline de punta a punta

La @fig-pipeline muestra el recorrido de los datos. Del lado del entrenamiento, los archivos oficiales se convierten en pares de oraciones y luego en piezas BPE con las que se afina BART. Del lado del uso, el párrafo del estudiante pasa por el mismo modelo, por ERRANT y por las plantillas de explicación. ERRANT aparece dos veces: tipifica las ediciones en la aplicación y las cuenta en la evaluación.

#let caja(titulo, detalle, color: rgb("#A4262C")) = box(
  width: 100%, inset: (x: 5pt, y: 5pt), radius: 2pt, fill: white,
  stroke: (paint: luma(190), thickness: 0.5pt), [
    #block(stroke: (left: 2pt + color), inset: (left: 4pt), spacing: 0pt)[
      #text(size: 8.4pt, weight: "bold", titulo) \
      #text(size: 7.4pt, fill: luma(60), detalle)
    ]
  ])
#let flecha = align(center + horizon, text(size: 11pt, fill: luma(110), sym.arrow.r))
#let etapa(t) = text(size: 7pt, tracking: 0.08em, fill: luma(110), upper(t))

#figure(
  block(width: 100%, inset: 6pt, fill: luma(248), radius: 3pt)[
    #grid(
      columns: (1fr, 10pt, 1fr, 10pt, 1fr, 10pt, 1fr),
      row-gutter: 5pt,
      etapa[gestión de datos], [], [], [], etapa[aprendizaje], [], [],
      caja[1 · Corpus][FCE + W&I (BEA-2019): M2 + JSON, 71.928 oraciones], flecha,
      caja[2 · Pares][aplicar ediciones del anotador 0 y destokenizar], flecha,
      caja[3 · Filtros][sin `UNK` ni duplicados: 56.254 pares de train], flecha,
      caja[4 · BART afinado][BPE de 128 piezas; AdamW; selección por F0.5 en dev],
    )
    #v(4pt)
    #grid(
      columns: (1fr, 10pt, 1fr, 10pt, 1fr, 10pt, 1fr),
      row-gutter: 5pt,
      etapa[despliegue], [], [], [], [], [], etapa[verificación],
      caja(color: rgb("#2f6f8f"))[8 · App web][FastAPI; tacha, subraya y explica cada cambio], text(size: 11pt, fill: luma(110), align(center + horizon, sym.arrow.l)),
      caja(color: rgb("#2f6f8f"))[7 · Explicación][28 plantillas en español + genérica; confianza por tipo], text(size: 11pt, fill: luma(110), align(center + horizon, sym.arrow.l)),
      caja(color: rgb("#2f6f8f"))[6 · ERRANT + spaCy][POS y lemas → ediciones con tipo (`R:PREP`, `M:PRON`…)], text(size: 11pt, fill: luma(110), align(center + horizon, sym.arrow.l)),
      caja(color: rgb("#4f7a3a"))[5 · Evaluación][F0.5 de ERRANT en FCE test y en hispanohablantes],
    )
  ],
  kind: image,
  caption: [Pipeline del tutor. Rojo: entrenamiento; azul: uso en la aplicación; verde: verificación. Los pasos 4 → 6 → 7 → 8 forman el camino que recorre el párrafo del estudiante.],
) <fig-pipeline>

La @tab-pipeline resume qué hace cada paso, con qué código y qué le pasa a una oración real. Usamos siempre la misma oración, escrita por un hispanohablante en el test de FCE; el modelo nunca la vio durante el entrenamiento.

#figure(
  table(
    columns: (1.05fr, 1.1fr, 1.45fr),
    table.header[Paso y código][Transformación][La oración de ejemplo],
    [1 · `corpus.leer_m2`], [archivo M2 → tokens + ediciones con tipo y anotador], [`S As you know I am in charge for organise …` y cuatro líneas `A`],
    [2 · `corpus.aplicar_ediciones`, `texto.destokenizar`], [tokens + ediciones → par original / corregida en texto natural], [`… in charge for organise …` → `… in charge of organising …`],
    [3 · `preparar.pares_train`], [quita pares con `UNK` y duplicados], [se conserva: no tiene `UNK` ni está repetida],
    [4 · tokenizador de BART], [texto → ids BPE (`input_ids`, `labels`)], [`Ġfor` = 13, `Ġorganise` = 23750 → `Ġof` = 9, `Ġorganising` = 22821],
    [4 · `Corrector.corregir`], [ids → oración generada con _beam search_], [`As you know I am in charge of organising the conference …`],
    [5 · `evaluacion.f05`], [hipótesis + referencia M2 → TP, FP, FN, F0.5], [2 TP (`of`, `organising`); 1 FP (_left_ → _give_); 2 FN (la coma y _leave_)],
    [6--7 · `explicar.explicar`], [par de oraciones → ediciones con tipo y explicación], [`R:PREP`: «La preposición correcta es «of»…»],
    [8 · `POST /api/corregir`], [párrafo → JSON con oraciones, posiciones y explicaciones], [la página tacha _for_ y subraya _of_],
  ),
  caption: [El pipeline paso a paso, con el módulo de `src/` que lo implementa y una oración real.],
) <tab-pipeline>
