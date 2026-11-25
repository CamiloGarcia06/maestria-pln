= Verificación

Ashmore et al. piden que la verificación sea _completa_ (que mida en datos que no se usaron para decidir), _relevante para el contexto_ (que mida en las condiciones del usuario real) y _comprensible_. Por eso el notebook 05 mide *una sola vez* el modelo elegido, en tres conjuntos que nadie usó para decidir nada, y reporta aparte a los hispanohablantes.

== Líneas base y techo

Un número de F0.5 suelto no dice mucho, así que lo leemos entre dos extremos medidos con el mismo camino de evaluación:

- *Copiar* (devolver la oración sin cambios): F0.5 = 0. Es la línea base que hay que superar.
- *Techo*: la corrección de referencia, destokenizada y pasada por spaCy y ERRANT, da F0.5 = 0,855 en FCE test, no 1. Al volver a extraer las ediciones, ERRANT no siempre las corta igual que el anotador. Ningún sistema puede pasar de ese valor en esta evaluación, así que el resultado del modelo se lee contra ese techo.

La evaluación de cualquier salida son dos comandos de ERRANT:

```bash
# extrae y tipa las ediciones de la hipótesis
errant_parallel -orig orig.txt -cor hyp.txt -out hyp.m2
# las cruza con la referencia: TP, FP, FN, P, R, F0.5
errant_compare  -hyp hyp.m2 -ref ref.m2
```

== Resultados

#figure(
  table(
    columns: (1.6fr, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right),
    table.header[Conjunto][Oraciones][Copiar][BART-base][BART-large][Techo],
    [FCE test, todos los autores], [2.695], [0], [0,493], [*0,527*], [0,855],
    [FCE dev, hispanohablantes], [533], [0], [0,483], [*0,520*], [0,844],
    [FCE test, hispanohablantes], [224], [0], [0,465], [*0,515*], [0,853],
  ),
  caption: [F0.5 en los conjuntos de verificación. BART-base se mide como referencia; la decisión ya estaba tomada en dev.],
) <tab-resultados>

*BART-large logra F0.5 = 0,527 en FCE test* (TP 1.699, FP 1.197, FN 2.850; precisión 0,587, recall 0,373). Es casi lo mismo que en dev (0,531), así que la selección no sobreajustó a dev. Supera a BART-base en los tres conjuntos, por 3 a 5 puntos, y recorre el 62 % del camino entre copiar (0) y el techo (0,855). En los hispanohablantes, el usuario final, el resultado baja poco: 0,515 en test y 0,520 en dev. Su test tiene más errores por oración y difiere más de train (sección 4.3). La precisión se mantiene en 0,57--0,59 en todos los conjuntos: cuando el tutor propone un cambio, casi seis de cada diez veces coincide con el anotador. Muchos de los otros cuatro son alternativas válidas que la referencia no incluye, como las comas opcionales.

== Resultados por tipo de error

`evaluacion.f05_por_tipo` corre `errant_compare -cat 2`, que cruza las mismas ediciones pero las agrupa por tipo. Esto conecta la verificación con la explicación: el F0.5 de cada tipo decide el nivel de confianza que muestra la aplicación (capítulo 7).

#figure(
  image("../../figuras/18_f05_por_tipo.png", width: 68%),
  caption: [F0.5 de BART-large por tipo de error en FCE test (tipos con al menos 40 errores en la referencia).],
) <fig-tipos>

Los resultados se reparten en tres franjas:

- *Fuertes* (F0.5 ≥ 0,63): ortografía (0,78), concordancia sujeto--verbo (0,76), forma verbal (0,71), singular/plural (0,68), morfología (0,66), mayúsculas (0,64) y artículos (0,63). Son errores con una regla clara, donde el contexto basta para decidir.
- *Medios* (0,4--0,6): preposiciones (0,60), pronombres (0,46), tiempo verbal (0,45), puntuación (0,45) y orden de palabras (0,42). La puntuación es el caso típico: acierta 206 pero propone 254 cambios que la referencia no tiene, porque muchas comas son opcionales.
- *Débiles* (< 0,35): elegir otra palabra (sustantivo 0,20, adjetivo y adverbio 0,27, verbo 0,32) y las reescrituras (`OTHER`, 0,28). Aquí la corrección depende del significado y hay muchas respuestas válidas.

En los hispanohablantes se repite el patrón: ortografía 0,77, forma verbal 0,80, preposiciones 0,58. Son justamente tipos en que ellos se equivocan más (capítulo 1).

== Ejemplos y comportamiento con oraciones correctas

En `fce_test_es` (224 oraciones), el modelo deja exactamente como el anotador 24 oraciones con errores (_I sugget you to go_ → _I suggest you go_; _Is situated … part on the city_ → _It is situated … part of the city_). Corrige parte de los errores en 90. En 43 deja la oración igual aunque la referencia la cambia. Esos 43 casos son de dos clases. Unos son reescrituras de estilo (_walking_ → _on foot_), donde preferimos que no toque. Otros son errores simples que debía ver (_Musseum_, _an historical_): ese es el margen de mejora real.

Para el usuario importa sobre todo lo que pasa con las oraciones *correctas*. De las 67 oraciones sin error de los hispanohablantes, el modelo *deja intactas 61 (91 %)*. Solo en 6 propone un cambio que la referencia no pide. Un tutor que «corrige» lo que está bien enseña reglas falsas, y este comportamiento es el que buscábamos al elegir F0.5.
