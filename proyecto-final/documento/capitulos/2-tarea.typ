= Tarea de PLN

El sistema combina dos tareas de las vistas en clase. Mantenemos los nombres que usamos desde la entrega 1. La @tab-ficha las resume y las secciones siguientes las definen con precisión.

#figure(
  table(
    columns: (auto, 1fr, 1fr),
    table.header[][Tarea principal: generación de lenguaje][Tarea complementaria: POS tagging],
    [Problema], [corregir los errores gramaticales de una oración (GEC)], [saber qué cambió y de qué tipo es cada cambio],
    [Entrada], [una oración en inglés escrita por el estudiante, con o sin errores], [la oración original y la corregida],
    [Salida], [la misma oración corregida, cambiando solo lo necesario], [lista de ediciones con su tipo (`R:PREP`, `M:PRON`…), de 55 posibles],
    [Tipo de tarea], [secuencia a secuencia (texto → texto, mismo idioma)], [etiquetado de secuencias + clasificación por reglas],
    [Modelo], [BART-large afinado completo], [spaCy `en_core_web_sm` dentro de ERRANT 3.0.2],
    [Datos], [56.254 pares de FCE y W&I (BEA-2019)], [los mismos pares, sin entrenamiento adicional],
    [Métrica], [F0.5 sobre ediciones (ERRANT)], [cobertura de plantillas y estabilidad del POS],
    [Para qué la usa el tutor], [producir el texto corregido], [elegir la explicación en español de cada cambio],
  ),
  caption: [Ficha de las dos tareas de PLN del proyecto.],
) <tab-ficha>

== Tarea principal: generación de lenguaje

La corrección gramatical (_grammatical error correction_, GEC) se plantea como *generación de secuencia a secuencia*. Dada una oración $x = (x_1, dots, x_n)$ escrita por un estudiante, el modelo genera una oración $y = (y_1, dots, y_m)$ con el mismo contenido y sin errores, cambiando solo lo necesario. El modelo aprende la distribución
$ p_theta (y | x) = product_(t=1)^m p_theta (y_t | y_(<t), x), $
y al corregir busca la $y$ más probable con _beam search_. La literatura la describe a veces como una traducción monolingüe, de «inglés con errores» a «inglés correcto» @bryant2023survey. Aun así, la tarea del curso es la generación de lenguaje: entrada y salida están en el mismo idioma, y en una oración con errores el 86 % de los tokens se copian tal cual.

Usamos BART @lewis2020bart, un transformer codificador--decodificador @vaswani2017attention preentrenado para reconstruir texto dañado (con partes borradas, tapadas o desordenadas). Corregir errores se parece mucho a ese preentrenamiento, y por eso BART es una base fuerte para GEC @katsumata2020bart. Lo *afinamos completo* con pares (oración original, oración corregida), sin _prompting_ ni capas añadidas.

== Tarea complementaria: POS tagging

La salida de BART es una oración, no una lista de errores. Para explicar hay que saber *qué* cambió y *de qué tipo* es el cambio. Eso lo hace ERRANT @bryant2017errant. Sobre las categorías gramaticales (POS), las etiquetas finas y los lemas que asigna spaCy @honnibal2020spacy, ERRANT:

+ alinea la oración original con la corregida, token por token;
+ agrupa las diferencias en *ediciones*: falta (`M`), sobra (`U`) o se reemplaza (`R`);
+ clasifica cada edición con reglas sobre el POS y el lema. Si _help_ pasa a _helped_, el lema es el mismo, el POS es `VERB` y cambia la forma, así que el tipo es `R:VERB:FORM`. Si aparece un pronombre que no estaba, es `M:PRON`. Si la palabra original no está en el diccionario, es `R:SPELL`.

Sin el POS tagging no podríamos tipificar el error ni, por lo tanto, explicarlo. ERRANT cumple además un segundo papel: es la herramienta oficial con que se evalúa la corrección (sección 2.3).

== Métrica: F0.5 sobre ediciones

Comparar la salida con la referencia token a token premiaría copiar, porque la mayoría de los tokens no cambian. Por eso GEC se evalúa sobre *ediciones* @dahlmeier2012m2 @bryant2019bea. ERRANT extrae las ediciones de la hipótesis y las cruza con las de la referencia. Una edición propuesta que también está en la referencia es un acierto (TP). Una que no está es un falso positivo (FP), y un error de la referencia que el modelo no corrige es un falso negativo (FN). Con eso se calculan
$ P = "TP" / ("TP" + "FP"), quad R = "TP" / ("TP" + "FN"), quad F_beta = ((1 + beta^2) P R) / (beta^2 P + R), quad beta = 0.5. $
Con $beta = 0.5$ la precisión pesa el doble que el recall. Es lo que necesita el usuario del capítulo 1: es preferible dejar pasar un error a «corregir» algo que estaba bien. La línea base trivial, devolver la oración sin cambios, da F0.5 = 0.

*Un ejemplo con números.* Tomamos la oración real que recorre todo el documento (capítulo 1). La referencia del anotador tiene cuatro ediciones y BART-large propone tres:

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, left, left, center),
    table.header[Edición][Referencia][BART-large][Resultado],
    [coma tras _As you know_], [`M:PUNCT` _+ ,_], [no la pone], [FN],
    [_for → of_], [`R:PREP`], [_for → of_], [TP],
    [_organise → organising_], [`R:VERB:FORM`], [_organise → organising_], [TP],
    [_left_], [`R:VERB:FORM` _→ leave_], [`R:VERB` _→ give_], [FP y FN],
  ),
  caption: [Cruce de ediciones en la oración de ejemplo.],
) <tab-f05-ejemplo>

Hay 2 TP, 1 FP y 2 FN, así que $P = 2 / 3 = "0,667"$, $R = 2 / 4 = "0,5"$ y
$ F_(0.5) = ("1,25" times "0,667" times "0,5") / ("0,25" times "0,667" + "0,5") = "0,417" / "0,667" = "0,625". $
La F1, que pesa igual precisión y recall, daría 0,571: F0.5 premia más que el modelo haya acertado en casi todo lo que tocó. _give_ por _leave_ cuenta dos veces en contra: como cambio que no está en la referencia (FP) y como error de la referencia sin corregir (FN). En el corpus completo, ERRANT suma los TP, FP y FN de todas las oraciones y calcula P, R y F0.5 una sola vez sobre los totales; no promedia por oración.

*¿Por qué no BLEU ni ROUGE?* Son las métricas habituales de la generación de lenguaje: BLEU @papineni2002bleu en traducción y ROUGE @lin2004rouge en resumen. Las dos cuentan cuántos n-gramas de la salida coinciden con la referencia. En corrección gramatical eso no sirve, porque la salida es casi igual a la entrada. Lo medimos en FCE test con tres sistemas (@tab-bleu):

#figure(
  table(
    columns: (1.6fr, auto, auto, auto),
    align: (left, right, right, right),
    table.header[Sistema][BLEU][ROUGE-L][F0.5],
    [Copiar la entrada, sin corregir nada], [74,6], [90,2], [0],
    [Meter errores: _the_ → _a_ en 1.224 oraciones], [68,3], [87,4], [0,002],
    [*BART-large (el modelo elegido)*], [*81,2*], [*92,3*], [*0,527*],
  ),
  caption: [BLEU, ROUGE-L y F0.5 de tres sistemas en FCE test (2.695 oraciones). BLEU de `sacrebleu` a nivel de corpus; ROUGE-L, F promedio por oración.],
) <tab-bleu>

Un sistema que no corrige nada saca 74,6 de BLEU y 90,2 de ROUGE-L, cifras que parecen excelentes, y uno que daña el texto apenas pierde seis puntos. F0.5 los pone donde corresponde: 0 a los dos, y 0,527 al modelo. Hay tres razones de fondo:

+ *BLEU y ROUGE premian copiar.* Como el 86 % de los tokens se copia, cualquier salida parecida a la entrada coincide en casi todo con la referencia. Las diferencias entre sistemas quedan comprimidas en pocos puntos (74,6 contra 81,2).
+ *No distinguen corregir de más de corregir de menos.* Una corrección falsa y un error sin corregir les cuestan lo mismo: unos n-gramas menos. F0.5 los separa en FP y FN y, con $beta = 0.5$, castiga el doble lo que más daño le hace a nuestro usuario: enseñarle una regla falsa.
+ *No dicen qué se corrigió.* F0.5 de ERRANT se puede desglosar por tipo de error (sección 6.3), y la aplicación usa ese desglose para decidir la confianza de cada explicación (capítulo 7).

Además, F0.5 de ERRANT es la métrica oficial de BEA-2019 @bryant2019bea, el origen de nuestros datos, así que el resultado se puede comparar con la literatura. GLEU @napoles2015gleu, una variante de BLEU pensada para corrección, se usa en otros corpus como JFLEG, pero hereda el problema de puntuar la oración completa y no da el desglose por tipo.
