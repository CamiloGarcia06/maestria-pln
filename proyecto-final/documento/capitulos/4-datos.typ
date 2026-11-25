= Gestión de datos

== Recolección: los corpus de BEA-2019

Usamos los dos corpus de la tarea compartida BEA-2019 @bryant2019bea, que se descargan sin registro. Ambos son ensayos de examen de estudiantes de inglés corregidos por anotadores expertos (desiderátum _relevante_). *FCE v2.1* @yannakoudakis2011fce contiene exámenes First Certificate de 2000 y 2001, y para cada texto registra la lengua materna (L1) del autor. Así encontramos los 398 textos de hispanohablantes. *W&I v2.1* trae textos de la plataforma Write & Improve en los niveles A, B y C. *LOCNESS* (ensayos de nativos) se usa solo como referencia. Los tres tienen licencia no comercial y LOCNESS no se puede redistribuir. Por eso `datos/` no está en el repositorio y `make datos` lo descarga.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, left, right, right, right, right),
    table.header[Corpus][Split][Textos][Oraciones][Ediciones][Sin error],
    [FCE], [train], [2.116], [28.350], [44.382], [36,3 %],
    [FCE], [dev], [159], [2.191], [3.529], [36,3 %],
    [FCE], [test], [194], [2.695], [4.760], [32,2 %],
    [W&I], [train], [3.000], [34.308], [63.683], [32,8 %],
    [W&I], [dev], [300], [3.396], [6.682], [31,1 %],
    [LOCNESS], [dev], [50], [988], [950], [47,8 %],
  ),
  caption: [Tamaño de los corpus (ediciones del anotador 0).],
) <tab-corpus>

Los datos vienen en formato *M2*. La línea `S` es la oración tokenizada. Cada línea `A` es una edición con la posición de inicio y fin en tokens, el tipo de ERRANT, la corrección y el número de anotador. Este es el bloque de la oración de ejemplo:

```
S As you know I am in charge for organise the conference in my college . Do not worry
  if Richard did not left you all the information .
A 3 3|||M:PUNCT|||,|||REQUIRED|||-NONE-|||0
A 7 8|||R:PREP|||of|||REQUIRED|||-NONE-|||0
A 8 9|||R:VERB:FORM|||organising|||REQUIRED|||-NONE-|||0
A 22 23|||R:VERB:FORM|||leave|||REQUIRED|||-NONE-|||0
```

`corpus.cargar` lee los M2 y los alinea con el JSON de cada ensayo comparando el texto sin espacios, para que cada oración herede la L1 y el identificador del autor. En total son 71.928 oraciones y 123.986 ediciones.

== Preprocesamiento: de M2 a pares de texto natural

*Aplicar las ediciones.* El destino que BART debe generar es la oración con las correcciones del anotador 0. Las ediciones se aplican en orden, y un desfase lleva la cuenta de cuánto se alargó o acortó la oración para que las posiciones de las siguientes no se corran:

```python
def aplicar_ediciones(tokens, ediciones, anotador=0):
    corregida, desfase = list(tokens), 0
    propias = (e for e in ediciones if e.anotador == anotador)
    for e in sorted(propias, key=lambda e: (e.inicio, e.fin)):
        nuevos = e.correccion.split()
        corregida[e.inicio + desfase:e.fin + desfase] = nuevos
        desfase += len(nuevos) - (e.fin - e.inicio)
    return corregida
```

*Destokenizar.* Los M2 separan la puntuación y las contracciones (`do n't`, `it 's`). BART se preentrenó con texto normal, y el estudiante escribe texto normal. Por eso `texto.destokenizar` reconstruye el texto natural (@tab-destok). La ida y vuelta, de destokenizar y volver a tokenizar con spaCy, recupera exactamente la oración en el 99,34 % de los casos. Guardamos ambas versiones porque ERRANT evalúa sobre los tokens del M2.

#figure(
  table(
    columns: (1fr, 1fr),
    table.header[Tokens del M2][Texto natural (lo que ve BART)],
    [`I do n't know , it 's fine .`], [`I don't know, it's fine.`],
    [`He said " yes " ( twice ) .`], [`He said "yes" (twice).`],
    [`As you know , I am in charge of organising …`], [`As you know, I am in charge of organising …`],
  ),
  caption: [Destokenización con las reglas de `src/texto.py`.],
) <tab-destok>

== Análisis: cada hallazgo, una decisión

Los notebooks 01 y 02 analizan los datos en función de las dos tareas. La @tab-decisiones recoge las decisiones que cambiaron el pipeline.

#figure(
  table(
    columns: (1.1fr, 1.9fr, auto),
    table.header[Hallazgo][Decisión][Desiderátum],
    [34,4 % de las oraciones de train no tienen error], [se conservan: enseñan al modelo a no tocar lo que está bien], [balanceados],
    [2.779 pares (4,4 %) tienen ediciones `UNK`: error marcado pero no corregido], [se excluyen; si no, BART aprendería a dejar errores reconocidos], [precisos],
    [3.625 copias sobrantes de 631 pares (_Yours sincerely_ ×167, _Dear Kim_ ×114)], [se deduplican; si no, se sobrerrepresentan fórmulas sin error], [balanceados],
    [ERRANT 3.0.2 reproduce el 86,9 % de las ediciones anotadas y el mismo tipo en el 94,9 %], [se retipan todas las ediciones con ERRANT, la misma herramienta que usa la app], [precisos],
    [el 86 % de los tokens de una oración con errores se copia], [la métrica es F0.5 sobre ediciones, no exactitud por token], [---],
    [128 piezas BPE cubren el 99,95 % de los pares], [largo máximo 128 en entrada y salida; se recortan 15 pares], [completos],
    [el BPE de BART no produce ningún `<unk>`; una palabra mal escrita usa 2,5 piezas frente a 1,1], [el tokenizador se usa sin cambios], [---],
    [el test de FCE (exámenes de 2001) difiere de train (JSD 0,084 contra 0,036 del azar)], [se mantiene el split oficial y se espera un test más difícil], [completos],
    [hispanohablantes: 11 de 52 tipos de error con frecuencia distinta], [se evalúan aparte y se priorizan sus tipos en las plantillas], [relevantes],
  ),
  caption: [Decisiones de la gestión de datos (`resultados/decisiones.csv` y `decisiones_tareas.csv`).],
) <tab-decisiones>

#figure(
  image("../../figuras/06_errores_hispanohablantes.png", width: 66%),
  caption: [Tipos de error con frecuencia distinta en hispanohablantes frente al resto de FCE (razón de tasas por 1.000 tokens, χ² con corrección de Benjamini--Hochberg).],
) <fig-es>

== Conjuntos finales

`preparar.py` (o `make preparar`) produce los archivos de @tab-conjuntos en `datos/procesados/entrenamiento/`. El entrenamiento queda en *56.254 pares* tras pasar por el embudo 62.658 → 59.879 (sin `UNK`) → 56.254 (sin duplicados). Cada línea de `train.jsonl` es un par en texto natural:

```json
{"id": 5,
 "original": "On the other hand, the theatre restaurant was closed because unknown reasons.",
 "corregida": "In addition, the theatre restaurant was closed for unknown reasons."}
```

#figure(
  table(
    columns: (auto, 1.6fr, auto, 1fr),
    align: (left, left, right, left),
    table.header[Conjunto][Contenido][Oraciones][Uso],
    [`train.jsonl`], [FCE train + W&I train, filtrado], [56.254 pares], [afinar BART],
    [`dev_seleccion`], [W&I dev + FCE dev *sin* hispanohablantes], [5.054], [elegir el checkpoint],
    [`fce_dev_es`], [FCE dev, solo hispanohablantes (44 textos)], [533], [evaluación aparte],
    [`fce_test`], [FCE test completo (194 textos)], [2.695], [resultado final],
    [`fce_test_es`], [FCE test, solo hispanohablantes (20 textos)], [224], [evaluación aparte],
  ),
  caption: [Conjuntos de entrenamiento y evaluación.],
) <tab-conjuntos>

Los hispanohablantes de FCE dev se sacan de la selección del checkpoint. Así su resultado no queda sesgado por haber elegido el modelo mirándolos. Cada conjunto de evaluación trae tres archivos: `entrada.jsonl` (texto natural para BART), `orig.txt` (tokens del M2) y `ref.m2` (la anotación oficial). Antes de entrenar comprobamos la cadena de evaluación con dos extremos conocidos. Copiar la entrada da F0.5 = 0, y pasar la corrección de referencia por el mismo camino da 0,855 (sección 6.1).
