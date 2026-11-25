= Aprendizaje del modelo

== Selección del modelo: BART

Ashmore et al. piden que el modelo sea _reutilizable_ y que su desempeño se justifique. Elegimos BART @lewis2020bart, con los pesos de Hugging Face Transformers @wolf2020transformers, porque ya sabe inglés y porque su preentrenamiento consiste en reconstruir texto dañado, igual que corregir. Probamos los dos tamaños públicos (@tab-arq).

#figure(
  table(
    columns: (1.4fr, auto, auto),
    align: (left, right, right),
    table.header[Arquitectura][`facebook/bart-base`][`facebook/bart-large`],
    [Parámetros], [139 M], [406 M],
    [Capas del codificador / decodificador], [6 / 6], [12 / 12],
    [Dimensión de los vectores ($d$)], [768], [1.024],
    [Cabezas de atención], [12], [16],
    [Vocabulario BPE], [50.265 piezas], [50.265 piezas],
  ),
  caption: [Las dos variantes de BART. El vocabulario es el mismo en las dos.],
) <tab-arq>

*Vocabulario y longitud de secuencia son cosas distintas.* El vocabulario (50.265) es el número de piezas BPE @sennrich2016bpe distintas que el modelo conoce. Fija el tamaño de la matriz de embeddings ($50.265 times d$) y de la capa de salida, que da una probabilidad por pieza. La longitud de secuencia (máximo 128) es cuántas piezas tiene *una* oración. No cambia ningún parámetro: solo decide cuánto se rellena o se recorta. En la oración de ejemplo, la entrada tiene 30 piezas y el destino 31, por la coma que se agrega:

```
input_ids (30)  <s> As Ġyou Ġknow ĠI Ġam Ġin Ġcharge Ġfor Ġorganise Ġthe … </s>
                [0, 1620, 47, 216, 38, 524, 11, 1427, 13, 23750, 5, …, 2]
labels (31)     <s> As Ġyou Ġknow , ĠI Ġam Ġin Ġcharge Ġof Ġorganising Ġthe … </s>
                [0, 1620, 47, 216, 6, 38, 524, 11, 1427, 9, 22821, 5, …, 2]
```

`Ġ` marca el espacio que precede a la pieza. Las únicas piezas que cambian son las corregidas (`,` = 6, `Ġof` = 9, `Ġorganising` = 22821, `Ġleave` = 989); el resto se copia.

*Forma de los tensores.* Con un lote de $B = 32$ pares y la oración más larga del lote con $L$ piezas ($L <= 128$):
`input_ids` $(B, L)$ → embeddings + posición $(B, L, d)$ → codificador (6 capas de autoatención y red densa) $(B, L, d)$. El decodificador recibe las piezas del destino corridas una posición, $(B, M)$, atiende a sí mismo con máscara causal y al codificador, y produce $(B, M, d)$. La capa de salida proyecta eso a $(B, M, 50.265)$: una probabilidad para cada pieza del vocabulario en cada posición.

== Entrenamiento

Un lote real de dos pares muestra cómo se rellena (@tab-lote). El `DataCollatorForSeq2Seq` completa `input_ids` con `<pad>` (id 1) y marca el relleno con 0 en `attention_mask`. En `labels` el relleno es −100, el valor que la pérdida ignora. `decoder_input_ids` es `labels` corrido una posición a la derecha: para predecir la pieza $t$, el decodificador ve las piezas correctas hasta $t-1$ (_teacher forcing_).

#figure(
  table(
    columns: (auto, 1fr),
    table.header[Tensor][Primer par: _She don't know._ → _She doesn't know._],
    [`input_ids`], [`[0, 2515, 218, 75, 216, 4, 2, 1, 1, 1]`],
    [`attention_mask`], [`[1, 1, 1, 1, 1, 1, 1, 0, 0, 0]`],
    [`labels`], [`[0, 2515, 630, 75, 216, 4, 2, -100, -100, -100]`],
    [`decoder_input_ids`], [`[2, 0, 2515, 630, 75, 216, 4, 2, 1, 1]`],
  ),
  caption: [Un lote por dentro. El segundo par del lote (_I am in charge for organise it._) es más largo y fija $L = 10$.],
) <tab-lote>

La pérdida es la entropía cruzada entre la distribución que predice el decodificador y la pieza correcta de `labels`, promediada sobre las piezas distintas de −100. Lleva _label smoothing_ de 0,1 @szegedy2016inception, que no exige probabilidad 1 a la pieza correcta. AdamW @loshchilov2019adamw ajusta todos los pesos. `src/entrenar.py` arma el entrenamiento en cinco pasos que el notebook 04 recorre uno por uno: `cargar_modelo` → `preparar_datos` → `argumentos` → `crear_trainer` → `ejecutar`. La tokenización de los pares es una línea:

```python
def tokenizar(lote):
    return tokenizador(lote["original"], text_target=lote["corregida"],
                       max_length=128, truncation=True)
```

Al cargar el modelo hicimos dos ajustes que no son obvios:

```python
modelo = AutoModelForSeq2SeqLM.from_pretrained(config.modelo, dtype=torch.float32)
modelo.generation_config.no_repeat_ngram_size = 0   # BART-base trae 3
```

El primero: transformers 5 carga BART-large en el float16 en que está guardado, y AdamW lo vuelve `NaN` en el primer paso. Por eso forzamos float32, y bf16 se usa solo para los cálculos. El segundo: BART-base prohíbe repetir trigramas al generar, algo útil para resumir pero dañino para un corrector, que *tiene que copiar*. Sin el ajuste, el modelo sin afinar convierte _I think that it is a good idea and I think that it is cheap._ en _… and I think that #underline[the] it is cheap._: inventa un error para no repetir.

#figure(
  table(
    columns: (auto, auto, 1fr),
    table.header[Hiperparámetro][Valor][De dónde sale y qué cambia si se mueve],
    [Tasa de aprendizaje], [3e-5], [habitual para afinar BART; probamos 1e-5 y 5e-5 (@tab-experimentos)],
    [Calentamiento], [5 % (879 pasos)], [sube la tasa desde 0 para no romper los pesos preentrenados],
    [Weight decay], [0,01], [penaliza pesos grandes (AdamW)],
    [Lote], [32 pares (large: 16 × 2)], [prueba de humo: 234 pares/s con 4,4 GB; 64 rozó la memoria de 16 GB],
    [Precisión], [bf16 mixta], [la mitad de memoria por activación; pesos maestros en float32],
    [Label smoothing], [0,1], [generaliza mejor; por eso la pérdida no baja de \~1,7],
    [Largo máximo], [128 piezas], [entrega 1; cubre el 99,95 % de los pares],
    [Haces (_beam search_)], [4], [explorar 4 hipótesis al generar; se guarda con el modelo],
    [Épocas / paciencia], [10 / 3], [parada temprana si F0.5 en dev no mejora en 3 épocas],
    [Selección], [F0.5 en `dev_seleccion`], [el checkpoint se elige generando, no por la pérdida],
  ),
  caption: [Hiperparámetros de entrenamiento (`entrenar.Config`). Una época de BART-base son 1.758 pasos.],
) <tab-hiper>

== Selección del checkpoint y experimentos

Al final de cada época, el `Trainer` genera las 5.054 oraciones de `dev_seleccion` y calcula su F0.5 con el mismo camino de la evaluación final (spaCy → `errant_parallel` → `errant_compare`). BART-base sin afinar (época 0) solo copia: F0.5 = 0,007. Tras una época llega a 0,454 y tras dos a 0,503. Desde ahí el modelo empieza a reescribir más: el recall sube (0,308 → 0,359) pero la precisión cae más rápido (0,597 → 0,535), y la parada temprana corta en la época 5. El entrenamiento tardó 30 minutos en una RTX 4060 Ti de 16 GB.

#figure(
  image("../../figuras/15_f05_por_epoca.png", width: 66%),
  caption: [F0.5, precisión y recall de BART-base en `dev_seleccion` por época. La línea punteada es el techo de la evaluación (0,855).],
) <fig-epocas>

Para mejorar probamos primero lo barato, otras tasas de aprendizaje, y después un modelo más grande (@tab-experimentos). La tasa no mueve a BART-base de 0,50: con 1e-5 llega más lento al mismo punto y con 5e-5 se pasa antes. El tamaño sí lo mueve: BART-large conserva la precisión y encuentra más errores (recall 0,383 contra 0,308). También probamos corregir en varias pasadas, volviendo a pasar la salida por el modelo, y no ayudó (0,5025 → 0,5017).

#figure(
  table(
    columns: (1.4fr, auto, auto, auto, auto),
    align: (left, center, right, right, right),
    table.header[Corrida][Mejor época][Precisión][Recall][F0.5 dev],
    [BART-base · 3e-5], [2], [0,597], [0,308], [0,503],
    [BART-base · 1e-5], [7], [0,587], [0,324], [0,505],
    [BART-base · 5e-5], [2], [0,586], [0,317], [0,501],
    [*BART-large · 3e-5 (elegido)*], [*3*], [*0,588*], [*0,383*], [*0,531*],
  ),
  caption: [Experimentos, comparados solo en `dev_seleccion`. El test no se usó para elegir.],
) <tab-experimentos>

*Qué aprendió.* Para hacer el modelo más _interpretable_, comparamos la probabilidad que le da cada modelo a las piezas de la corrección de la oración de ejemplo, que el modelo nunca vio. BART-base sin afinar le da 0,002 a `Ġof` y 0,008 a `Ġorganising`; el modelo afinado, 0,909 y 0,956. La coma tras _As you know_ sube de 0,000 a 0,332, pero eso no basta para que el _beam search_ la elija. El modelo es *conservador*: solo cambia algo cuando está bastante seguro, que es justo lo que premia F0.5.
