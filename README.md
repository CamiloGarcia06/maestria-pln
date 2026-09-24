# Taller 3: Bayes ingenuo, LSA y LDiA sobre textos de salud mental

**Procesamiento de Lenguaje Natural · Maestría · Camilo García**

> **Pregunta del taller:** ¿qué tan bien distingue cada modelo una publicación «Normal» de una con señales de
> depresión, y qué nos enseña cada uno sobre los textos?

## Contenido

1. [Objetivo](#1-objetivo)
2. [Los datos](#2-los-datos)
3. [Metodología común](#3-metodología-común)
4. [Modelo 1: Bayes ingenuo](#4-modelo-1-bayes-ingenuo)
5. [Hallazgo: los emojis rotos delatan la clase](#5-hallazgo-los-emojis-rotos-delatan-la-clase)
6. [Modelo 2: LSA](#6-modelo-2-lsa-latent-semantic-analysis)
7. [Modelo 3: LDiA](#7-modelo-3-ldia-latent-dirichlet-allocation)
8. [Comparación de resultados](#8-comparación-de-resultados)
9. [Conclusiones](#9-conclusiones)
10. [Limitaciones y siguientes pasos](#10-limitaciones-y-siguientes-pasos)
11. [Estructura del repositorio y cómo ejecutar](#11-estructura-del-repositorio-y-cómo-ejecutar)

---

## 1. Objetivo

Ajustar **tres tipos de modelos** sobre el mismo conjunto de datos de salud mental y compararlos:

| # | Modelo | Idea central |
|---|---|---|
| 1 | **Bayes ingenuo** | Probabilidad de cada palabra en cada clase |
| 2 | **LSA** | Comprimir la matriz TF-IDF en unos pocos «tópicos» con álgebra lineal (SVD) |
| 3 | **LDiA** | Modelar cada texto como una mezcla probabilística de tópicos |

> **Nomenclatura** (del libro *Natural Language Processing in Action*, cap. 4):
> **LDA** = *Linear Discriminant Analysis*, un **clasificador**.
> **LDiA** = *Latent Dirichlet Allocation*, un **modelo de tópicos**.
> LSA y LDiA no clasifican por sí solos: producen una representación del texto y sobre ella clasificamos con LDA.

---

## 2. Los datos

Archivo: `datos_salud_mental.csv`, con **31.755 publicaciones cortas** de redes sociales traducidas al español.

| Columna | Contenido |
|---|---|
| `status` | Etiqueta: `Normal` o `Depression` |
| `statement_es_es` | Texto de la publicación |

| Clase | Textos | % | Código |
|---|---|---|---|
| Normal | 16.351 | 51% | 1 |
| Depression | 15.404 | 49% | 0 |

Ejemplos:

- *«Me siento feliz hoy»* → **Normal**
- *«Simplemente me acuesto en la cama y veo películas y duermo todo el día»* → **Depression**

**Observaciones:**

- Las clases están **balanceadas**, así que la *accuracy* es una métrica razonable. Aun así reportamos
  precisión, recall y F1.
- Los textos son **traducciones automáticas**: quedan rastros como «wkwk» (risa en indonesio) o referencias al Eid.
- Los textos de depresión suelen ser **publicaciones largas en primera persona**, y los normales, frases cortas.

---

## 3. Metodología común

```
Texto ──► Tokenización ──► Vectorización ──► Modelo ──► Métricas
          TweetTokenizer   frecuencias       Bayes        accuracy
          o spaCy          TF-IDF            LSA + LDA    precisión
                           conteos (BoW)     LDiA + LDA   recall, F1
```

Dos decisiones hacen que la comparación sea válida:

1. **Mismo split para todos:** 80% entrenamiento (25.404 textos) y 20% prueba (6.351 textos), con
   `random_state=271828`. Todos los modelos se evalúan sobre exactamente los mismos textos de prueba.
2. **Sin fuga de datos:** los vectorizadores (TF-IDF, conteos), el SVD y LDiA se ajustan **solo con
   entrenamiento** y después se aplican a prueba. El notebook de partida ajustaba TF-IDF y SVD con todos los
   datos antes de dividir; lo corregimos para que el conjunto de prueba nunca influya en la representación.

**Métricas** (clase positiva = `Normal`):

| Métrica | Pregunta que responde |
|---|---|
| Accuracy | ¿Qué fracción de todos los textos clasificó bien? |
| Precisión | De los que predijo como Normal, ¿cuántos eran Normal? |
| Recall | De los que eran Normal, ¿cuántos encontró? |
| F1 | Media armónica de precisión y recall |

---

## 4. Modelo 1: Bayes ingenuo

📓 `modelo_bayes_ingenuo.ipynb`

### Cómo funciona

Implementado **desde cero**, sin scikit-learn:

1. **Preprocesamiento** (`process_tweet`): se quitan URLs y `#`, se tokeniza con `TweetTokenizer`, se eliminan
   stopwords y puntuación en español y se aplica **stemming** Snowball (`deprimido` → `deprim`).
2. **Conteo** (`count_tweets`): cuántas veces aparece cada par `(palabra, clase)` en entrenamiento.
3. **Entrenamiento** (`train_naive_bayes`):

   - **Log prior**: qué tan probable es cada clase antes de leer el texto.

     $$\text{logprior} = \log\frac{D_{pos}}{D_{neg}}$$

   - **Probabilidad de cada palabra por clase**, con **suavizado de Laplace** (+1) para que una palabra nunca
     vista en una clase no dé probabilidad cero:

     $$P(w \mid pos) = \frac{freq_{pos}(w) + 1}{N_{pos} + V} \qquad P(w \mid neg) = \frac{freq_{neg}(w) + 1}{N_{neg} + V}$$

   - **Log-verosimilitud** de cada palabra: positiva si la palabra «empuja» hacia Normal, negativa si empuja
     hacia Depression.

     $$\lambda(w) = \log\frac{P(w \mid pos)}{P(w \mid neg)}$$

4. **Predicción**: se suman el prior y la log-verosimilitud de cada palabra del texto.

   $$\text{puntaje} = \text{logprior} + \sum_{w \in \text{texto}} \lambda(w) \qquad \text{puntaje} > 0 \Rightarrow \text{Normal}$$

Es «ingenuo» porque **asume que las palabras son independientes entre sí** dada la clase. Es falso en el
lenguaje, pero funciona sorprendentemente bien.

### Resultados

| Accuracy | Precisión (Normal) | Recall (Normal) | F1 |
|---|---|---|---|
| **0,824** | 0,962 | 0,687 | 0,801 |

**Interpretación:** cuando el modelo dice «Normal», acierta el 96% de las veces. Pero solo encuentra el 69% de
los textos normales: **casi 1 de cada 3 textos normales lo clasifica como depresión**. Es un modelo
«conservador» hacia la clase Normal.

**Raíces más asociadas a Depression** (log-verosimilitud más negativa):
`suic`, `depresion`, `entumec`, `cobard`, `miser`, `colaps`, `desesperanz`, `ideacion`. Tienen sentido clínico.

---

## 5. Hallazgo: los emojis rotos delatan la clase

Al revisar las palabras más asociadas a **Normal**, las primeras **no son palabras**:

| Token | En Normal | En Depression |
|---|---|---|
| `ðÿ` | 261 | 0 |
| `â` | 197 | 0 |
| `™` | 150 | 0 |

Son **emojis cuya codificación se rompió** (*mojibake*) en algún paso de la preparación del dataset. Como solo
aparecen en una clase, el modelo los usa como pista, pero es una **correlación espuria**: no dice nada sobre
salud mental. En un escenario real habría que limpiarlos antes de entrenar. Este hallazgo muestra una ventaja
de los modelos transparentes como Bayes: **permiten auditar qué aprendió el modelo**.

---

## 6. Modelo 2: LSA (Latent Semantic Analysis)

📓 `topic_model.ipynb`

### Cómo funciona

1. **TF-IDF** con tokenizer de spaCy: cada texto es un vector de 2.000 dimensiones. Cada palabra pesa más si es
   frecuente en el texto pero rara en el corpus.
2. **SVD truncado** (`TruncatedSVD`, 16 componentes): descompone la matriz TF-IDF y la proyecta en las
   **16 direcciones que más varianza explican**. Cada dirección es un «tópico», una combinación lineal de
   palabras con pesos positivos y negativos.
3. **LDA** (Linear Discriminant Analysis) clasifica sobre esos 16 números por texto.

Como línea base, también entrenamos **LDA directamente sobre las 2.000 columnas TF-IDF**, sin reducir.

### Ajustes respecto al notebook de partida

| Problema | Corrección |
|---|---|
| `min_df=0.1` descartaba toda palabra presente en menos del 10% de los textos, y casi no quedaba vocabulario | `min_df=5` (mínimo 5 documentos) y `max_features=2000` para que la matriz densa quepa en memoria |
| La variable `terms` no estaba definida | `terms = tfidf_model.get_feature_names_out()` |
| TF-IDF y SVD se ajustaban antes de dividir los datos | Se ajustan solo con entrenamiento |
| Tokenizar con el pipeline completo de spaCy era muy lento | Se usa solo `nlp.tokenizer`: mismos tokens, mucho más rápido |

### Resultados

| Representación | Dimensiones | Train acc. | Test acc. | Test F1 |
|---|---|---|---|---|
| TF-IDF + LDA | 2.000 | 0,949 | **0,926** | **0,930** |
| LSA + LDA | 16 | 0,896 | 0,895 | 0,902 |

**Interpretación:**

- Con **125 veces menos dimensiones**, LSA pierde solo unos 3 puntos de accuracy.
- LSA **no sobreajusta** (train ≈ test), mientras que TF-IDF sí un poco (0,949 → 0,926).
- El **tópico 0** es el que más separa las clases: su peso medio es **0,415 en Depression frente a 0,149 en
  Normal**. Sus palabras principales son `que, y, de, no, me`, es decir, captura **textos largos en primera
  persona**.
- **Desventaja:** como no se quitan stopwords, los tópicos están dominados por palabras vacías y son difíciles
  de interpretar.

---

## 7. Modelo 3: LDiA (Latent Dirichlet Allocation)

📓 `topic_model.ipynb`

### Cómo funciona

LDiA es un **modelo generativo probabilístico**. Supone que:

- cada **tópico** es una distribución de probabilidad sobre palabras, y
- cada **texto** es una mezcla de tópicos (por ejemplo, 60% «trabajo» y 40% «familia»).

A diferencia de LSA, LDiA trabaja con **conteos enteros** (bolsa de palabras), no con TF-IDF, y todos sus pesos
son **positivos**. Por eso sus tópicos se interpretan como «de qué habla el texto».

Pasos:

1. `CountVectorizer` con tokenizer de spaCy, **sin stopwords** en español y solo tokens alfabéticos
   (5.000 términos).
2. `LatentDirichletAllocation` con **16** y con **32** tópicos (`learning_method='online'` por velocidad).
3. Cada texto queda como un vector de proporciones de tópicos, y **LDA** clasifica sobre él.

### Tópicos encontrados (16 tópicos, selección)

| Palabras principales del tópico | Peso medio en Depression | Peso medio en Normal |
|---|---|---|
| quiero, vida, simplemente, siento, realmente | **0,206** | 0,068 |
| siento, cosas, odio, triste, sentir | **0,116** | 0,073 |
| trabajo, depresión, años, ayuda, terapia | **0,113** | 0,069 |
| mental, salud, ansiedad, pánico, estrés | 0,065 | 0,059 |
| madre, noche, casa, dormir, sueño | 0,027 | **0,071** |

**Interpretación:**

- Los tópicos más asociados a **depresión** hablan de la **vida, de sentirse mal, del trabajo, de pedir ayuda y
  de la terapia**.
- Los de la **vida cotidiana** (familia, noche, dormir) pesan más en los textos **normales**.
- Curiosamente, el tópico de **salud mental y ansiedad** pesa casi igual en ambas clases.

### Resultados

| Representación | Train acc. | Test acc. | Test F1 |
|---|---|---|---|
| LDiA 16 tópicos + LDA | 0,794 | 0,797 | 0,791 |
| LDiA 32 tópicos + LDA | 0,827 | **0,827** | **0,824** |

LDiA es el modelo **más interpretable**, pero el que peor clasifica: resume cada texto en unas pocas
proporciones y pierde detalle. Duplicar los tópicos (16 → 32) mejora unos 3 puntos.

---

## 8. Comparación de resultados

Todos sobre los mismos **6.351 textos de prueba**:

| Modelo | Representación | Dim. | Train acc. | Test acc. | Precisión | Recall | F1 |
|---|---|---|---|---|---|---|---|
| LDA | TF-IDF | 2.000 | 0,949 | **0,926** | 0,903 | **0,960** | **0,930** |
| LDA | LSA | 16 | 0,896 | 0,895 | 0,871 | 0,934 | 0,902 |
| LDA | LDiA | 32 | 0,827 | 0,827 | 0,871 | 0,781 | 0,824 |
| Bayes ingenuo | BoW + stemming | ~23.000 | 0,866 | 0,824 | **0,962** | 0,687 | 0,801 |
| LDA | LDiA | 16 | 0,794 | 0,797 | 0,843 | 0,746 | 0,791 |

```
Accuracy en prueba
TF-IDF + LDA   ██████████████████████████████████████████████  0,926
LSA 16 + LDA   █████████████████████████████████████████████   0,895
LDiA 32 + LDA  █████████████████████████████████████████       0,827
Bayes ingenuo  █████████████████████████████████████████       0,824
LDiA 16 + LDA  ████████████████████████████████████████        0,797
```

---

## 9. Conclusiones

| Objetivo | Modelo recomendado | Por qué |
|---|---|---|
| **Clasificar** | TF-IDF o **LSA** | TF-IDF es el más preciso (0,926). LSA logra 0,895 con solo 16 dimensiones y sin sobreajuste: el mejor equilibrio. |
| **Entender** de qué hablan los textos | **LDiA** | Sus tópicos se leen solos y muestran qué temas caracterizan cada clase. |
| **Línea base y auditoría** | **Bayes ingenuo** | Simple, rápido y transparente. Gracias a él detectamos el artefacto de los emojis. |

**Idea principal:** no hay un modelo que gane en todo. Hay un **intercambio entre rendimiento e
interpretabilidad**, y la elección depende de para qué se use el modelo.

---

## 10. Limitaciones y siguientes pasos

**Limitaciones**

- Los textos son **traducciones automáticas**, con ruido lingüístico.
- Los **emojis mal codificados** filtran la etiqueta y pueden inflar los resultados, sobre todo en Bayes.
- Se usó **una sola partición** train/test, sin validación cruzada: no sabemos cuánto varían las métricas.
- Todas las representaciones se clasificaron con un modelo **lineal** (LDA).

**Siguientes pasos propuestos**

- Limpiar la codificación de los emojis y volver a medir.
- Validación cruzada para estimar la variabilidad de las métricas.
- Explorar más valores del número de tópicos en LSA y LDiA.
- Probar otros clasificadores (regresión logística, SVM) o representaciones densas (*embeddings*).

---

## 11. Estructura del repositorio y cómo ejecutar

| Archivo | Descripción |
|---|---|
| `datos_salud_mental.csv` | Dataset (31.755 textos) |
| `modelo_bayes_ingenuo.ipynb` | Modelo 1: Bayes ingenuo desde cero. Genera `resultados_bayes.csv`. |
| `topic_model.ipynb` | Línea base TF-IDF, Modelo 2 (LSA) y Modelo 3 (LDiA), con la tabla comparativa final |
| `resultados_bayes.csv` | Métricas del Bayes ingenuo, que usa la tabla comparativa |

**Orden de ejecución:**

1. `modelo_bayes_ingenuo.ipynb` (≈ 1 minuto)
2. `topic_model.ipynb` (≈ 5 minutos; su tabla final incluye el Bayes ingenuo)

**Dependencias:** `pandas`, `numpy`, `scikit-learn`, `nltk`, `spacy` (con el modelo `es_core_news_sm`) y
`matplotlib`.

**En Google Colab:** subir `datos_salud_mental.csv` y descomentar la celda de instalación al inicio de
`topic_model.ipynb`.

**Referencia:** Lane, H. & Dyshel, M. *Natural Language Processing in Action*, 2.ª ed., cap. 4:
[notebook del capítulo](https://gitlab.com/tangibleai/nlpia2/-/blob/main/notebooks/Chapter-04_Finding-Meaning-in-Word-Counts-Semantic-Analysis.ipynb).
