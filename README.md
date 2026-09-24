# Taller 3: Bayes ingenuo, LSA y LDiA sobre datos de salud mental

Rama del Taller 3 del curso de Procesamiento de Lenguaje Natural (maestría).

## Objetivo

Ajustar tres tipos de modelos (**Bayes ingenuo**, **LSA** y **LDiA**) para clasificar textos en español
como `Normal` o `Depression`, y comparar su rendimiento e interpretabilidad.

## Datos

`datos_salud_mental.csv`: 31.755 textos cortos (tipo publicaciones de redes sociales) traducidos al español.

| Columna | Descripción |
|---|---|
| `status` | Etiqueta: `Normal` (16.351) o `Depression` (15.404) |
| `statement_es_es` | Texto |

En los notebooks: `Normal = 1`, `Depression = 0`.

## Contenido

| Archivo | Descripción |
|---|---|
| `modelo_bayes_ingenuo.ipynb` | Bayes ingenuo implementado desde cero (TweetTokenizer, stopwords, stemmer Snowball, suavizado de Laplace). Guarda sus métricas en `resultados_bayes.csv`. |
| `topic_model.ipynb` | Basado en el capítulo 4 de *NLP in Action*: línea base TF-IDF + LDA, **LSA** (TF-IDF → TruncatedSVD, 16 tópicos) y **LDiA** (bolsa de palabras → Latent Dirichlet Allocation, 16 y 32 tópicos), con tabla comparativa y conclusiones. |
| `resultados_bayes.csv` | Métricas del Bayes ingenuo para la tabla comparativa final. |

> Nomenclatura del libro: **LDA** = *Linear Discriminant Analysis* (el clasificador),
> **LDiA** = *Latent Dirichlet Allocation* (el modelo de tópicos).

Todos los modelos usan el mismo split 80/20 (`random_state=271828`). Los vectorizadores, el SVD y LDiA se
ajustan solo con los datos de entrenamiento.

## Resultados

| Modelo | Accuracy (test) | F1 (test) |
|---|---|---|
| Bayes ingenuo (BoW + stemming) | 0,824 | 0,801 |
| TF-IDF + LDA (línea base) | 0,926 | 0,930 |
| LSA, 16 tópicos + LDA | 0,895 | 0,902 |
| LDiA, 16 tópicos + LDA | 0,797 | 0,791 |
| LDiA, 32 tópicos + LDA | 0,827 | 0,824 |

- **LSA**: reduce 2000 términos a 16 tópicos y pierde solo unos 3 puntos frente a TF-IDF, sin sobreajuste.
- **LDiA**: da los tópicos más interpretables (trabajo, familia, ansiedad/salud mental, sueño...), aunque clasifica peor.
- **Bayes ingenuo**: alta precisión para `Normal` (0,96) pero bajo recall (0,69). Además reveló un artefacto del
  dataset: los emojis mal codificados (`ðÿ`, `â`, `™`) aparecen casi solo en la clase `Normal`.
- **En síntesis:** para clasificar conviene TF-IDF o LSA; para entender de qué hablan los textos, LDiA.

## Cómo ejecutar

1. Ejecutar primero `modelo_bayes_ingenuo.ipynb` (genera `resultados_bayes.csv`).
2. Luego `topic_model.ipynb` (unos 5 minutos; la tabla final incluye el Bayes ingenuo).

Dependencias: `pandas`, `numpy`, `scikit-learn`, `nltk`, `spacy` (modelo `es_core_news_sm`) y `matplotlib`.
En Colab, descomentar la celda de instalación al inicio de `topic_model.ipynb` y subir `datos_salud_mental.csv`.
