# Maestría · Procesamiento de Lenguaje Natural — Grupo 6

Juan Camilo Sandoval García · Nestor Fandiño · David Nicolás Lasso

## Proyecto final: un tutor que corrige y explica

**[`proyecto-final/`](proyecto-final/)** — corrector gramatical para estudiantes hispanohablantes de inglés que, por cada corrección, explica el error en español.

| | |
|---|---|
| **Usuario final** | estudiante universitario colombiano que debe certificar inglés (B1–B2) y practica la escritura sin retroalimentación |
| **Tarea principal** | generación de lenguaje: BART-large afinado reescribe la oración con errores en su versión corregida |
| **Tarea complementaria** | POS tagging: ERRANT, sobre las etiquetas y lemas de spaCy, tipifica cada edición para elegir su explicación |
| **Datos** | FCE v2.1 y W&I v2.1 (BEA-2019): 56.254 pares de entrenamiento |
| **Resultado** | F0.5 = 0,527 en FCE test (0,515 en hispanohablantes); deja intactas el 91 % de las oraciones correctas |

- 📄 **Documento:** [`proyecto-final/documento/proyecto_final_grupo6.pdf`](proyecto-final/documento/proyecto_final_grupo6.pdf)
- 🔬 **Pipeline paso a paso:** notebooks [`01`](proyecto-final/notebooks/01_analisis_datos.ipynb) a [`06`](proyecto-final/notebooks/06_explicacion.ipynb), con el código en [`proyecto-final/src/`](proyecto-final/src/)
- 🛠️ **Requerimientos e instalación:** [`proyecto-final/README.md`](proyecto-final/README.md)
- 🌐 **App web:** [`proyecto-final/app/`](proyecto-final/app/) (FastAPI; también corre en CPU)

## Talleres del curso

Cada taller está en su propia rama: `taller-3`, `taller-5`, `taller-6` y `taller-7`.
