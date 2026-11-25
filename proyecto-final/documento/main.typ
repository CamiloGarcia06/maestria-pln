// Documento final del proyecto de PLN · Grupo 6
// Compilar desde proyecto-final/:  typst compile --root . documento/main.typ documento/proyecto_final_grupo6.pdf

#let titulo = "Un tutor que corrige y explica: corrección gramatical para estudiantes hispanohablantes de inglés con BART y ERRANT"
#let repo = "https://github.com/CamiloGarcia06/maestria-pln/tree/proyecto-final"

#set document(title: titulo, author: ("Juan Camilo Sandoval García", "Nestor Fandiño", "David Nicolás Lasso"))
#set text(lang: "es", region: "co", font: "Libertinus Serif", size: 11pt)
#set par(justify: true, leading: 0.62em, spacing: 0.9em)
#set page(paper: "a4", margin: (x: 2.3cm, y: 2.2cm))
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => { v(1.1em); text(size: 16pt, it); v(0.35em) }
#show heading.where(level: 2): it => { v(0.5em); text(size: 12.5pt, it); v(0.2em) }
#show raw: set text(font: "DejaVu Sans Mono", size: 8.3pt)
#show raw.where(block: true): it => block(breakable: false, fill: luma(246), stroke: (left: 2pt + rgb("#A4262C")), inset: (x: 9pt, y: 7pt), width: 100%, radius: 2pt, it)
#show link: set text(fill: rgb("#7a1d22"))
#set figure(gap: 0.7em)
#show figure.caption: set text(size: 9.5pt)
#show figure.where(kind: table): set figure.caption(position: top)
#set table(align: left, stroke: (x, y) => if y == 0 { (bottom: 0.7pt) } else { none }, inset: (x: 5pt, y: 3.6pt))
#show table: set text(size: 9.3pt)
#show table.cell.where(y: 0): strong

// ---------------------------------------------------------------- portada
#page(numbering: none, margin: (x: 2.6cm, y: 3cm))[
  #v(2.5cm)
  #text(size: 10pt, tracking: 0.12em, fill: rgb("#A4262C"))[PROCESAMIENTO DE LENGUAJE NATURAL · PROYECTO FINAL]
  #v(0.6cm)
  #text(size: 24pt, weight: "bold")[Un tutor que corrige y explica]
  #v(0.2cm)
  #text(size: 14pt)[Corrección gramatical para estudiantes hispanohablantes de inglés con BART y ERRANT]
  #v(2.2cm)
  #text(size: 12pt)[*Grupo 6*] \
  #v(0.15cm)
  Juan Camilo Sandoval García \
  Nestor Fandiño \
  David Nicolás Lasso
  #v(1.6cm)
  Maestría · Procesamiento de Lenguaje Natural \
  Noviembre de 2026
  #v(1fr)
  #line(length: 100%, stroke: 0.5pt + luma(180))
  #set text(size: 9.5pt)
  *Código, requerimientos e instalación:* #link(repo)[#repo] \
  Los notebooks `01` a `06` del repositorio ejecutan cada etapa que describe este documento.
]

// ---------------------------------------------------------------- resumen y contenido
#set page(numbering: "i", number-align: center)
#counter(page).update(1)

#align(center, text(size: 13pt, weight: "bold")[Resumen])
#v(0.2em)
Construimos un tutor de escritura para estudiantes universitarios colombianos que necesitan certificar inglés. El estudiante pega un párrafo y recibe el texto corregido y, por cada cambio, el tipo de error y una explicación en español. El sistema combina dos tareas de PLN vistas en clase. La *generación de lenguaje* es la principal: afinamos BART con 56.254 pares de oraciones de los corpus FCE y W&I (BEA-2019) para que reescriba una oración con errores en su versión corregida. El *POS tagging* es la complementaria: ERRANT usa las etiquetas y los lemas de spaCy para alinear la entrada con la salida y clasificar cada edición en uno de 55 tipos; ese tipo elige la plantilla de la explicación. Seguimos el ciclo de vida de Ashmore et al. (2021): gestión de datos, aprendizaje, verificación y despliegue. Cada etapa termina en decisiones medidas. BART-large alcanza F0.5 = 0,527 en el test de FCE (precisión 0,587, recall 0,373), frente a 0 si no se corrige nada y un techo de 0,855 que impone la propia evaluación. En los textos de hispanohablantes el resultado es 0,515, y el modelo deja intactas el 91 % de sus oraciones correctas. Las plantillas explican el 91,4 % de las ediciones que propone el modelo, y la aplicación web responde en unos 120 ms por oración.

#v(0.8em)
#{
  set text(size: 10pt)
  show heading: it => { v(0.3em); text(size: 13pt, it); v(0.1em) }
  outline(title: [Contenido], indent: auto, depth: 2)
}

#pagebreak()
#set page(numbering: "1")
#counter(page).update(1)

// ---------------------------------------------------------------- capítulos
#include "capitulos/1-introduccion.typ"
#include "capitulos/2-tarea.typ"
#include "capitulos/3-metodologia.typ"
#include "capitulos/4-datos.typ"
#include "capitulos/5-aprendizaje.typ"
#include "capitulos/6-verificacion.typ"
#include "capitulos/7-explicacion.typ"
#include "capitulos/8-despliegue.typ"
#include "capitulos/9-cierre.typ"

// ---------------------------------------------------------------- referencias
#pagebreak()
#bibliography("referencias.bib", title: [Referencias], style: "apa")
