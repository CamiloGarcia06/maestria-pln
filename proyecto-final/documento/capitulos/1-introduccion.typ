= Introducción

En Colombia, muchos programas universitarios exigen un nivel B1 o B2 de inglés para graduarse, y las pruebas Saber Pro e IELTS evalúan la escritura. Quien se prepara para esas pruebas escribe textos de práctica, pero casi nunca recibe una revisión detallada. Un profesor particular es caro. Los correctores comerciales como Grammarly corrigen en inglés y no explican por qué algo está mal. Este proyecto construye un tutor que corrige el texto y además explica cada error en español.

== Usuario final

El usuario es *un estudiante universitario colombiano, hispanohablante, que debe certificar inglés (B1--B2) y practica la escritura sin retroalimentación*. Tres rasgos lo definen y orientan decisiones del sistema:

- *Escribe párrafos cortos de práctica*, del tipo de los exámenes (cartas, reseñas, ensayos). Por eso entrenamos con FCE y W&I, que son ensayos de examen de estudiantes de inglés, y no con texto de nativos ni de la web.
- *Comete los errores típicos del hispanohablante.* No comete más errores que otros estudiantes, sino otros. En los 398 textos de FCE escritos por hispanohablantes, 11 de 52 tipos de error aparecen con una frecuencia significativamente distinta a la del resto (prueba χ² por tipo, con corrección de Benjamini--Hochberg). Omiten el sujeto 1,74 veces más (`M:PRON`, «_Is situated…_» por «_It is situated…_»). Fallan en la puntuación 1,69 veces más, en la forma verbal 1,30 veces más, en la ortografía 1,24 veces más y en las preposiciones 1,17 veces más, porque las traducen palabra por palabra. Las explicaciones de esos tipos mencionan el contraste con el español.
- *Necesita confiar en la corrección.* Si el tutor marca como error algo que está bien, el estudiante aprende una regla falsa. Este rasgo fija la métrica: F0.5, que pesa más la precisión que el recall (capítulo 2).

*Quién no es el usuario.* No es un hablante nativo que revisa estilo, ni un traductor profesional, ni un estudiante de nivel C que ya escribe casi sin errores. Para ellos sobran las explicaciones básicas y las herramientas comerciales bastan. Tampoco es el profesor: el tutor no califica ni reemplaza su revisión; da retroalimentación inmediata entre una clase y otra.

Un rasgo queda fuera a propósito: el tutor *no* estima el nivel CEFR del estudiante. W&I trae el nivel de cada texto, pero lo usamos solo para describir los datos. Así el proyecto se centra en corregir y explicar.

== La aplicación

Una página web donde el estudiante pega un párrafo en inglés de hasta 2.000 caracteres y, en menos de un segundo, ve su texto corregido. Lo que sobra aparece tachado y lo que falta, subrayado. Junto a cada cambio aparecen el tipo de error, una explicación corta en español y un nivel de confianza: _Regla_, _Probable_ o _Sugerencia_. Al final queda el texto corregido completo para copiarlo. Un ejemplo real del test de FCE, escrito por un hispanohablante:

#block(inset: (x: 10pt, y: 8pt), fill: luma(248), stroke: 0.5pt + luma(210), radius: 2pt, width: 100%)[
  #set text(size: 10pt)
  *Entrada:* As you know I am in charge for organise the conference in my college. \
  *Salida:* As you know I am in charge #underline[of] #underline[organising] the conference in my college.
  #v(0.2em)
  1. `R:PREP` · _for → of_ --- La preposición correcta es «of», no «for». Las preposiciones no se traducen palabra por palabra del español. \
  2. `R:VERB:FORM` · _organise → organising_ --- Aquí va la forma en -ing del verbo. Después de una preposición («of») el verbo va en -ing.
]

El resto del documento describe cómo funciona por dentro. El capítulo 2 define las dos tareas de PLN y la métrica. El capítulo 3 presenta la metodología de Ashmore et al. y el pipeline completo. Los capítulos 4 a 8 recorren el pipeline en el orden de esa metodología, con las transformaciones reales que sufren los datos y el código que las hace. El capítulo 9 cierra con las limitaciones y las conclusiones.
