= Limitaciones y trabajo futuro

- *No comparamos con un LLM.* En la entrega 1 propusimos dos líneas base: no corregir y un LLM sin entrenar. Solo medimos la primera, más BART-large sin afinar (F0.5 = 0,014). Una comparación con un LLM por _prompting_ sobre FCE test, con la misma evaluación de ERRANT, diría si vale la pena el afinado frente a un modelo general, sobre todo en precisión.
- *El test de hispanohablantes es pequeño.* Son 224 oraciones de 20 textos, todas del examen de 2001, y su distribución de errores difiere de train (JSD 0,215 contra 0,103 del azar). Las diferencias de uno o dos puntos en ese conjunto no son concluyentes.
- *Las explicaciones no se evaluaron con usuarios.* Medimos cobertura (91,4 %), no si el estudiante entiende la explicación ni si aprende. Además, la explicación es tan buena como la corrección: si BART hace un cambio innecesario, la plantilla lo explica igual de convencida. La confianza por tipo mitiga eso, pero no lo elimina.
- *El modelo trabaja oración por oración.* No ve el párrafo, así que no corrige la concordancia de tiempos entre oraciones ni la coherencia. Por diseño, también deja pasar las reescrituras de estilo.
- *Falta la etapa de actualización del despliegue.* Ashmore et al. incluyen monitorear el sistema en uso y reentrenarlo. La app no guarda las consultas. Un siguiente paso sería registrar, con consentimiento, las correcciones que el estudiante rechaza y usarlas como señal para afinar.

El margen de mejora más directo está en los errores simples que el modelo deja pasar (_Musseum_, _an historical_). Dos caminos para atacarlo son ensamblar varios checkpoints y agregar datos sintéticos con errores de ortografía.

= Conclusiones

Construimos un tutor que corrige textos en inglés de estudiantes hispanohablantes y explica cada error en español. Combina dos tareas del curso: *generación de lenguaje*, con BART-large afinado de punta a punta sobre 56.254 pares de FCE y W&I, y *POS tagging*, con las etiquetas y los lemas de spaCy dentro de ERRANT, que tipifica cada edición para elegir su explicación.

Seguir el ciclo de vida de #cite(<ashmore2021>, form: "prose") nos obligó a que cada análisis terminara en una decisión medida. En la gestión de datos excluimos los pares con `UNK`, deduplicamos, conservamos las oraciones sin error y separamos a los hispanohablantes de la selección. En el aprendizaje elegimos por F0.5 en dev y comparamos cuatro corridas antes de quedarnos con BART-large. En la verificación medimos una sola vez en test, contra una línea base y un techo explícitos. En el despliegue mostramos la confianza de cada explicación según el desempeño medido de su tipo.

El resultado es F0.5 = 0,527 en FCE test y 0,515 en los textos de hispanohablantes, el 62 % del camino hasta el techo que permite la evaluación. El modelo es fuerte justamente en tipos donde los hispanohablantes se equivocan más, como la ortografía (0,77) y la forma verbal (0,80), y aceptable en las preposiciones (0,58). Deja intactas nueve de cada diez oraciones correctas, explica con plantilla propia el 91,4 % de sus correcciones y responde en una fracción de segundo. Es lo que necesita el usuario del capítulo 1: practicar con retroalimentación cada vez que escribe, sin esperar a que alguien revise su texto.
