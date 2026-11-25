= Despliegue: la aplicación web

== Integración

La aplicación (`app/servidor.py`) es un servicio FastAPI que carga BART-large una sola vez al arrancar, en unos 15 segundos, y expone dos rutas. `GET /api/salud` devuelve el modelo y el dispositivo. `POST /api/corregir` recibe hasta 2.000 caracteres y ejecuta el pipeline de uso completo, que en el código es una sola función:

```python
def corregir_y_explicar(corrector, texto):
    """Párrafo → oraciones → BART → ERRANT → explicaciones."""
    originales = oraciones(texto)                  # spaCy divide el párrafo
    corregidas = corrector.corregir(originales)[0] if originales else []
    return [{"original": o, "corregida": c, "ediciones": explicar(o, c)}
            for o, c in zip(originales, corregidas)]
```

La respuesta es un JSON con cada oración, su versión corregida y la lista de ediciones. Cada edición trae el tipo, el texto que sale y el que entra, la explicación, la confianza y su posición en caracteres. Así, la página (`app/estatico/index.html`) tacha lo que sobra y subraya lo que falta *sobre el texto que escribió el estudiante*, numera cada cambio y muestra la explicación con su etiqueta: _Regla_ (confianza alta), _Probable_ (media) o _Sugerencia_ (baja). Debajo queda el texto corregido completo con un botón para copiarlo. Un candado deja una sola generación a la vez en la GPU.

El sistema corre en Docker Compose, con una imagen única para Jupyter y la app (Python 3.12, PyTorch 2.14 con CUDA 13, transformers 5.20, ERRANT 3.0.2, spaCy 3.8), sobre una RTX 4060 Ti de 16 GB. El `README.md` del repositorio detalla los requerimientos y la instalación paso a paso: de `make env` y `make up` hasta `make entrenar` y `make evaluar`.

== Desiderata del despliegue

- *Adecuado al propósito.* La app corrige oración por oración mientras el estudiante espera, así que importa la latencia con lote de 1. BART-large tarda unos *120 ms por oración* (BART-base, 73 ms), así que un párrafo de cinco oraciones queda listo en menos de un segundo.
- *Tolerado.* El usuario debe poder confiar en lo que ve. Para eso, el modelo deja intactas el 91 % de las oraciones correctas, y las correcciones de los tipos débiles se presentan como sugerencias, no como reglas.
- *Adaptable.* El modelo es intercambiable: basta cambiar la variable `MODELO` del servicio para usar otro checkpoint, y `Corrector` es la misma clase que usa el notebook 05. Los tests de la app (`tests/test_app.py`) usan un corrector falso, así que corren sin GPU.
