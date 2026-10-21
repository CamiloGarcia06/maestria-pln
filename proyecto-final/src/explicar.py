"""Convierte cada corrección del modelo en una explicación en español (tarea complementaria: POS tagging).

ERRANT alinea la oración original con la corregida, extrae las ediciones y las tipa con las etiquetas
POS y la morfología de spaCy (R:VERB:FORM, M:PRON…). El tipo elige la plantilla y las etiquetas de las
palabras completan el detalle: qué forma verbal se pedía, de qué categoría a cuál cambió la palabra.
Los tipos sin plantilla propia reciben una explicación genérica según la operación (falta, sobra, cambia).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import errant

POS = {
    "NOUN": "sustantivo", "PROPN": "nombre propio", "VERB": "verbo", "AUX": "verbo auxiliar", "ADJ": "adjetivo",
    "ADV": "adverbio", "PRON": "pronombre", "DET": "determinante", "ADP": "preposición", "CCONJ": "conjunción",
    "SCONJ": "conjunción", "PART": "partícula", "NUM": "número", "PUNCT": "signo de puntuación",
}
FORMAS = {
    "VB": "la forma base", "VBG": "la forma en -ing", "VBN": "el participio", "VBD": "el pasado",
    "VBZ": "el presente con -s", "VBP": "el presente sin -s", "MD": "un verbo modal",
}
SUJETOS = {"i", "you", "he", "she", "it", "we", "they", "there"}

# Confianza por categoría según el F0.5 de BART-large en FCE test (notebook 05, sección 3)
ALTA = {"SPELL", "VERB:SVA", "VERB:FORM", "NOUN:NUM", "MORPH", "ORTH", "DET"}
BAJA = {"NOUN", "ADJ", "ADV", "OTHER", "VERB"}


@dataclass
class Edicion:
    tipo: str            # tipo de ERRANT, p. ej. R:VERB:FORM
    original: str        # texto que se quita ("" si falta algo)
    corregido: str       # texto que se pone ("" si sobra algo)
    inicio: int          # posición en tokens de la oración original
    fin: int
    explicacion: str
    confianza: str       # alta · media · baja (sugerencia)


@cache
def _anotador():
    return errant.load("en")


def confianza(tipo: str) -> str:
    categoria = tipo[2:]
    return "alta" if categoria in ALTA else "baja" if categoria in BAJA else "media"


def _verbo(span):
    """El último verbo del fragmento (en «have gone» interesa «gone»), o el último token."""
    verbos = [t for t in span if t.tag_ in FORMAS]
    return verbos[-1] if verbos else span[-1]


def _anterior(span):
    return span.doc[span.start - 1] if span.start > 0 else None


# ---------------------------------------------------------------- plantillas
# Cada plantilla recibe la edición de ERRANT (o_str, c_str y los tokens de spaCy o_toks, c_toks).

def _falta_puntuacion(e):
    signo = e.c_toks[0].text
    # ERRANT junta el punto con la mayúscula siguiente: "my" → ". My"
    if signo in ".!?" and e.o_str:
        return f"Falta «{signo}»: aquí termina una oración y la siguiente empieza con mayúscula («{e.c_toks[-1].text}»)."
    return f"Falta «{e.c_str}»." + (" En inglés la coma separa las expresiones introductorias y las oraciones unidas"
                                     " por conjunción («As you know, …»)." if e.c_str == "," else "")


def _cambia_puntuacion(e):
    # ERRANT junta la coma con la mayúscula siguiente: ", it" → ". It"
    if e.o_str.startswith(",") and e.c_str.startswith("."):
        return "En inglés dos oraciones completas no se unen con coma, como se hace a menudo en español: se separan con punto."
    return f"Cambia «{e.o_str}» por «{e.c_str}»."


def _falta_determinante(e):
    # ERRANT puede juntar el artículo con la mayúscula vecina: "Computer" → "The computer"
    articulo = next((t.text for t in e.c_toks if t.pos_ == "DET"), e.c_str)
    return (f"Falta el artículo «{articulo}». En inglés los sustantivos contables en singular necesitan artículo"
            " (a, an, the) o un determinante.")


def _sobra_determinante(e):
    if e.o_str.lower() in ("a", "an"):
        return (f"Sobra «{e.o_str}». El artículo indefinido solo va con sustantivos contables en singular:"
                " no con plurales ni con incontables como «information», «advice» o «furniture».")
    return (f"Sobra «{e.o_str}». En inglés no se pone artículo a los sustantivos usados en sentido general"
            " («la gente» → «people», «la música» → «music»), como sí se hace en español.")


def _cambia_determinante(e):
    if {e.o_str.lower(), e.c_str.lower()} == {"a", "an"}:
        return "Se usa «an» antes de sonido vocálico y «a» antes de sonido consonante."
    return f"El determinante que corresponde es «{e.c_str}», no «{e.o_str}»."


def _cambia_preposicion(e):
    return (f"La preposición correcta es «{e.c_str}», no «{e.o_str}». Las preposiciones no se traducen palabra por"
            " palabra del español: conviene aprenderlas junto con la palabra que las rige.")


def _ortografia(e):
    return f"Ortografía: «{e.o_str}» se escribe «{e.c_str}»."


def _mayusculas(e):
    if e.o_str.lower() == e.c_str.lower():
        if e.c_str[:1].isupper() and e.o_toks[0].is_sent_start:
            return f"«{e.c_str}» va con mayúscula porque empieza la oración."
        if e.c_str[:1].isupper():
            return (f"«{e.c_str}» va con mayúscula. En inglés la llevan también los días, meses, idiomas,"
                    " nacionalidades y el pronombre «I».")
        return f"«{e.c_str}» va con minúscula."
    return f"Espacios: «{e.o_str}» se escribe «{e.c_str}»."


def _tiempo_verbal(e):
    if not e.o_str:
        return f"Falta «{e.c_str}» para formar el tiempo verbal."
    if not e.c_str:
        return f"Sobra «{e.o_str}»: el tiempo verbal no lo necesita."
    o, c = _verbo(e.o_toks), _verbo(e.c_toks)
    detalle = ""
    if o.tag_ in FORMAS and c.tag_ in FORMAS and o.tag_ != c.tag_:
        detalle = f" (de {FORMAS[o.tag_]} a {FORMAS[c.tag_]})".replace("de el ", "del ").replace(" a el ", " al ")
    return f"Tiempo verbal: «{e.o_str}» → «{e.c_str}»{detalle}. Revisa cuándo ocurre la acción."


def _forma_verbal(e):
    c = _verbo(e.c_toks)
    forma = FORMAS.get(c.tag_, "otra forma")
    texto = f"Aquí va {forma} del verbo: «{e.c_str}» en lugar de «{e.o_str}»."
    anterior = _anterior(e.c_toks)
    if c.tag_ == "VBG" and anterior is not None and anterior.pos_ == "ADP":
        texto += f" Después de una preposición («{anterior.text}») el verbo va en -ing."
    elif c.tag_ == "VB" and anterior is not None and (anterior.lower_ == "to" or anterior.tag_ == "MD"):
        texto += f" Después de «{anterior.text}» va la forma base, sin -s, -ing ni -ed."
    return texto


def _concordancia(e):
    c = _verbo(e.c_toks)
    if c.tag_ == "VBZ":
        regla = "con sujeto en tercera persona del singular (he, she, it, un nombre singular) el presente lleva -s"
    elif c.tag_ == "VBP":
        regla = "con sujeto plural o con I, you, we, they el presente no lleva -s"
    else:
        regla = "el verbo concuerda en número con su sujeto"
    return f"Concordancia: «{e.o_str}» → «{e.c_str}»; {regla}."


def _numero(e):
    numero = "plural" if _verbo(e.c_toks).tag_ in ("NNS", "NNPS") else "singular"
    return f"Va en {numero}: «{e.c_str}» en lugar de «{e.o_str}»."


def _morfologia(e):
    o, c = POS.get(e.o_toks[-1].pos_), POS.get(e.c_toks[-1].pos_)
    # Una palabra común escrita con mayúscula sale como nombre propio: ahí la categoría no es fiable
    if o and c and o != c and "nombre propio" not in (o, c):
        return f"Misma raíz, otra categoría: «{e.o_str}» es {o} y aquí hace falta {'un' if c != 'preposición' else 'una'} {c}, «{e.c_str}»."
    return f"Otra forma de la misma palabra: «{e.c_str}» en lugar de «{e.o_str}»."


def _orden(e):
    return f"Orden de palabras: «{e.o_str}» → «{e.c_str}». El orden en inglés es más fijo que en español."


def _falta_pronombre(e):
    # ERRANT puede juntar el pronombre con la palabra vecina: "Is" → "It is"
    pronombre = next((t.text for t in e.c_toks if t.pos_ == "PRON"), e.c_str)
    if pronombre.lower() in SUJETOS:
        return (f"Falta el sujeto «{pronombre}». En inglés el sujeto no se omite aunque el verbo ya lo indique"
                " («es grande» → «it is big»).")
    return f"Falta el pronombre «{pronombre}»."


def _palabra(e):
    categoria = POS.get(e.c_toks[-1].pos_, "palabra") if e.c_toks else "palabra"
    return f"Sugerencia de vocabulario: «{e.c_str}» ({categoria}) expresa mejor la idea que «{e.o_str}»."


def _reformulacion(e):
    return f"Sugerencia de redacción: «{e.o_str}» → «{e.c_str}»."


PLANTILLAS = {
    "M:PUNCT": _falta_puntuacion, "R:PUNCT": _cambia_puntuacion,
    "M:DET": _falta_determinante, "U:DET": _sobra_determinante, "R:DET": _cambia_determinante,
    "R:PREP": _cambia_preposicion,
    "M:PREP": lambda e: f"Falta la preposición «{e.c_str}».",
    "U:PREP": lambda e: f"Sobra la preposición «{e.o_str}».",
    "R:SPELL": _ortografia, "R:ORTH": _mayusculas,
    "R:VERB:TENSE": _tiempo_verbal, "M:VERB:TENSE": _tiempo_verbal, "U:VERB:TENSE": _tiempo_verbal,
    "R:VERB:FORM": _forma_verbal, "R:VERB:SVA": _concordancia, "R:NOUN:NUM": _numero, "R:MORPH": _morfologia,
    "R:WO": _orden, "M:PRON": _falta_pronombre,
    "R:PRON": lambda e: f"El pronombre que corresponde es «{e.c_str}», no «{e.o_str}».",
    "M:VERB": lambda e: f"Falta el verbo «{e.c_str}».",
    "R:VERB": _palabra, "R:NOUN": _palabra, "R:ADJ": _palabra, "R:ADV": _palabra,
    "R:OTHER": _reformulacion,
    "M:OTHER": lambda e: f"Falta «{e.c_str}».",
    "U:OTHER": lambda e: f"Sobra «{e.o_str}».",
}

GENERICAS = {
    "M": lambda e: f"Falta «{e.c_str}».",
    "U": lambda e: f"Sobra «{e.o_str}».",
    "R": lambda e: f"Cambia «{e.o_str}» por «{e.c_str}».",
}


def explicar(original: str, corregida: str) -> list[Edicion]:
    """Ediciones de ERRANT entre las dos oraciones, cada una con su explicación en español."""
    anotador = _anotador()
    o, c = anotador.parse(original, tokenise=True), anotador.parse(corregida, tokenise=True)
    ediciones = []
    for e in anotador.annotate(o, c):
        plantilla = PLANTILLAS.get(e.type) or GENERICAS[e.type[0]]
        ediciones.append(Edicion(e.type, e.o_str, e.c_str, e.o_start, e.o_end, plantilla(e), confianza(e.type)))
    return ediciones


def oraciones(texto: str) -> list[str]:
    """Divide un párrafo en oraciones con el mismo spaCy que usa ERRANT."""
    return [s.text.strip() for s in _anotador().nlp(texto).sents if s.text.strip()]


def corregir_y_explicar(corrector, texto: str) -> list[dict]:
    """El pipeline completo de la app: párrafo → oraciones → BART → ERRANT → explicaciones."""
    originales = oraciones(texto)
    corregidas = corrector.corregir(originales)[0] if originales else []
    return [{"original": o, "corregida": c, "ediciones": explicar(o, c)} for o, c in zip(originales, corregidas)]
