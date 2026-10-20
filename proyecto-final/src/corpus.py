"""Lectura de los corpus de BEA-2019 (FCE v2.1 y W&I+LOCNESS v2.1).

Convierte los archivos M2 en dos tablas (oraciones y ediciones), reconstruye la
oración corregida y liga cada oración con su ensayo en el JSON para recuperar
los metadatos que el M2 no trae: lengua materna (FCE) y nivel CEFR (W&I).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# Misma normalización de puntuación que json_to_m2.py, que generó los M2 oficiales
NORMALIZACION = str.maketrans({
    "’": "'", "´": "'", "‘": "'", "′": "'", "`": "'",
    "“": '"', "”": '"', "˝": '"', "¨": '"', "„": '"', "『": '"', "』": '"',
    "–": "-", "—": "-", "―": "-", "¬": "-",
    "、": ",", "，": ",", "：": ":", "；": ";", "？": "?", "！": "!",
    "ِ": " ", "​": " ",
})

# (corpus, split, nivel, archivo m2, archivo json); nivel None = no viene en el corpus
ARCHIVOS = [
    *[("fce", s, None, f"fce/m2/fce.{s}.gold.bea19.m2", f"fce/json/fce.{s}.json")
      for s in ("train", "dev", "test")],
    *[("wi", s, n, f"wi+locness/m2/{n}.{s}.gold.bea19.m2", f"wi+locness/json/{n}.{s}.json")
      for s in ("train", "dev") for n in "ABC"],
    ("locness", "dev", "N", "wi+locness/m2/N.dev.gold.bea19.m2", "wi+locness/json/N.dev.json"),
]


@dataclass
class Edicion:
    inicio: int
    fin: int
    tipo: str
    correccion: str
    anotador: int


@dataclass
class Oracion:
    tokens: list[str]
    ediciones: list[Edicion] = field(default_factory=list)


def leer_m2(texto: str) -> list[Oracion]:
    """Un bloque S/A por oración; las líneas noop se descartan (oración sin error)."""
    oraciones = []
    for bloque in texto.strip().split("\n\n"):
        lineas = bloque.split("\n")
        oracion = Oracion(tokens=lineas[0][2:].split())
        for linea in lineas[1:]:
            span, tipo, correccion, _, _, anotador = linea[2:].split("|||")
            if tipo == "noop":
                continue
            inicio, fin = map(int, span.split())
            oracion.ediciones.append(Edicion(inicio, fin, tipo, correccion, int(anotador)))
        oraciones.append(oracion)
    return oraciones


def aplicar_ediciones(tokens: list[str], ediciones: list[Edicion], anotador: int = 0) -> list[str]:
    """Reconstruye la oración corregida aplicando las ediciones de un anotador."""
    corregida = list(tokens)
    desfase = 0
    for e in sorted((e for e in ediciones if e.anotador == anotador), key=lambda e: (e.inicio, e.fin)):
        nuevos = e.correccion.split()
        corregida[e.inicio + desfase:e.fin + desfase] = nuevos
        desfase += len(nuevos) - (e.fin - e.inicio)
    return corregida


def _sin_espacios(texto: str) -> str:
    return re.sub(r"\s", "", texto.translate(NORMALIZACION))


def alinear_con_ensayos(oraciones: list[Oracion], ensayos: list[dict]) -> list[int]:
    """Índice del ensayo de cada oración.

    Los M2 recorren los ensayos en el mismo orden que el JSON, así que basta con
    consumir el texto de cada ensayo (sin espacios) con los tokens de cada oración.
    """
    textos = [_sin_espacios(e["text"]) for e in ensayos]
    indices, i, pos = [], 0, 0
    for n, oracion in enumerate(oraciones):
        trozo = "".join(oracion.tokens)
        while i < len(textos) and pos == len(textos[i]):
            i, pos = i + 1, 0
        if i == len(textos) or not textos[i].startswith(trozo, pos):
            raise ValueError(f"La oración {n} no coincide con el ensayo {i} en la posición {pos}: {trozo[:60]!r}")
        indices.append(i)
        pos += len(trozo)
    if i != len(textos) - 1 or pos != len(textos[-1]):
        raise ValueError(f"Sobran ensayos sin oraciones a partir del {i}")
    return indices


def cargar(raiz: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tablas de oraciones y ediciones (anotador 0) de todos los archivos."""
    raiz = Path(raiz)
    filas_o, filas_e = [], []
    for corpus, split, nivel, ruta_m2, ruta_json in ARCHIVOS:
        oraciones = leer_m2((raiz / ruta_m2).read_text(encoding="utf-8"))
        ensayos = [json.loads(l) for l in (raiz / ruta_json).read_text(encoding="utf-8").splitlines()]
        for oracion, i in zip(oraciones, alinear_con_ensayos(oraciones, ensayos)):
            id_oracion = len(filas_o)
            ensayo = ensayos[i]
            ediciones = [e for e in oracion.ediciones if e.anotador == 0]
            filas_o.append({
                "id": id_oracion, "corpus": corpus, "split": split,
                "nivel": nivel or ensayo.get("cefr"),
                # En FCE el id es del autor (respondió dos preguntas, q); en W&I es del texto
                "texto": f'{ensayo["id"]}#{ensayo["q"]}' if "q" in ensayo else ensayo["id"],
                "autor": ensayo.get("userid", ensayo["id"]),
                "l1": ensayo.get("l1"), "edad": ensayo.get("age"),
                "original": " ".join(oracion.tokens),
                "corregida": " ".join(aplicar_ediciones(oracion.tokens, ediciones)),
                "n_tokens": len(oracion.tokens), "n_ediciones": len(ediciones),
            })
            for e in ediciones:
                operacion, _, categoria = e.tipo.partition(":")
                filas_e.append({
                    "id_oracion": id_oracion, "inicio": e.inicio, "fin": e.fin,
                    "tipo": e.tipo, "operacion": operacion, "categoria": categoria or e.tipo,
                    "original": " ".join(oracion.tokens[e.inicio:e.fin]), "correccion": e.correccion,
                })
    return pd.DataFrame(filas_o), pd.DataFrame(filas_e)
