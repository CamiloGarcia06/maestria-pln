"""F0.5 de ERRANT para la salida de BART (texto natural) contra la anotación oficial.

Mismo camino que el techo del notebook 02 (P5): la salida se tokeniza con spaCy en_core_web_sm,
errant_parallel extrae sus ediciones frente a orig.txt y errant_compare las cruza con ref.m2.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import spacy


@dataclass
class Conjunto:
    """Un conjunto de evaluación de preparar.py, alineado línea a línea."""
    entrada: list[dict]   # {"id", "original", "corregida"} en texto natural
    orig: list[str]       # oraciones originales en tokens del M2
    ref: list[str]        # bloques M2 oficiales

    def __len__(self) -> int:
        return len(self.orig)

    def primeros(self, n: int) -> Conjunto:
        return Conjunto(self.entrada[:n], self.orig[:n], self.ref[:n])


def leer_conjunto(carpeta: str | Path) -> Conjunto:
    carpeta = Path(carpeta)
    entrada = [json.loads(l) for l in (carpeta / "entrada.jsonl").read_text(encoding="utf-8").splitlines()]
    orig = (carpeta / "orig.txt").read_text(encoding="utf-8").splitlines()
    ref = (carpeta / "ref.m2").read_text(encoding="utf-8").strip().split("\n\n")
    if not len(entrada) == len(orig) == len(ref):
        raise ValueError(f"{carpeta}: entrada, orig y ref tienen largos distintos")
    return Conjunto(entrada, orig, ref)


@cache
def _nlp():
    return spacy.load("en_core_web_sm")


def tokenizar(textos: list[str]) -> list[str]:
    """Texto natural → tokens de spaCy separados por espacios, como orig.txt."""
    tokenizador = _nlp().tokenizer
    return [" ".join(t.text for t in tokenizador(s.strip())) for s in textos]


def f05(conjunto: Conjunto, hipotesis: list[str]) -> dict[str, float]:
    """F0.5 de ERRANT de hipótesis ya tokenizadas (una por oración del conjunto)."""
    if len(hipotesis) != len(conjunto):
        raise ValueError(f"{len(hipotesis)} hipótesis para {len(conjunto)} oraciones")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "orig.txt").write_text("\n".join(conjunto.orig) + "\n", encoding="utf-8")
        (tmp / "hyp.txt").write_text("\n".join(hipotesis) + "\n", encoding="utf-8")
        (tmp / "ref.m2").write_text("\n\n".join(conjunto.ref) + "\n\n", encoding="utf-8")
        subprocess.run(["errant_parallel", "-orig", tmp / "orig.txt", "-cor", tmp / "hyp.txt", "-out", tmp / "hyp.m2"],
                       check=True, capture_output=True)
        salida = subprocess.run(["errant_compare", "-hyp", tmp / "hyp.m2", "-ref", tmp / "ref.m2"],
                                check=True, capture_output=True, text=True).stdout.split()
    i = salida.index("F0.5") + 1
    tp, fp, fn, p, r, f = salida[i:i + 6]
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "precision": float(p), "recall": float(r), "f05": float(f)}
