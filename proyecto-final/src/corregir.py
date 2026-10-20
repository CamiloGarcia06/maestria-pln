"""Corrige oraciones con un modelo afinado por entrenar.py (lo usan el notebook 05 y la app).

La corrección puede ser iterativa: la salida vuelve a entrar al modelo hasta `pasadas` veces,
o hasta que deja de cambiar. Así se recogen los errores que una sola pasada deja sin tocar.

Uso: python src/corregir.py modelos/bart-base-gec/mejor --dev [--pasadas 3]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import evaluacion

RAIZ = Path(__file__).resolve().parents[1]


class Corrector:
    def __init__(self, ruta: str | Path, dispositivo: str | None = None, lote: int = 64):
        self.dispositivo = dispositivo or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizador = AutoTokenizer.from_pretrained(ruta)
        self.modelo = AutoModelForSeq2SeqLM.from_pretrained(ruta).to(self.dispositivo).eval()
        if self.dispositivo == "cuda":
            self.modelo = self.modelo.to(torch.bfloat16)
        self.lote = lote

    @torch.inference_mode()
    def _una_pasada(self, textos: list[str]) -> list[str]:
        salida = []
        largo = self.modelo.generation_config.max_length
        for i in range(0, len(textos), self.lote):
            x = self.tokenizador(textos[i:i + self.lote], max_length=largo, truncation=True,
                                 padding=True, return_tensors="pt").to(self.dispositivo)
            ids = self.modelo.generate(**x)
            salida += [t.strip() for t in self.tokenizador.batch_decode(ids, skip_special_tokens=True)]
        return salida

    def corregir(self, textos: list[str], pasadas: int = 1) -> list[list[str]]:
        """Devuelve, por pasada, la corrección de cada texto; solo regenera los que aún cambian."""
        actuales, activos, historia = list(textos), range(len(textos)), []
        for _ in range(pasadas):
            nuevos = self._una_pasada([actuales[i] for i in activos]) if activos else []
            siguiente = list(actuales)
            for i, t in zip(activos, nuevos):
                siguiente[i] = t
            activos = [i for i in activos if siguiente[i] != actuales[i]]
            historia.append(siguiente)
            actuales = siguiente
        return historia


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("modelo", type=Path)
    parser.add_argument("--conjunto", default="dev_seleccion")
    parser.add_argument("--pasadas", type=int, default=3)
    parser.add_argument("--datos", type=Path, default=RAIZ / "datos" / "procesados" / "entrenamiento")
    args = parser.parse_args()
    conjunto = evaluacion.leer_conjunto(args.datos / args.conjunto)
    corrector = Corrector(args.modelo)
    historia = corrector.corregir([f["original"] for f in conjunto.entrada], args.pasadas)
    for n, textos in enumerate(historia, 1):
        r = evaluacion.f05(conjunto, evaluacion.tokenizar(textos))
        print(json.dumps({"conjunto": args.conjunto, "pasadas": n, **r}))


if __name__ == "__main__":
    main()
