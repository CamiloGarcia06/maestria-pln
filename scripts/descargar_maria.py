"""Descarga «María» (Jorge Isaacs, 1867, dominio público) desde es.wikisource y la guarda en datos/maria.txt.

Uso: python scripts/descargar_maria.py
"""
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://es.wikisource.org/w/api.php"
OBRA = "María (Isaacs)"
SALIDA = Path(__file__).resolve().parent.parent / "datos" / "maria.txt"
ROMANOS = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


def consultar(**parametros):
    url = API + "?" + urllib.parse.urlencode({**parametros, "format": "json"})
    peticion = urllib.request.Request(url, headers={"User-Agent": "maestria-pln-taller6/1.0"})
    for intento in range(6):   # wikisource responde 429 si se piden muchas páginas seguidas
        try:
            with urllib.request.urlopen(peticion) as respuesta:
                time.sleep(1)
                return json.load(respuesta)
        except urllib.error.HTTPError as error:
            if error.code != 429:
                raise
            time.sleep(5 * 2 ** intento)
    raise RuntimeError(f"wikisource sigue limitando las peticiones: {url}")


def romano_a_entero(romano):
    valores = [ROMANOS[c] for c in romano]
    return sum(-v if v < siguiente else v for v, siguiente in zip(valores, valores[1:] + [0]))


def texto_plano(titulo):
    paginas = consultar(action="query", prop="extracts", explaintext=1, titles=titulo)["query"]["pages"]
    texto = next(iter(paginas.values()))["extract"]
    texto = re.sub(r"^=+.*?=+\s*$", "", texto, flags=re.M)   # encabezados vacíos de wikisource
    return re.sub(r"\n{3,}", "\n\n", texto).strip()


def main():
    titulos = [p["title"] for p in consultar(action="query", list="allpages", apprefix=OBRA, aplimit=500)["query"]["allpages"]]
    capitulos = sorted((t for t in titulos if re.fullmatch(rf"{re.escape(OBRA)}/[IVXLC]+", t)),
                       key=lambda t: romano_a_entero(t.rsplit("/", 1)[1]))
    partes = [f"Capítulo {t.rsplit('/', 1)[1]}\n\n{texto_plano(t)}" for t in capitulos]
    SALIDA.write_text("\n\n\n".join(partes) + "\n", encoding="utf-8")
    print(f"{len(capitulos)} capítulos, {SALIDA.stat().st_size:,} bytes -> {SALIDA}")


if __name__ == "__main__":
    main()
