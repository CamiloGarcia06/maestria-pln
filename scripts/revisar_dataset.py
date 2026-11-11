"""Revisión del dataset MEDIQA-CORR 2024 (MS, entrenamiento).

Calcula las cifras que usa el README para decidir cada etapa del pipeline y las
guarda en resultados/revision_dataset.json.

    uv run --with pandas --with transformers scripts/revisar_dataset.py
"""

import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
from rouge_score import rouge_scorer
from transformers import AutoTokenizer

RAIZ = Path(__file__).resolve().parent.parent
CSV = RAIZ / "datos" / "MEDIQA-CORR-2024-MS-TrainingData.csv"
SALIDA = RAIZ / "resultados" / "revision_dataset.json"

TOKENIZADORES = {
    "detector": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "corrector": "GanjinZero/biobart-v2-large",
    "flan_t5_referencia": "google/flan-t5-base",
}


SIMILITUD_MISMO_CASO = 0.7  # la distribución tiene un hueco entre 0,4 y 0,8


def parsear_oraciones(columna: str) -> dict[int, str]:
    """'0 texto\\n1 texto' → {0: 'texto', 1: 'texto'}.

    Las líneas sin id (valores de laboratorio como 'Platelet count 100,000') se
    pegan a la oración anterior: son parte de ella y pueden traer el error.
    """
    oraciones: dict[int, str] = {}
    ultimo = None
    for linea in columna.split("\n"):
        m = re.match(r"^(\d+) (.*)$", linea)
        if m:
            ultimo = int(m.group(1))
            oraciones[ultimo] = m.group(2).strip()
        elif linea.strip() and ultimo is not None:
            oraciones[ultimo] += "\n" + linea.strip()
    return oraciones


def identificar_casos(textos: list[str], oraciones: list[dict[int, str]]) -> list[int]:
    """Une las notas del mismo caso: comparten al menos una oración y su texto
    se parece en ≥ SIMILITUD_MISMO_CASO. La primera oración sola no sirve:
    junta casos distintos que abren igual ('... routine health maintenance
    examination.') y separa los pares cuyo error está en la oración 0."""
    padre = list(range(len(textos)))

    def raiz(i: int) -> int:
        while padre[i] != i:
            padre[i] = padre[padre[i]]
            i = padre[i]
        return i

    indice = defaultdict(set)
    for i, o in enumerate(oraciones):
        for oracion in o.values():
            indice[oracion].add(i)
    candidatos = set()
    for notas in indice.values():
        if len(notas) < 300:  # oraciones genéricas ('Vital signs are within normal limits.')
            notas = sorted(notas)
            candidatos |= {(a, b) for k, a in enumerate(notas) for b in notas[k + 1 :]}
    palabras = [t.split() for t in textos]
    for a, b in candidatos:
        if SequenceMatcher(None, palabras[a], palabras[b], autojunk=False).ratio() >= SIMILITUD_MISMO_CASO:
            padre[raiz(a)] = raiz(b)
    return [raiz(i) for i in range(len(textos))]


def main() -> None:
    df = pd.read_csv(CSV)
    df["oraciones"] = df["Sentences"].apply(parsear_oraciones)
    df["n_oraciones"] = df["oraciones"].str.len()
    df["caso"] = identificar_casos(df["Text"].tolist(), df["oraciones"].tolist())
    errores = df[df["Error Flag"] == 1]

    r: dict = {}
    r["notas"] = len(df)
    r["con_error"] = int(df["Error Flag"].sum())
    r["exactitud_clase_mayoritaria"] = round(df["Error Flag"].mean(), 3)
    r["oraciones_por_nota"] = df["n_oraciones"].describe()[["mean", "50%", "max"]].round(1).to_dict()
    r["oraciones_total"] = int(df["n_oraciones"].sum())
    r["negativas_por_positiva_a_nivel_de_oracion"] = round((r["oraciones_total"] - r["con_error"]) / r["con_error"], 1)

    ultima = errores["Error Sentence ID"] == errores["n_oraciones"] - 1
    r["error_en_ultima_oracion"] = round(ultima.mean(), 3)
    r["error_en_primera_mitad"] = round((errores["Error Sentence ID"] < errores["n_oraciones"] / 2).mean(), 3)
    r["error_en_oracion_0"] = int((errores["Error Sentence ID"] == 0).sum())

    tam = df.groupby("caso").agg(n=("Error Flag", "size"), errores=("Error Flag", "sum"))
    r["casos"] = len(tam)
    r["casos_par_correcta_y_error"] = int(((tam["n"] == 2) & (tam["errores"] == 1)).sum())
    r["casos_solo_version_con_error"] = int(((tam["n"] == 1) & (tam["errores"] == 1)).sum())
    r["casos_de_mas_de_2_notas"] = int((tam["n"] > 2).sum())
    r["notas_en_un_par"] = round(df["caso"].map(tam["n"]).gt(1).mean(), 3)
    primera = df["oraciones"].str[0]
    r["clave_primera_oracion"] = {
        "casos": int(primera.nunique()),
        "grupos_de_mas_de_2_notas": int(primera.value_counts().gt(2).sum()),
        "pares_separados": int(df.groupby("caso")[["oraciones"]].apply(lambda g: g["oraciones"].str[0].nunique() > 1).sum()),
    }

    cortas = df["oraciones"].apply(lambda o: sum(len(t.split()) <= 3 for t in o.values()))
    r["notas_con_fragmentos_de_3_palabras_o_menos"] = int(cortas.gt(0).sum())
    sin_id = df["Sentences"].apply(lambda s: [x for x in s.split("\n") if x.strip() and not re.match(r"^\d+ ", x)])
    r["notas_con_lineas_sin_id"] = int(sin_id.str.len().gt(0).sum())
    r["lineas_sin_id_con_texto"] = int(sin_id.str.len().sum())

    def palabras_cambiadas(a: str, b: str) -> int:
        sa, sb = a.split(), b.split()
        ops = SequenceMatcher(None, sa, sb).get_opcodes()
        return sum(max(i2 - i1, j2 - j1) for t, i1, i2, j1, j2 in ops if t != "equal")

    cambios = errores.apply(lambda f: palabras_cambiadas(f["Error Sentence"], f["Corrected Sentence"]), axis=1)
    r["palabras_cambiadas_por_correccion"] = cambios.describe()[["mean", "50%", "max"]].round(1).to_dict()

    # El control de calidad de WangLab descarta la corrección si ROUGE-L(original, corregida) < 0,7.
    # Aplicado a las correcciones de referencia dice cuántas correctas rechazaría.
    rouge = rouge_scorer.RougeScorer(["rougeL"])
    rl = errores.apply(lambda f: rouge.score(f["Error Sentence"], f["Corrected Sentence"])["rougeL"].fmeasure, axis=1)
    r["rougeL_original_vs_referencia"] = {
        "mediana": round(rl.median(), 3),
        "p10": round(rl.quantile(0.1), 3),
        "rechazadas_con_0.7": int((rl < 0.7).sum()),
        "fraccion_rechazada_con_0.7": round((rl < 0.7).mean(), 3),
    }

    r["tokens"] = {}
    for rol, nombre in TOKENIZADORES.items():
        tok = AutoTokenizer.from_pretrained(nombre)
        notas = df["Text"].apply(lambda t: len(tok(t)["input_ids"]))
        pares = errores.apply(lambda f: len(tok(f["Error Sentence"], f["Text"])["input_ids"]), axis=1)
        r["tokens"][rol] = {
            "modelo": nombre,
            "vocabulario": tok.vocab_size,
            "tokens_por_nota": notas.describe()[["mean", "50%", "max"]].round(0).to_dict(),
            "tokens_par_oracion_nota_max": int(pares.max()),
            "ejemplo": tok.tokenize(" Haemophilus influenzae metoprolol"),
        }

    SALIDA.parent.mkdir(exist_ok=True)
    SALIDA.write_text(json.dumps(r, ensure_ascii=False, indent=2))
    print(json.dumps(r, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
