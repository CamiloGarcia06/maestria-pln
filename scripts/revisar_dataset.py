"""Revisión del dataset MEDIQA-CORR 2024 (MS, entrenamiento).

Calcula las cifras que usa el README para decidir cada etapa del pipeline y las
guarda en resultados/revision_dataset.json.

    uv run --with pandas --with transformers scripts/revisar_dataset.py
"""

import json
import re
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

RAIZ = Path(__file__).resolve().parent.parent
CSV = RAIZ / "datos" / "MEDIQA-CORR-2024-MS-TrainingData.csv"
SALIDA = RAIZ / "resultados" / "revision_dataset.json"

TOKENIZADORES = {
    "detector": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "corrector": "GanjinZero/biobart-v2-large",
    "flan_t5_referencia": "google/flan-t5-base",
}


def parsear_oraciones(columna: str) -> dict[int, str]:
    """'0 texto\\n1 texto' → {0: 'texto', 1: 'texto'}; descarta líneas sin id."""
    oraciones = {}
    for linea in columna.split("\n"):
        m = re.match(r"^(\d+) (.*)$", linea)
        if m:
            oraciones[int(m.group(1))] = m.group(2).strip()
    return oraciones


def main() -> None:
    df = pd.read_csv(CSV)
    df["oraciones"] = df["Sentences"].apply(parsear_oraciones)
    df["n_oraciones"] = df["oraciones"].str.len()
    df["caso"] = df["oraciones"].str[0]  # las dos versiones de un caso comparten la primera oración
    errores = df[df["Error Flag"] == 1]

    r: dict = {}
    r["notas"] = len(df)
    r["con_error"] = int(df["Error Flag"].sum())
    r["exactitud_clase_mayoritaria"] = round(df["Error Flag"].mean(), 3)
    r["oraciones_por_nota"] = df["n_oraciones"].describe()[["mean", "50%", "max"]].round(1).to_dict()
    r["oraciones_total"] = int(df["n_oraciones"].sum())

    ultima = errores["Error Sentence ID"] == errores["n_oraciones"] - 1
    r["error_en_ultima_oracion"] = round(ultima.mean(), 3)
    r["error_en_primera_mitad"] = round((errores["Error Sentence ID"] < errores["n_oraciones"] / 2).mean(), 3)

    tam = df.groupby("caso").agg(n=("Error Flag", "size"), errores=("Error Flag", "sum"))
    r["casos"] = len(tam)
    r["casos_par_correcta_y_error"] = int(((tam["n"] == 2) & (tam["errores"] == 1)).sum())
    r["notas_en_un_par"] = round(df["caso"].map(tam["n"]).gt(1).mean(), 3)

    cortas = df["oraciones"].apply(lambda o: sum(len(t.split()) <= 3 for t in o.values()))
    r["notas_con_fragmentos_de_3_palabras_o_menos"] = int(cortas.gt(0).sum())
    r["notas_con_lineas_sin_id"] = int(
        df["Sentences"].apply(lambda s: any(not re.match(r"^\d+ ", x) for x in s.split("\n"))).sum()
    )

    def palabras_cambiadas(a: str, b: str) -> int:
        from difflib import SequenceMatcher

        sa, sb = a.split(), b.split()
        ops = SequenceMatcher(None, sa, sb).get_opcodes()
        return sum(max(i2 - i1, j2 - j1) for t, i1, i2, j1, j2 in ops if t != "equal")

    cambios = errores.apply(lambda f: palabras_cambiadas(f["Error Sentence"], f["Corrected Sentence"]), axis=1)
    r["palabras_cambiadas_por_correccion"] = cambios.describe()[["mean", "50%", "max"]].round(1).to_dict()

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
