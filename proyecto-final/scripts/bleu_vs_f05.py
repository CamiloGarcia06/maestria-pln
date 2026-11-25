"""Tabla «¿Por qué no BLEU ni ROUGE?» del documento (sección 2.3).

Paso 1, en el contenedor de Jupyter (GPU): genera las salidas de BART-large en FCE test,
un sistema que mete errores (the → a) y calcula su F0.5 con ERRANT.
    python scripts/bleu_vs_f05.py generar
Paso 2, donde haya sacrebleu y rouge-score (no están en la imagen):
    uv run --no-project --with sacrebleu --with rouge-score python scripts/bleu_vs_f05.py medir
Resultado en resultados/bleu_vs_f05.csv.
"""
import csv
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
PREDS = RAIZ / "resultados" / "fce_test_salidas.json"
SISTEMAS = {"orig": "copiar", "mal": "meter errores (the→a)", "hyp": "BART-large"}


def generar() -> None:
    sys.path.insert(0, str(RAIZ / "src"))
    import evaluacion
    from corregir import Corrector
    c = evaluacion.leer_conjunto(RAIZ / "datos/procesados/entrenamiento/fce_test")
    orig = [f["original"] for f in c.entrada]
    d = {"orig": orig, "ref": [f["corregida"] for f in c.entrada],
         "hyp": Corrector(RAIZ / "modelos/experimentos/large-3e-5/mejor").corregir(orig)[0],
         "mal": [o.replace(" the ", " a ", 1) for o in orig]}
    d["f05"] = {k: evaluacion.f05(c, evaluacion.tokenizar(d[k]))["f05"] for k in SISTEMAS}
    PREDS.write_text(json.dumps(d, ensure_ascii=False))


def medir() -> None:
    import sacrebleu
    from rouge_score import rouge_scorer
    d = json.loads(PREDS.read_text())
    sc = rouge_scorer.RougeScorer(["rougeL"])
    filas = []
    for k, nombre in SISTEMAS.items():
        bleu = sacrebleu.corpus_bleu(d[k], [d["ref"]]).score
        rouge = 100 * sum(sc.score(r, h)["rougeL"].fmeasure for r, h in zip(d["ref"], d[k])) / len(d[k])
        filas.append({"sistema": nombre, "bleu": round(bleu, 1), "rougeL": round(rouge, 1),
                      "f05": round(d["f05"][k], 3)})
    with open(RAIZ / "resultados" / "bleu_vs_f05.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=filas[0])
        w.writeheader()
        w.writerows(filas)
    for fila in filas:
        print(fila)


if __name__ == "__main__":
    {"generar": generar, "medir": medir}[sys.argv[1]]()
