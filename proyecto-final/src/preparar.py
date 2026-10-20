"""Prepara los conjuntos de entrenamiento y evaluación de BART-base a partir de los corpus.

Aplica las decisiones del análisis (resultados/decisiones.csv y decisiones_tareas.csv):

- train: FCE train + W&I train (A, B, C), sin pares con ediciones UNK y sin pares duplicados;
- dev_seleccion: W&I dev + FCE dev sin hispanohablantes, para elegir el checkpoint;
- evaluación aparte: hispanohablantes de FCE dev y de FCE test, y FCE test completo,
  cada uno con su M2 oficial para errant_compare;
- BART ve texto natural (destokenizado); LOCNESS (nativos) queda fuera.

Uso: python src/preparar.py [--crudos datos/crudos] [--salida datos/procesados/entrenamiento]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import corpus
import texto


def bloques_m2(raiz: str | Path) -> list[str]:
    """Bloques S/A de los M2 oficiales, en el mismo orden (e id) que las oraciones de corpus.cargar."""
    bloques = []
    for *_, ruta_m2, _ in corpus.ARCHIVOS:
        bloques += (Path(raiz) / ruta_m2).read_text(encoding="utf-8").strip().split("\n\n")
    return bloques


def marcar(oraciones: pd.DataFrame, ediciones: pd.DataFrame) -> pd.DataFrame:
    """Agrega si la oración tiene ediciones UNK y su texto natural (original y corregida)."""
    o = oraciones.copy()
    o["tiene_unk"] = o.id.isin(ediciones.loc[ediciones.tipo == "UNK", "id_oracion"])
    o["original_texto"] = o.original.str.split().map(texto.destokenizar)
    o["corregida_texto"] = o.corregida.str.split().map(texto.destokenizar)
    return o


def pares_train(o: pd.DataFrame) -> pd.DataFrame:
    """Pares para afinar BART: sin UNK (su destino conserva un error) y sin duplicados exactos."""
    train = o[(o.split == "train") & o.corpus.isin(["fce", "wi"]) & ~o.tiene_unk]
    return train.drop_duplicates(["original_texto", "corregida_texto"])


def conjuntos_evaluacion(o: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Los hispanohablantes de FCE dev quedan fuera de la selección para poder evaluarlos aparte."""
    fce, es = o.corpus.eq("fce"), o.l1.eq("es")
    return {
        "dev_seleccion": o[(o.corpus.eq("wi") & o.split.eq("dev")) | (fce & o.split.eq("dev") & ~es)],
        "fce_dev_es": o[fce & o.split.eq("dev") & es],
        "fce_test": o[fce & o.split.eq("test")],
        "fce_test_es": o[fce & o.split.eq("test") & es],
    }


def _jsonl(df: pd.DataFrame, ruta: Path) -> None:
    filas = df[["id", "original_texto", "corregida_texto"]].rename(
        columns={"original_texto": "original", "corregida_texto": "corregida"})
    ruta.write_text("".join(json.dumps(f, ensure_ascii=False) + "\n" for f in filas.to_dict("records")),
                    encoding="utf-8")


def escribir(o: pd.DataFrame, bloques: list[str], salida: Path) -> dict:
    """train.jsonl y, por cada conjunto de evaluación, entrada.jsonl + orig.txt + ref.m2."""
    salida.mkdir(parents=True, exist_ok=True)
    train = pares_train(o)
    _jsonl(train, salida / "train.jsonl")
    resumen = {"train": {"pares": len(train),
                         "excluidos_unk": int((o.split.eq("train") & o.corpus.isin(["fce", "wi"]) & o.tiene_unk).sum()),
                         "duplicados_quitados": int((o.split.eq("train") & o.corpus.isin(["fce", "wi"]) & ~o.tiene_unk).sum()
                                                    - len(train))}}
    for nombre, df in conjuntos_evaluacion(o).items():
        carpeta = salida / nombre
        carpeta.mkdir(exist_ok=True)
        _jsonl(df, carpeta / "entrada.jsonl")
        # orig.txt va en los tokens del M2: errant_parallel compara la salida de BART contra ellos
        (carpeta / "orig.txt").write_text("\n".join(df.original) + "\n", encoding="utf-8")
        (carpeta / "ref.m2").write_text("\n\n".join(bloques[i] for i in df.id) + "\n\n", encoding="utf-8")
        resumen[nombre] = {"oraciones": len(df), "textos": int(df.texto.nunique())}
    (salida / "resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return resumen


def main() -> None:
    raiz = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--crudos", type=Path, default=raiz / "datos" / "crudos")
    parser.add_argument("--salida", type=Path, default=raiz / "datos" / "procesados" / "entrenamiento")
    args = parser.parse_args()

    oraciones, ediciones = corpus.cargar(args.crudos)
    bloques = bloques_m2(args.crudos)
    if len(bloques) != len(oraciones) or any(b.split("\n")[0][2:] != s for b, s in zip(bloques, oraciones.original)):
        raise ValueError("Los bloques M2 no coinciden con las oraciones leídas")
    resumen = escribir(marcar(oraciones, ediciones), bloques, args.salida)
    print(json.dumps(resumen, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
