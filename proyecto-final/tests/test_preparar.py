import json

import pandas as pd

import corpus
from preparar import bloques_m2, conjuntos_evaluacion, escribir, marcar, pares_train


def _oraciones():
    filas = [
        # id, corpus, split, l1, original, corregida, texto
        (0, "fce", "train", "es", "I has a dog .", "I have a dog .", "t0"),
        (1, "fce", "train", "fr", "I has a dog .", "I have a dog .", "t1"),  # duplicado del 0
        (2, "wi", "train", None, "It 's fine .", "It 's fine .", "t2"),
        (3, "wi", "train", None, "He go home .", "He go home .", "t3"),  # solo UNK
        (4, "locness", "dev", None, "Natives write .", "Natives write .", "t4"),
        (5, "fce", "dev", "es", "She do n't know .", "She does n't know .", "t5"),
        (6, "fce", "dev", "de", "We was here .", "We were here .", "t6"),
        (7, "wi", "dev", None, "They is ok .", "They are ok .", "t7"),
        (8, "fce", "test", "es", "A apple .", "An apple .", "t8"),
        (9, "fce", "test", "ja", "Book is red .", "The book is red .", "t9"),
    ]
    return pd.DataFrame(filas, columns=["id", "corpus", "split", "l1", "original", "corregida", "texto"])


def _ediciones():
    return pd.DataFrame({"id_oracion": [0, 1, 3], "tipo": ["R:VERB:SVA", "R:VERB:SVA", "UNK"]})


def test_marcar_destokeniza_y_detecta_unk():
    o = marcar(_oraciones(), _ediciones())
    assert o.loc[o.id == 5, "original_texto"].item() == "She don't know."
    assert o.tiene_unk.tolist() == [False, False, False, True] + [False] * 6


def test_train_excluye_unk_duplicados_y_locness():
    assert pares_train(marcar(_oraciones(), _ediciones())).id.tolist() == [0, 2]


def test_hispanohablantes_de_dev_quedan_fuera_de_la_seleccion():
    conjuntos = conjuntos_evaluacion(marcar(_oraciones(), _ediciones()))
    assert conjuntos["dev_seleccion"].id.tolist() == [6, 7]
    assert conjuntos["fce_dev_es"].id.tolist() == [5]
    assert conjuntos["fce_test"].id.tolist() == [8, 9]
    assert conjuntos["fce_test_es"].id.tolist() == [8]


def test_escribir_deja_m2_de_referencia_alineado(tmp_path):
    o = marcar(_oraciones(), _ediciones())
    bloques = [f"S {s}\nA -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0" for s in o.original]
    resumen = escribir(o, bloques, tmp_path)
    assert resumen["train"] == {"pares": 2, "excluidos_unk": 1, "duplicados_quitados": 1}
    assert (tmp_path / "fce_test_es" / "ref.m2").read_text().startswith("S A apple .")
    assert (tmp_path / "fce_test_es" / "orig.txt").read_text() == "A apple .\n"
    fila = json.loads((tmp_path / "train.jsonl").read_text().splitlines()[1])
    assert fila == {"id": 2, "original": "It's fine.", "corregida": "It's fine."}


def test_bloques_m2_sigue_el_orden_de_archivos(tmp_path, monkeypatch):
    (tmp_path / "a.m2").write_text("S uno\n\nS dos\n")
    (tmp_path / "b.m2").write_text("S tres\n")
    monkeypatch.setattr(corpus, "ARCHIVOS", [("x", "train", None, "a.m2", ""), ("y", "dev", None, "b.m2", "")])
    assert bloques_m2(tmp_path) == ["S uno", "S dos", "S tres"]
