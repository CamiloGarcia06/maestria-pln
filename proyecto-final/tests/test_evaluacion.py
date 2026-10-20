import pytest

from evaluacion import Conjunto, f05, leer_conjunto, tokenizar

REF = [
    "S He go home .\nA 1 2|||R:VERB:SVA|||goes|||REQUIRED|||-NONE-|||0",
    "S It is fine .\nA -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0",
]
CONJUNTO = Conjunto(
    entrada=[{"id": 0, "original": "He go home.", "corregida": "He goes home."},
             {"id": 1, "original": "It is fine.", "corregida": "It is fine."}],
    orig=["He go home .", "It is fine ."],
    ref=REF,
)


def test_tokenizar_como_los_m2():
    assert tokenizar(["I don't know, it's fine. "]) == ["I do n't know , it 's fine ."]


def test_copiar_da_cero_y_la_referencia_da_uno():
    assert f05(CONJUNTO, tokenizar(["He go home.", "It is fine."]))["f05"] == 0
    assert f05(CONJUNTO, tokenizar(["He goes home.", "It is fine."]))["f05"] == 1


def test_un_cambio_innecesario_es_falso_positivo():
    r = f05(CONJUNTO, tokenizar(["He goes home.", "It was fine."]))
    assert (r["tp"], r["fp"], r["fn"]) == (1, 1, 0)


def test_hipotesis_de_largo_distinto_falla():
    with pytest.raises(ValueError):
        f05(CONJUNTO, ["He goes home ."])


def test_leer_conjunto_y_recortar(tmp_path):
    (tmp_path / "entrada.jsonl").write_text(
        '{"id": 0, "original": "He go home.", "corregida": "He goes home."}\n'
        '{"id": 1, "original": "It is fine.", "corregida": "It is fine."}\n')
    (tmp_path / "orig.txt").write_text("He go home .\nIt is fine .\n")
    (tmp_path / "ref.m2").write_text("\n\n".join(REF) + "\n\n")
    conjunto = leer_conjunto(tmp_path)
    assert conjunto.ref == REF and len(conjunto) == 2
    assert conjunto.primeros(1).orig == ["He go home ."]
