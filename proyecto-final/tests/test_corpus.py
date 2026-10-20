import pytest

from corpus import Edicion, Oracion, aplicar_ediciones, alinear_con_ensayos, leer_m2

M2 = """S This technology has help to made my life easier .
A 3 4|||R:VERB:FORM|||helped|||REQUIRED|||-NONE-|||0
A 5 6|||R:VERB:FORM|||make|||REQUIRED|||-NONE-|||0

S It is fine .
A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0
"""


def test_leer_m2_separa_oraciones_y_descarta_noop():
    oraciones = leer_m2(M2)
    assert [len(o.ediciones) for o in oraciones] == [2, 0]
    assert oraciones[0].ediciones[0] == Edicion(3, 4, "R:VERB:FORM", "helped", 0)


def test_aplicar_ediciones_reconstruye_la_corregida():
    oracion = leer_m2(M2)[0]
    assert " ".join(aplicar_ediciones(oracion.tokens, oracion.ediciones)) == \
        "This technology has helped to make my life easier ."


def test_aplicar_ediciones_con_insercion_y_borrado_ajusta_posiciones():
    tokens = "I want go to the the park".split()
    ediciones = [Edicion(2, 2, "M:VERB:FORM", "to", 0), Edicion(4, 5, "U:DET", "", 0)]
    assert aplicar_ediciones(tokens, ediciones) == "I want to go to the park".split()


def test_aplicar_ediciones_ignora_otros_anotadores():
    assert aplicar_ediciones(["a"], [Edicion(0, 1, "R:OTHER", "b", 1)]) == ["a"]


def test_alinear_liga_cada_oracion_con_su_ensayo_pese_a_espacios_y_comillas():
    ensayos = [{"text": "I like it.  It’s good.\n\nBye."}, {"text": "Second essay."}]
    oraciones = [Oracion(t.split()) for t in ["I like it .", "It 's good .", "Bye .", "Second essay ."]]
    assert alinear_con_ensayos(oraciones, ensayos) == [0, 0, 0, 1]


def test_alinear_falla_si_el_texto_no_coincide():
    with pytest.raises(ValueError):
        alinear_con_ensayos([Oracion(["Other"])], [{"text": "Text"}])


def test_posiciones_en_corregida_sigue_inserciones_y_borrados():
    from corpus import posiciones_en_corregida
    tokens = "I want go to the the park".split()
    ediciones = [Edicion(4, 5, "U:DET", "", 0), Edicion(2, 2, "M:VERB:FORM", "to", 0)]
    corregida = aplicar_ediciones(tokens, ediciones)
    assert posiciones_en_corregida(ediciones) == [(2, 3), (5, 5)]
    assert corregida[2:3] == ["to"]
