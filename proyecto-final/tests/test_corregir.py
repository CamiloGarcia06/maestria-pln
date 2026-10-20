from corregir import Corrector

ARREGLOS = {"He go home yesterday": "He went home yesterday", "I has a cat": "I have a cat"}


def falso(llamadas):
    """Un Corrector sin modelo: cada pasada aplica un arreglo y anota qué textos recibió."""
    corrector = Corrector.__new__(Corrector)

    def una_pasada(textos):
        llamadas.append(list(textos))
        return [ARREGLOS.get(t, t) for t in textos]

    corrector._una_pasada = una_pasada
    return corrector


def test_una_pasada_por_defecto():
    assert falso([]).corregir(["I has a cat"]) == [["I have a cat"]]


def test_solo_regenera_lo_que_sigue_cambiando():
    llamadas = []
    historia = falso(llamadas).corregir(["He go home yesterday", "It is fine"], pasadas=3)
    assert historia[-1] == ["He went home yesterday", "It is fine"]
    assert len(historia) == 3
    # 1.ª pasada: ambos; 2.ª: solo el que cambió; 3.ª: nada, porque ya no cambió
    assert llamadas == [["He go home yesterday", "It is fine"], ["He went home yesterday"]]
