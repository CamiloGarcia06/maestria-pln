from texto import destokenizar


def test_pega_puntuacion_y_contracciones():
    assert destokenizar("I do n't know , it 's fine .".split()) == "I don't know, it's fine."


def test_comillas_abren_y_cierran():
    assert destokenizar('He said " hello " to me .'.split()) == 'He said "hello" to me.'


def test_parentesis_y_simbolos_de_apertura():
    assert destokenizar("It costs ( about ) $ 5 .".split()) == "It costs (about) $5."


def test_guion_entre_palabras_va_pegado():
    assert destokenizar("a well - known far - away place".split()) == "a well-known far-away place"


def test_guion_suelto_no_se_pega():
    assert destokenizar("war - .".split()) == "war -."
