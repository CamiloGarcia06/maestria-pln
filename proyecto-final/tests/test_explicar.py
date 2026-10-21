from explicar import GENERICAS, PLANTILLAS, confianza, corregir_y_explicar, explicar, oraciones


def unica(original, corregida):
    ediciones = explicar(original, corregida)
    assert len(ediciones) == 1, ediciones
    return ediciones[0]


def test_concordancia_explica_la_s():
    e = unica("He go home every day.", "He goes home every day.")
    assert e.tipo == "R:VERB:SVA" and "-s" in e.explicacion and e.confianza == "alta"
    assert (e.original, e.corregido, e.inicio, e.fin) == ("go", "goes", 1, 2)


def test_forma_verbal_tras_preposicion_usa_la_etiqueta_pos():
    e = unica("I am in charge of organise the party.", "I am in charge of organising the party.")
    assert e.tipo == "R:VERB:FORM"
    assert "-ing" in e.explicacion and "«of»" in e.explicacion


def test_sujeto_omitido():
    e = unica("Is a very good film.", "It is a very good film.")
    assert e.tipo == "M:PRON" and "sujeto" in e.explicacion


def test_articulo_sobrante_en_sentido_general():
    e = unica("The people is very friendly.", "People is very friendly.")
    assert e.tipo.startswith("U:DET") and "sentido general" in e.explicacion


def test_coma_entre_oraciones():
    ediciones = explicar("It is cheap, it is near.", "It is cheap. It is near.")
    assert any("punto" in e.explicacion for e in ediciones)


def test_vocabulario_es_sugerencia():
    assert confianza("R:NOUN") == "baja" and confianza("R:PREP") == "media"


def test_tipo_sin_plantilla_usa_la_generica():
    assert "U:ADJ" not in PLANTILLAS and "U" in GENERICAS
    e = unica("It is a very big nice house.", "It is a very nice house.")
    assert e.tipo == "U:ADJ" and e.explicacion == "Sobra «big»."


def test_sin_cambios_no_hay_ediciones():
    assert explicar("It is fine.", "It is fine.") == []


def test_parrafo_completo_con_corrector_falso():
    class Falso:
        def corregir(self, textos):
            return [[t.replace("go", "goes") for t in textos]]

    salida = corregir_y_explicar(Falso(), "He go home. It is fine.")
    assert [s["original"] for s in salida] == oraciones("He go home. It is fine.") == ["He go home.", "It is fine."]
    assert salida[0]["ediciones"][0].tipo == "R:VERB:SVA" and salida[1]["ediciones"] == []


def test_ediciones_juntadas_por_errant():
    punto = unica("I like it my sister too.", "I like it. My sister too.")
    assert punto.tipo == "M:PUNCT" and punto.explicacion.startswith("Falta «.»")
    articulo = unica("Computer is useful.", "The computer is useful.")
    assert articulo.tipo == "M:DET" and "«The»" in articulo.explicacion


def test_indefinido_sobrante_y_mayuscula_inicial():
    assert "contables en singular" in unica("I need an information.", "I need information.").explicacion
    assert "empieza la oración" in unica("the city is big.", "The city is big.").explicacion


def test_tiempo_verbal_contrae_el_articulo():
    e = unica("Yesterday I go to the cinema.", "Yesterday I went to the cinema.")
    assert e.tipo == "R:VERB:TENSE" and "(del presente sin -s al pasado)" in e.explicacion


def test_posiciones_en_caracteres_para_subrayar():
    original, corregida = "Is a good film, he go home.", "It is a good film. He goes home."
    for e in explicar(original, corregida):
        desde, hasta = e.caracteres_original
        assert original[desde:hasta] == e.original
        desde, hasta = e.caracteres_corregida
        assert corregida[desde:hasta] == e.corregido
    insercion = unica("I like music very much.", "I like the music very much.")
    assert insercion.caracteres_original == (7, 7) and insercion.caracteres_corregida == (7, 10)
