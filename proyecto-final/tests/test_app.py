import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from servidor import LARGO_MAXIMO, crear_app  # noqa: E402


class Falso:
    dispositivo = "cpu"

    def corregir(self, textos):
        return [[t.replace("He go", "He goes") for t in textos]]


def cliente():
    return TestClient(crear_app(Falso()))


def test_corrige_y_explica_cada_oracion():
    with cliente() as c:
        r = c.post("/api/corregir", json={"texto": "He go home. It is fine."})
    assert r.status_code == 200
    oraciones = r.json()["oraciones"]
    assert [o["corregida"] for o in oraciones] == ["He goes home.", "It is fine."]
    e = oraciones[0]["ediciones"][0]
    assert e["tipo"] == "R:VERB:SVA" and e["confianza"] == "alta" and e["caracteres_original"] == [3, 5]
    assert oraciones[1]["ediciones"] == []


def test_rechaza_texto_vacio_o_largo():
    with cliente() as c:
        assert c.post("/api/corregir", json={"texto": ""}).status_code == 422
        assert c.post("/api/corregir", json={"texto": "a" * (LARGO_MAXIMO + 1)}).status_code == 422


def test_pagina_y_salud():
    with cliente() as c:
        assert "Corrector gramatical explicado" in c.get("/").text
        assert c.get("/api/salud").json()["dispositivo"] == "cpu"
