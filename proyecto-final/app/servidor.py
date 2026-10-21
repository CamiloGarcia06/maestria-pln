"""App web del corrector: una página y una API sobre explicar.corregir_y_explicar.

Carga BART-large una sola vez al arrancar. Uso (lo hace el servicio `app` de infra/compose.yaml):
    uvicorn servidor:app --app-dir app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
import explicar  # noqa: E402

MODELO = Path(os.environ.get("MODELO", RAIZ / "modelos" / "experimentos" / "large-3e-5" / "mejor"))
LARGO_MAXIMO = 2000  # caracteres por consulta: unas 20 oraciones, ~2-3 s en la RTX 4060 Ti


class Consulta(BaseModel):
    texto: str = Field(min_length=1, max_length=LARGO_MAXIMO)


def crear_app(corrector=None) -> FastAPI:
    """corrector=None carga BART-large al arrancar; los tests pasan uno falso."""
    estado = {"corrector": corrector}
    candado = threading.Lock()  # una generación a la vez en la GPU

    @asynccontextmanager
    async def ciclo(_):
        if estado["corrector"] is None:
            from corregir import Corrector
            estado["corrector"] = Corrector(MODELO)
        explicar.oraciones("Warm up.")  # carga spaCy y ERRANT antes de la primera consulta
        yield

    app = FastAPI(title="Corrector gramatical explicado", lifespan=ciclo)

    @app.get("/")
    def pagina():
        return FileResponse(Path(__file__).parent / "estatico" / "index.html")

    @app.get("/api/salud")
    def salud():
        c = estado["corrector"]
        return {"modelo": MODELO.parent.name if corrector is None else "prueba",
                "dispositivo": getattr(c, "dispositivo", "?")}

    @app.post("/api/corregir")
    def corregir(consulta: Consulta):
        inicio = time.perf_counter()
        with candado:
            resultado = explicar.corregir_y_explicar(estado["corrector"], consulta.texto)
        return {
            "oraciones": [{**r, "ediciones": [asdict(e) for e in r["ediciones"]]} for r in resultado],
            "milisegundos": round(1000 * (time.perf_counter() - inicio)),
        }

    return app


app = crear_app()
