"""Paso entre los tokens de spaCy de los M2 y el texto natural que ve BART."""
from __future__ import annotations

# Partes de contracciones que spaCy separa: "do n't" -> "don't", "it 's" -> "it's"
CONTRACCIONES = {"n't", "'s", "'re", "'ve", "'ll", "'d", "'m", "'"}
PEGADOS_A_LA_IZQUIERDA = set(".,!?;:%)]}") | {"...", "..", "!!", "??", "?!", "!?"}
PEGADOS_A_LA_DERECHA = set("([{$#")


def _guion_interno(tokens: list[str], i: int) -> bool:
    """'well - known' -> 'well-known': en el texto crudo el guion entre palabras va pegado 5 de cada 6 veces."""
    return (tokens[i] == "-" and 0 < i < len(tokens) - 1
            and tokens[i - 1][-1].isalnum() and tokens[i + 1][0].isalnum())


def destokenizar(tokens: list[str]) -> str:
    """Une tokens al estilo spaCy en texto natural (reglas para inglés)."""
    texto, pegar, comilla_abierta = "", True, False
    for i, tok in enumerate(tokens):
        if _guion_interno(tokens, i):
            texto += tok
            pegar = True
            continue
        if tok == '"':
            if comilla_abierta:
                texto += tok
                pegar = False
            else:
                texto += ("" if pegar else " ") + tok
                pegar = True
            comilla_abierta = not comilla_abierta
            continue
        if tok in PEGADOS_A_LA_IZQUIERDA or tok.lower() in CONTRACCIONES:
            texto += tok
        else:
            texto += ("" if pegar else " ") + tok
        pegar = tok in PEGADOS_A_LA_DERECHA
    return texto
