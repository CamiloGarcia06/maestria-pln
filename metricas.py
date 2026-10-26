"""BLEU, ROUGE y F1 implementados desde cero para evaluar traducciones.

Todas las funciones reciben listas de tokens (strings) ya normalizados: la hipótesis es lo que produce el
modelo y la referencia es la traducción humana. En el notebook se comparan contra sacrebleu y rouge-score
para comprobar que dan lo mismo.
"""
import math
from collections import Counter


def ngramas(tokens, n):
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def bleu_corpus(hipotesis, referencias, n_max=4):
    """BLEU de corpus (Papineni et al., 2002) con una referencia por oración.

    Precisión de n-gramas recortada (un n-grama de la hipótesis cuenta como acierto a lo sumo tantas veces
    como aparece en la referencia), sumada sobre todo el corpus, media geométrica de n = 1..4 y penalización
    por brevedad si las hipótesis son en total más cortas que las referencias. Devuelve un valor en [0, 100].
    """
    aciertos = [0] * n_max
    totales = [0] * n_max
    largo_hip = largo_ref = 0
    for hip, ref in zip(hipotesis, referencias):
        largo_hip += len(hip)
        largo_ref += len(ref)
        for n in range(1, n_max + 1):
            conteo_hip, conteo_ref = ngramas(hip, n), ngramas(ref, n)
            aciertos[n - 1] += sum(min(c, conteo_ref[g]) for g, c in conteo_hip.items())
            totales[n - 1] += max(len(hip) - n + 1, 0)
    if min(aciertos) == 0:
        return 0.0
    log_precision = sum(math.log(a / t) for a, t in zip(aciertos, totales)) / n_max
    brevedad = 1.0 if largo_hip > largo_ref else math.exp(1 - largo_ref / max(largo_hip, 1))
    return 100 * brevedad * math.exp(log_precision)


def _prf(aciertos, n_hip, n_ref):
    if aciertos == 0:
        return 0.0, 0.0, 0.0
    p, r = aciertos / n_hip, aciertos / n_ref
    return p, r, 2 * p * r / (p + r)


def rouge_n(hip, ref, n):
    """ROUGE-N (Lin, 2004) de una oración: traslape de n-gramas. Devuelve (precisión, recall, F1)."""
    conteo_hip, conteo_ref = ngramas(hip, n), ngramas(ref, n)
    aciertos = sum((conteo_hip & conteo_ref).values())
    return _prf(aciertos, sum(conteo_hip.values()), sum(conteo_ref.values()))


def _largo_lcs(a, b):
    # Programación dinámica clásica, guardando solo la fila anterior.
    anterior = [0] * (len(b) + 1)
    for x in a:
        actual = [0]
        for j, y in enumerate(b):
            actual.append(anterior[j] + 1 if x == y else max(anterior[j + 1], actual[j]))
        anterior = actual
    return anterior[-1]


def rouge_l(hip, ref):
    """ROUGE-L de una oración: subsecuencia común más larga (respeta el orden). Devuelve (P, R, F1)."""
    return _prf(_largo_lcs(hip, ref), len(hip), len(ref))


def f1_tokens(hip, ref):
    """F1 de tokens (bolsa de palabras, como en SQuAD): cuántas palabras de la referencia acierta la hipótesis
    sin importar el orden. Devuelve (precisión, recall, F1)."""
    aciertos = sum((Counter(hip) & Counter(ref)).values())
    return _prf(aciertos, len(hip), len(ref))


def evaluar(hipotesis, referencias):
    """Todas las métricas sobre un corpus. ROUGE y F1 son el promedio por oración de la medida F; BLEU es de
    corpus. Todo en escala 0-100."""
    por_oracion = {"rouge1": [], "rouge2": [], "rougeL": [], "f1": []}
    for hip, ref in zip(hipotesis, referencias):
        por_oracion["rouge1"].append(rouge_n(hip, ref, 1)[2])
        por_oracion["rouge2"].append(rouge_n(hip, ref, 2)[2])
        por_oracion["rougeL"].append(rouge_l(hip, ref)[2])
        por_oracion["f1"].append(f1_tokens(hip, ref)[2])
    resultado = {"bleu": bleu_corpus(hipotesis, referencias)}
    resultado.update({k: 100 * sum(v) / len(v) for k, v in por_oracion.items()})
    return resultado
