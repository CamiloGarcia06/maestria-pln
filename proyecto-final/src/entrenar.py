"""Afina todos los pesos de BART-base con train.jsonl y elige el checkpoint por F0.5 en dev_seleccion.

Al final de cada época genera con beam search sobre dev_seleccion y mide F0.5 con ERRANT
(src/evaluacion.py). En la carpeta de salida quedan:

- registro.jsonl: pérdida cada 50 pasos y F0.5 por época (lo lee el notebook 04);
- predicciones/evaluacion_NN.jsonl: lo que generó el modelo en cada evaluación (00 = antes de entrenar);
- mejor/: el checkpoint con mayor F0.5, listo para AutoModelForSeq2SeqLM.from_pretrained;
- config.json: la configuración usada.

Uso: python src/entrenar.py [--epocas 10] [--muestra 2000]
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, EarlyStoppingCallback,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback, set_seed)

import evaluacion

RAIZ = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    modelo: str = "facebook/bart-base"
    epocas: int = 10
    lote: int = 32                 # 4,4 GB de pico en la RTX 4060 Ti; 64 rozó el límite
    acumulacion: int = 1           # lotes por paso del optimizador (lote efectivo = lote × acumulacion)
    tasa: float = 3e-5             # learning rate de AdamW
    calentamiento: float = 0.05    # fracción de pasos con la tasa en subida
    decaimiento: float = 0.01      # weight decay
    suavizado: float = 0.1         # label smoothing
    largo: int = 128               # piezas BPE, entrada y salida (entrega 1, notebook 02 T3)
    haces: int = 4                 # beam search al generar
    paciencia: int = 3             # épocas sin mejorar F0.5 antes de parar
    semilla: int = 42
    muestra: int | None = None     # recorta train y dev para una prueba rápida


def cargar_pares(ruta: Path, muestra: int | None = None) -> Dataset:
    filas = [json.loads(l) for l in ruta.read_text(encoding="utf-8").splitlines()]
    return Dataset.from_list(filas[:muestra] if muestra else filas)


def tokenizar_pares(tokenizador, largo: int):
    def tokenizar(lote):
        return tokenizador(lote["original"], text_target=lote["corregida"], max_length=largo, truncation=True)
    return tokenizar


class Registro(TrainerCallback):
    """Escribe cada log del Trainer como una línea de registro.jsonl."""

    def __init__(self, ruta: Path):
        self.ruta = ruta

    def on_log(self, args, state, control, logs=None, **kwargs):
        with self.ruta.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"paso": state.global_step, "epoca": round(state.epoch or 0, 3), **(logs or {})}) + "\n")


def metrica_f05(tokenizador, dev: evaluacion.Conjunto, carpeta_predicciones: Path):
    """compute_metrics: decodifica lo generado, guarda las predicciones y devuelve F0.5 de ERRANT."""
    carpeta_predicciones.mkdir(parents=True, exist_ok=True)
    llamadas = 0

    def calcular(prediccion):
        nonlocal llamadas
        ids = np.where(prediccion.predictions != -100, prediccion.predictions, tokenizador.pad_token_id)
        textos = [t.strip() for t in tokenizador.batch_decode(ids, skip_special_tokens=True)]
        with (carpeta_predicciones / f"evaluacion_{llamadas:02d}.jsonl").open("w", encoding="utf-8") as f:
            for fila, texto in zip(dev.entrada, textos):
                f.write(json.dumps({"id": fila["id"], "prediccion": texto}, ensure_ascii=False) + "\n")
        llamadas += 1
        r = evaluacion.f05(dev, evaluacion.tokenizar(textos))
        return {"f05": r["f05"], "precision": r["precision"], "recall": r["recall"]}

    return calcular


def cargar_modelo(config: Config):
    """Tokenizador y modelo preentrenados, con la generación ajustada para corregir."""
    tokenizador = AutoTokenizer.from_pretrained(config.modelo)
    # transformers 5 carga en el tipo guardado: bart-large viene en float16 y AdamW lo vuelve NaN al primer paso.
    # Los pesos maestros van en float32; bf16=True solo hace los cálculos en bf16.
    modelo = AutoModelForSeq2SeqLM.from_pretrained(config.modelo, dtype=torch.float32)
    # bart-base prohíbe repetir trigramas al generar; en corrección hay que poder copiar "in the ... in the"
    modelo.generation_config.no_repeat_ngram_size = 0
    # Quedan guardados con el modelo: la app y el notebook 05 generan igual que la selección
    modelo.generation_config.max_length = config.largo
    modelo.generation_config.num_beams = config.haces
    return tokenizador, modelo


def preparar_datos(config: Config, datos: Path, tokenizador):
    """train.jsonl y dev_seleccion tokenizados, más dev_seleccion tal cual para medir F0.5."""
    dev = evaluacion.leer_conjunto(datos / "dev_seleccion")
    if config.muestra:
        dev = dev.primeros(config.muestra)
    tokenizar = tokenizar_pares(tokenizador, config.largo)
    columnas = ["id", "original", "corregida"]
    train_ds = cargar_pares(datos / "train.jsonl", config.muestra).map(tokenizar, batched=True, remove_columns=columnas)
    dev_ds = Dataset.from_list(dev.entrada).map(tokenizar, batched=True, remove_columns=columnas)
    return train_ds, dev_ds, dev


def argumentos(config: Config, salida: Path, n_train: int) -> Seq2SeqTrainingArguments:
    pasos = math.ceil(n_train / (config.lote * config.acumulacion)) * config.epocas
    return Seq2SeqTrainingArguments(
        output_dir=str(salida),
        num_train_epochs=config.epocas,
        per_device_train_batch_size=config.lote,
        gradient_accumulation_steps=config.acumulacion,
        per_device_eval_batch_size=64,
        learning_rate=config.tasa,
        warmup_steps=int(config.calentamiento * pasos),
        weight_decay=config.decaimiento,
        label_smoothing_factor=config.suavizado,
        bf16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        eval_on_start=True,
        save_total_limit=2,
        save_only_model=True,
        load_best_model_at_end=True,
        metric_for_best_model="f05",
        predict_with_generate=True,
        generation_max_length=config.largo,
        generation_num_beams=config.haces,
        logging_steps=50,
        dataloader_num_workers=2,
        report_to="none",
        seed=config.semilla,
    )


def crear_trainer(config: Config, salida: Path, tokenizador, modelo, train_ds, dev_ds, dev) -> Seq2SeqTrainer:
    return Seq2SeqTrainer(
        model=modelo,
        args=argumentos(config, salida, len(train_ds)),
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizador, model=modelo),
        processing_class=tokenizador,
        compute_metrics=metrica_f05(tokenizador, dev, salida / "predicciones"),
        callbacks=[Registro(salida / "registro.jsonl"), EarlyStoppingCallback(config.paciencia)],
    )


def ejecutar(trainer: Seq2SeqTrainer, config: Config, salida: Path) -> None:
    """Entrena desde cero el registro y guarda el mejor checkpoint en salida/mejor."""
    salida.mkdir(parents=True, exist_ok=True)
    (salida / "registro.jsonl").unlink(missing_ok=True)
    (salida / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")
    trainer.train()
    trainer.save_model(str(salida / "mejor"))
    (salida / "mejor" / "elegido.json").write_text(json.dumps({
        "checkpoint": trainer.state.best_model_checkpoint, "f05_dev_seleccion": trainer.state.best_metric,
    }, indent=2) + "\n", encoding="utf-8")


def entrenar(config: Config, datos: Path, salida: Path) -> Seq2SeqTrainer:
    """El pipeline completo; el notebook 04 llama los mismos pasos uno por uno."""
    set_seed(config.semilla)
    tokenizador, modelo = cargar_modelo(config)
    train_ds, dev_ds, dev = preparar_datos(config, datos, tokenizador)
    trainer = crear_trainer(config, salida, tokenizador, modelo, train_ds, dev_ds, dev)
    ejecutar(trainer, config, salida)
    return trainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datos", type=Path, default=RAIZ / "datos" / "procesados" / "entrenamiento")
    parser.add_argument("--salida", type=Path, default=RAIZ / "modelos" / "bart-base-gec")
    parser.add_argument("--modelo", default=Config.modelo)
    parser.add_argument("--epocas", type=int, default=Config.epocas)
    parser.add_argument("--lote", type=int, default=Config.lote)
    parser.add_argument("--acumulacion", type=int, default=Config.acumulacion)
    parser.add_argument("--tasa", type=float, default=Config.tasa)
    parser.add_argument("--muestra", type=int, default=None)
    args = parser.parse_args()
    config = Config(modelo=args.modelo, epocas=args.epocas, lote=args.lote, acumulacion=args.acumulacion,
                    tasa=args.tasa, muestra=args.muestra)
    entrenar(config, args.datos, args.salida)


if __name__ == "__main__":
    main()
