# Entorno del proyecto final en la torre: Jupyter en Docker, accesible por Tailscale.
COMPOSE := docker compose --env-file infra/.env -f infra/compose.yaml
PORT    ?= 8888
SALIDA  ?= modelos/bart-base-gec

.DEFAULT_GOAL := help
.PHONY: help env build up down restart logs status url shell gpu datos test analisis preparar entrenar entrenar-large progreso evaluar

help: ## Muestra esta ayuda
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  make %-15s %s\n", $$1, $$2}'

infra/.env:
	@printf 'JUPYTER_TOKEN=%s\nJUPYTER_BIND=%s\nJUPYTER_PORT=%s\nHOST_UID=%s\nHOST_GID=%s\n' \
	  "$$(openssl rand -hex 24)" "$$(tailscale ip -4)" "$(PORT)" "$$(id -u)" "$$(id -g)" > $@
	@echo "Creado $@ (token nuevo)"

env: infra/.env ## Crea infra/.env con token, IP de Tailscale y UID

build: infra/.env ## Construye la imagen
	$(COMPOSE) build

up: infra/.env ## Levanta Jupyter (se reinicia solo con la torre)
	$(COMPOSE) up -d --build
	@$(MAKE) --no-print-directory url

down: ## Detiene Jupyter
	$(COMPOSE) down

restart: ## Reinicia Jupyter
	$(COMPOSE) restart

logs: ## Sigue los logs
	$(COMPOSE) logs -f

status: ## Estado del contenedor
	$(COMPOSE) ps

url: infra/.env ## Imprime la URL para abrir desde el navegador
	@v() { sed -n "s/^$$1=//p" infra/.env; }; echo "http://$$(v JUPYTER_BIND):$$(v JUPYTER_PORT)/lab?token=$$(v JUPYTER_TOKEN)"

shell: ## Abre una terminal dentro del contenedor
	$(COMPOSE) exec jupyter bash

gpu: ## Comprueba que el contenedor ve la GPU y que PyTorch la usa (bf16)
	$(COMPOSE) exec jupyter nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
	$(COMPOSE) exec jupyter python -c "import torch; x = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16); \
	  print('torch', torch.__version__, '· CUDA', torch.version.cuda, '·', torch.cuda.get_device_name(0), \
	  '· bf16', torch.cuda.is_bf16_supported(), '· matmul ok', bool((x @ x).isfinite().all()))"

datos: ## Descarga FCE y W&I+LOCNESS en datos/crudos/
	./scripts/descargar_datos.sh

test: ## Corre los tests de src/
	$(COMPOSE) exec jupyter pytest -q

analisis: ## Ejecuta los notebooks de análisis en orden (el 02 usa las tablas del 01)
	$(COMPOSE) exec jupyter jupyter nbconvert --to notebook --execute --inplace \
	  --ExecutePreprocessor.timeout=1800 notebooks/01_analisis_datos.ipynb notebooks/02_analisis_tareas.ipynb

preparar: ## Genera train.jsonl y los conjuntos de evaluación en datos/procesados/entrenamiento/
	$(COMPOSE) exec jupyter python src/preparar.py

entrenar: ## Afina BART-base en segundo plano (≈ 30 min); el log queda en modelos/entrenamiento.log
	mkdir -p modelos
	$(COMPOSE) exec -d jupyter bash -c "python src/entrenar.py > modelos/entrenamiento.log 2>&1"
	@echo "Entrenando. Seguimiento: make progreso"

entrenar-large: ## Afina BART-large (el modelo elegido, ≈ 1,5 h) en segundo plano; log en modelos/entrenamiento-large.log
	mkdir -p modelos
	$(COMPOSE) exec -d jupyter bash -c "python src/entrenar.py --modelo facebook/bart-large --lote 16 --acumulacion 2 \
	  --salida modelos/experimentos/large-3e-5 > modelos/entrenamiento-large.log 2>&1"
	@echo "Entrenando. Seguimiento: make progreso SALIDA=modelos/experimentos/large-3e-5"

progreso: ## Pérdida y F0.5 por época del entrenamiento en SALIDA (por defecto modelos/bart-base-gec)
	@$(COMPOSE) exec jupyter python -c "import json; \
	  [print(f\"época {r['epoca']:>5} · F0.5 {r['eval_f05']:.4f} · P {r['eval_precision']:.3f} · R {r['eval_recall']:.3f}\") if 'eval_f05' in r \
	   else print(f\"época {r['epoca']:>5} · pérdida {r['loss']:.3f}\") for r in map(json.loads, open('$(SALIDA)/registro.jsonl')) \
	   if 'eval_f05' in r or r['paso'] % 500 == 0 and 'loss' in r]"

evaluar: ## Ejecuta el notebook 05 (selección del modelo y evaluación final en test)
	$(COMPOSE) exec jupyter jupyter nbconvert --to notebook --execute --inplace \
	  --ExecutePreprocessor.timeout=1800 notebooks/05_evaluacion.ipynb
