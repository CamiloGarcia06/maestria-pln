# Entorno del proyecto final en la torre: Jupyter en Docker, accesible por Tailscale.
COMPOSE := docker compose --env-file infra/.env -f infra/compose.yaml
PORT    ?= 8888

.DEFAULT_GOAL := help
.PHONY: help env build up down restart logs status url shell gpu datos

help: ## Muestra esta ayuda
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  make %-8s %s\n", $$1, $$2}'

infra/.env:
	@printf 'JUPYTER_TOKEN=%s\nJUPYTER_BIND=%s\nJUPYTER_PORT=%s\nUID=%s\nGID=%s\n' \
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
	@. ./infra/.env && echo "http://$$JUPYTER_BIND:$$JUPYTER_PORT/lab?token=$$JUPYTER_TOKEN"

shell: ## Abre una terminal dentro del contenedor
	$(COMPOSE) exec jupyter bash

gpu: ## Comprueba que el contenedor ve la GPU
	$(COMPOSE) exec jupyter nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv

datos: ## Descarga FCE y W&I+LOCNESS en datos/crudos/
	./scripts/descargar_datos.sh
