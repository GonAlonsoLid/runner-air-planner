.PHONY: help install build up down logs shell collect train predict

help:
	@echo "Comandos disponibles:"
	@echo "  make install     - Instalar dependencias con Poetry"
	@echo "  make build       - Construir imagen Docker"
	@echo "  make up           - Levantar contenedor con docker-compose"
	@echo "  make down         - Detener contenedor"
	@echo "  make logs         - Ver logs del contenedor"
	@echo "  make shell        - Abrir shell en el contenedor"
	@echo "  make collect      - Recopilar datos"
	@echo "  make train        - Entrenar modelo"
	@echo "  make predict      - Hacer predicciones"

install:
	poetry install

build:
	docker-compose build

up:
	docker-compose up -d
	@echo "App disponible en http://localhost:8501"

down:
	docker-compose down

logs:
	docker-compose logs -f

shell:
	docker-compose exec runner-air-planner bash

collect:
	docker-compose exec runner-air-planner poetry run collect --accumulate

train:
	docker-compose exec runner-air-planner poetry run train

predict:
	docker-compose exec runner-air-planner poetry run predict

# Development mode
dev-up:
	docker-compose -f docker-compose.dev.yml up

dev-down:
	docker-compose -f docker-compose.dev.yml down

dev-shell:
	docker-compose -f docker-compose.dev.yml exec app bash

