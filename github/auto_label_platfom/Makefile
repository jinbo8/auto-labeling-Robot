.PHONY: install api web sam2 sam3 init seed demo compose-up

install:
	pip install -r requirements.txt
	pip install -e packages/lerobot_index -e services/api -e services/worker -e services/sam2 -e services/sam3

init:
	PYTHONPATH=services/api/src:packages/lerobot_index/src:services/worker/src python -m alp_api.cli init

seed:
	PYTHONPATH=services/api/src:packages/lerobot_index/src:services/worker/src python -m alp_api.cli seed-demo

api:
	PYTHONPATH=services/api/src:packages/lerobot_index/src:services/worker/src \
		uvicorn alp_api.main:app --reload --host 0.0.0.0 --port 8000

web:
	cd apps/web && npm run dev

sam2:
	PYTHONPATH=services/sam2/src uvicorn alp_sam2.main:app --host 0.0.0.0 --port 8102

sam3:
	PYTHONPATH=services/sam3/src uvicorn alp_sam3.main:app --host 0.0.0.0 --port 8103

compose-up:
	docker compose -f deploy/compose/docker-compose.yml up --build
