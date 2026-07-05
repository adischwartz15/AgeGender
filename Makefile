.PHONY: install test download-data prepare-data pretrain train experiments \
        calibrate build-knn evaluate robustness gradcam architecture-report \
        api frontend docker-up

PYTHON ?= python
CHECKPOINT ?= checkpoints/multitask_best_balanced_score.pt
ARGS ?=

install:
	$(PYTHON) -m pip install -r requirements.txt
	cd frontend && npm install

test:
	$(PYTHON) -m pytest tests/ -v

download-data:
	$(PYTHON) scripts/download_kaggle_data.py

prepare-data:
	$(PYTHON) scripts/prepare_data.py

pretrain:
	$(PYTHON) scripts/pretrain.py $(ARGS)

train:
	$(PYTHON) scripts/train.py $(ARGS)

experiments:
	$(PYTHON) scripts/run_experiments.py $(ARGS)

calibrate:
	$(PYTHON) scripts/calibrate.py --checkpoint $(CHECKPOINT)

build-knn:
	$(PYTHON) scripts/build_knn_index.py --checkpoint $(CHECKPOINT)

evaluate:
	$(PYTHON) scripts/evaluate.py --checkpoint $(CHECKPOINT) --compare-knn

robustness:
	$(PYTHON) scripts/run_robustness.py --checkpoint $(CHECKPOINT)

gradcam:
	$(PYTHON) scripts/generate_gradcam.py --checkpoint $(CHECKPOINT)

architecture-report:
	$(PYTHON) scripts/generate_architecture_report.py --checkpoint $(CHECKPOINT)

api:
	$(PYTHON) -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

frontend:
	cd frontend && npm run dev

docker-up:
	docker compose up --build
