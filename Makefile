.PHONY: install test download-data prepare-data pretrain train experiments \
        calibrate build-knn evaluate robustness gradcam architecture-report \
        run-seeds final-report demo-images demo-readiness demo api frontend

PYTHON ?= python
CHECKPOINT ?= checkpoints/multitask_best_balanced_score.pt
EXPERIMENT ?= exp_d_shared_adapters_learned_balance
SEEDS ?= 42,43,44
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

run-seeds:
	$(PYTHON) scripts/run_seeds.py --experiment $(EXPERIMENT) --seeds $(SEEDS)

final-report:
	$(PYTHON) scripts/generate_final_report.py

demo-images:
	$(PYTHON) scripts/generate_demo_images.py

demo-readiness:
	$(PYTHON) scripts/check_demo_readiness.py

demo:
	$(PYTHON) scripts/run_demo.py

api:
	$(PYTHON) -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

frontend:
	cd frontend && npm run dev
