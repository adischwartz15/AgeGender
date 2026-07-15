.PHONY: install test download-data prepare-data pretrain train experiments \
        calibrate build-knn evaluate robustness gradcam architecture-report \
        run-seeds final-report export-report compare-backbones

PYTHON ?= python
CHECKPOINT ?= checkpoints/multitask_best_balanced_score.pt
EXPERIMENT ?= exp_d_shared_adapters_learned_balance
SEEDS ?= 42,123,1729
ARGS ?=

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest tests/ -v

download-data:
	$(PYTHON) scripts/download_kaggle_data.py

prepare-data:
	$(PYTHON) scripts/prepare_data.py $(ARGS)

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

# Re-renders docs/architecture_analysis_generated.md from whatever is
# already in outputs/ (no recomputation) -- useful after re-running only
# some pipeline stages. See scripts/export_report.py.
export-report:
	$(PYTHON) scripts/export_report.py

# Example: make compare-backbones CHECKPOINTS="simple_cnn=checkpoints/exp_0_..._best_balanced_score.pt custom_resnet18=checkpoints/exp_d_..._best_balanced_score.pt" RESNET_NAME=custom_resnet18
CHECKPOINTS ?= simple_cnn=checkpoints/exp_0_simple_cnn_shared_adapters_learned_balance_best_balanced_score.pt custom_resnet18=checkpoints/exp_d_shared_adapters_learned_balance_best_balanced_score.pt
RESNET_NAME ?= custom_resnet18
compare-backbones:
	$(PYTHON) scripts/compare_backbones.py $(foreach c,$(CHECKPOINTS),--checkpoint $(c)) --resnet-name $(RESNET_NAME)
