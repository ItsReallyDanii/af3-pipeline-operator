# Makefile — OpenFold3 + DeepSpeed environment management
# Target: Linux x86_64 · Python 3.12 · CUDA 12.8 driver · single GPU
#
# Usage:
#   make env.create      # Create conda env from scratch
#   make env.verify      # Run preflight checks
#   make env.freeze      # Snapshot current env state
#   make diag.collect    # Gather full diagnostics bundle
#   make run.dry         # Validate config/paths only (no inference)
#
# Prerequisites:
#   - conda (miniconda or miniforge) on PATH
#   - NVIDIA driver >= 550 installed (for CUDA 12.8)

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help
MAKEFLAGS += --warn-undefined-variables --no-builtin-rules

# ── Configuration ─────────────────────────────────────────────────────────────

ENV_NAME      := of3ds
ENV_FILE      := environment.yml
REQ_LOCK      := requirements-lock.txt
CONFIG        := configs/smoke_test.yaml
OUTPUT_DIR    := runs
FREEZE_DIR    := .env_snapshots
DIAG_DIR      := .diagnostics
CONDA         := conda
CONDA_RUN     := $(CONDA) run -n $(ENV_NAME) --no-banner --live-stream
PYTHON        := $(CONDA_RUN) python
PIP           := $(CONDA_RUN) pip

# Timestamp for snapshot naming
TS := $(shell date +%Y%m%dT%H%M%S)

# ── Help ──────────────────────────────────────────────────────────────────────

.PHONY: help
help: ## Show this help
	@echo ""
	@echo "OpenFold3 + DeepSpeed Environment Targets"
	@echo "=========================================="
	@echo ""
	@grep -E '^[a-zA-Z_.]+:.*##' $(MAKEFILE_LIST) | \
		awk -F ':.*## ' '{printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ── env.create ────────────────────────────────────────────────────────────────

.PHONY: env.create
env.create: $(ENV_FILE) $(REQ_LOCK) ## Create conda environment from environment.yml
	@echo "=== Removing existing env '$(ENV_NAME)' if present ==="
	-$(CONDA) env remove -n $(ENV_NAME) -y 2>/dev/null || true
	@echo ""
	@echo "=== Creating conda env '$(ENV_NAME)' ==="
	$(CONDA) env create -f $(ENV_FILE) -n $(ENV_NAME) --yes
	@echo ""
	@echo "=== Installing pip dependencies (--no-deps to respect lock) ==="
	$(PIP) install --no-deps -r $(REQ_LOCK)
	@echo ""
	@echo "=== Verifying pip dependency consistency ==="
	$(PIP) check || echo "WARNING: pip check reported issues (review above)"
	@echo ""
	@echo "=== Environment '$(ENV_NAME)' created. Activate with: conda activate $(ENV_NAME) ==="

# ── env.verify ────────────────────────────────────────────────────────────────

.PHONY: env.verify
env.verify: ## Run preflight environment checks
	@echo "=== Running preflight checks ==="
	@mkdir -p $(DIAG_DIR)
	$(PYTHON) scripts/preflight.py --output $(DIAG_DIR)/preflight_$(TS).json
	@echo ""
	@echo "=== Preflight report: $(DIAG_DIR)/preflight_$(TS).json ==="

# ── env.freeze ────────────────────────────────────────────────────────────────

.PHONY: env.freeze
env.freeze: ## Snapshot current environment state
	@echo "=== Freezing environment state ==="
	@mkdir -p $(FREEZE_DIR)
	$(CONDA) run -n $(ENV_NAME) --no-banner conda env export > $(FREEZE_DIR)/conda_export_$(TS).yml
	$(CONDA) run -n $(ENV_NAME) --no-banner conda list --export > $(FREEZE_DIR)/conda_list_$(TS).txt
	$(CONDA) run -n $(ENV_NAME) --no-banner conda list --json > $(FREEZE_DIR)/conda_list_$(TS).json
	$(PIP) freeze > $(FREEZE_DIR)/pip_freeze_$(TS).txt
	$(PIP) list --format=json > $(FREEZE_DIR)/pip_list_$(TS).json
	$(PYTHON) -c "import torch; print(f'torch={torch.__version__} cuda={torch.version.cuda}')" \
		> $(FREEZE_DIR)/torch_version_$(TS).txt 2>&1 || true
	@echo ""
	@echo "=== Snapshot saved to $(FREEZE_DIR)/*_$(TS).* ==="
	@ls -la $(FREEZE_DIR)/*_$(TS).*

# ── diag.collect ──────────────────────────────────────────────────────────────

.PHONY: diag.collect
diag.collect: ## Collect full diagnostics bundle
	@echo "=== Collecting diagnostics ==="
	@mkdir -p $(DIAG_DIR)
	$(PYTHON) scripts/collect_diagnostics.py --output-dir $(DIAG_DIR)
	@echo ""
	@echo "=== Diagnostics written to $(DIAG_DIR)/ ==="
	@ls -la $(DIAG_DIR)/diagnostics_bundle.*

# ── run.dry ───────────────────────────────────────────────────────────────────

.PHONY: run.dry
run.dry: ## Dry-run: validate config, paths, inputs (NO inference)
	@echo "=== Dry-run validation ==="
	$(PYTHON) orchestrator_dry.py --config $(CONFIG) --output-base $(OUTPUT_DIR)
	@echo ""
	@echo "=== Dry-run complete ==="

# ── Convenience targets ──────────────────────────────────────────────────────

.PHONY: env.remove
env.remove: ## Remove the conda environment entirely
	@echo "=== Removing conda environment '$(ENV_NAME)' ==="
	$(CONDA) env remove -n $(ENV_NAME) --yes
	@echo "=== Removed ==="

.PHONY: env.rebuild
env.rebuild: env.remove env.create env.verify ## Full rebuild: remove + create + verify
	@echo "=== Rebuild complete ==="

.PHONY: env.diff
env.diff: ## Diff current env against target lockfiles
	@echo "=== Diffing current env vs target ==="
	@mkdir -p /tmp/_of3ds_diff
	@$(PIP) freeze | sort > /tmp/_of3ds_diff/pip_current.txt
	@grep -v '^\(#\|--\|-r\|$$\)' $(REQ_LOCK) | grep '==' | sort > /tmp/_of3ds_diff/pip_target.txt
	@echo "--- pip: current vs requirements-lock.txt ---"
	@diff --color=auto -u /tmp/_of3ds_diff/pip_target.txt /tmp/_of3ds_diff/pip_current.txt || true
	@echo ""
	@$(CONDA_RUN) conda list --export | grep -v '^#' | sort > /tmp/_of3ds_diff/conda_current.txt
	@echo "--- conda: current package count ---"
	@echo "Installed: $$(wc -l < /tmp/_of3ds_diff/conda_current.txt) conda packages"
	@echo "(Full conda diff: review /tmp/_of3ds_diff/conda_current.txt vs $(ENV_FILE))"
	@rm -rf /tmp/_of3ds_diff
	@echo ""
	@echo "=== Diff complete. Lines with + are extra; - are missing. ==="

.PHONY: env.rollback
env.rollback: ## Rollback: recreate env from most recent snapshot or lockfiles
	@LATEST=$$(ls -t $(FREEZE_DIR)/conda_export_*.yml 2>/dev/null | head -1); \
	if [ -n "$$LATEST" ]; then \
		echo "=== Rolling back to snapshot: $$LATEST ==="; \
		$(CONDA) env remove -n $(ENV_NAME) -y 2>/dev/null || true; \
		$(CONDA) env create -f "$$LATEST" -n $(ENV_NAME) --yes; \
	else \
		echo "=== No snapshot found. Rebuilding from lockfiles ==="; \
		$(CONDA) env remove -n $(ENV_NAME) -y 2>/dev/null || true; \
		$(CONDA) env create -f $(ENV_FILE) -n $(ENV_NAME) --yes; \
		$(PIP) install --no-deps -r $(REQ_LOCK); \
	fi
	@echo "=== Rollback complete. Run 'make env.verify' to confirm. ==="

.PHONY: clean
clean: ## Remove generated diagnostic/snapshot files
	rm -rf $(DIAG_DIR) $(FREEZE_DIR)
	rm -f preflight_report.json
	@echo "=== Cleaned diagnostic and snapshot directories ==="

.PHONY: all
all: env.create env.verify env.freeze diag.collect run.dry ## Full pipeline: create + verify + freeze + diag + dry-run
	@echo ""
	@echo "=== All targets complete ==="
