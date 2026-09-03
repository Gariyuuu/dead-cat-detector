# Dead Cat Detector - full reproduction pipeline
PY := .venv/bin/python
PIP := VIRTUAL_ENV=.venv uv pip

.PHONY: help setup data events study regress models robust figures notebooks analysis verify test all clean

help:
	@echo "make setup    - create .venv and install dependencies"
	@echo "make data     - build point-in-time universe and download prices"
	@echo "make events   - detect crash events, build features and outcomes"
	@echo "make study    - event study with bootstrap confidence bands"
	@echo "make regress  - OLS (HC3 + date-clustered) and logistic regressions"
	@echo "make models   - predictive experiment, calibration, SHAP"
	@echo "make robust   - 576-specification robustness grid"
	@echo "make figures  - regenerate every figure from persisted results"
	@echo "make notebooks- regenerate and execute the five notebooks"
	@echo "make analysis - rebuild every analysis stage from cached data"
	@echo "make test     - run the test suite"
	@echo "make verify   - check every documented number against results"
	@echo "make all      - the whole pipeline, end to end"

setup:
	uv venv --python 3.12 .venv
	$(PIP) install -e ".[dev]"

data:
	$(PY) scripts/01_build_dataset.py

events:
	$(PY) scripts/02_build_events.py

study:
	$(PY) scripts/03_event_study.py

regress:
	$(PY) scripts/04_regressions.py

models:
	$(PY) scripts/05_models.py

robust:
	$(PY) scripts/06_robustness.py

figures:
	$(PY) scripts/07_figures.py

notebooks:
	@for n in 01_data_audit 02_event_construction 03_event_study 04_predictive_models 05_robustness; do \
		$(PY) scripts/build_notebooks.py $$n.ipynb || exit 1; \
	done

verify:
	$(PY) scripts/verify_claims.py

test:
	$(PY) -m pytest tests/ -q

# Everything downstream of the download: assumes data/processed is already built.
analysis: events study regress models robust figures

all: data events study regress models robust figures notebooks test verify

clean:
	rm -rf results/figures/*.png results/tables/*.csv results/metrics/*.json
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
