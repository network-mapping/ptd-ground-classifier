PYTHON ?= python3
VENV   := .venv
PY     := $(VENV)/bin/python

.PHONY: venv build test regression memory tin dtm dtmdiff anchor validate clean

venv:
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e .

build:
	sh native/build.sh

test: build
	$(PY) tests/test_fast_engine.py tests/validation/clips

regression: build
	$(PY) tests/test_regression.py

memory: build
	$(PY) tests/test_memory.py

tin: build
	$(PY) tests/test_tin_export.py

dtm: build
	$(PY) tests/test_dtm.py

dtmdiff: build
	$(PY) tests/test_dtm_diff.py

anchor: build
	$(PY) tests/test_seed_anchor.py

validate: build
	$(PY) tests/validation/validate_suite.py

clean:
	rm -rf build
