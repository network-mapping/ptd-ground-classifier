PYTHON ?= python3
VENV   := .venv
PY     := $(VENV)/bin/python

.PHONY: venv build test validate clean

venv:
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e .

build:
	sh native/build.sh

test: build
	$(PY) tests/test_fast_engine.py tests/validation/clips

validate: build
	$(PY) tests/validation/validate_suite.py

clean:
	rm -rf build
