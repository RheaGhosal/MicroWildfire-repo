PY=python
CONFIG?=configs/config.yaml

all: split fusion ci tables

split:
	$(PY) scripts/split_data.py --config $(CONFIG)

fusion:
	$(PY) scripts/run_fusion.py --config $(CONFIG)

ci:
	$(PY) scripts/run_bootstrap_ci.py --config $(CONFIG)

tables:
	$(PY) scripts/reproduce_tables.py --config $(CONFIG)
