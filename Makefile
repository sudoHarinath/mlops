.PHONY: install install-remote train-local train-remote clean

install:
	python -m pip install -e .

install-remote:
	python -m pip install -e ".[remote]"

train-local:
	python -m src.train --profile local

train-remote:
	python -m src.train --profile remote

clean:
	rm -rf outputs/ wandb/ .pytest_cache/ *.egg-info/
