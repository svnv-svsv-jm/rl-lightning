#!/bin/bash
set -ex

PYTHON="${PYTHON:=/rl-lightning/bin/python}"

$PYTHON -m pytest --testmon --mypy --pylint --all | tee logs/pytest.log
$PYTHON -m pytest --testmon --nbmake --overwrite "./examples" | tee logs/pytest-nb.log