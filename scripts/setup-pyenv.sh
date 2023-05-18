#!/bin/bash
# This script is supposed to be run in the Docker image of the project

set -ex

export PYENV_ROOT="/rl-lightning/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"