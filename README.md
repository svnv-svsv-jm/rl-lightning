# Reinforcement Learning with PyTorch Lightning

> :warning: **Work in progress**: this is yet to be finished. This repository is currently available merely as a preview or a template for a deep learning project.

RL models implemented in [PyTorch Lightning](https://www.lightning.ai).

## Installation via virtual environment

### Pre-requisites

Make sure you have Python `>=3.8.13` (`3.10.10` recommended).

Create a virtual environment with any tool you prefer.

#### Create a virtual environment

Use any tool to create a virtual environment with the indicated Python version.

With [Pyenv](https://github.com/pyenv/pyenv) (**recommended**, here is the [installer](https://github.com/pyenv/pyenv-installer)), it is very easy to install the Python version you want, create a virtual environment and activate it.

Once Pyenv is installed, run:

```bash
pyenv install 3.10.10
pyenv virtualenv 3.10.10 rl-lightning
pyenv activate rl-lightning
```

or just:

```bash
make init
```

#### Requirements

Once the virtual environment is active, install all dependencies by running:

```bash
make install
```

For this command to work, you have to make sure that the `python` command resolves to the correct virtual environment (the one you just created).

### Getting Started

Install the present project by running:

```bash
# install in dev mode so you can also mess around
pip install -e .
```

This command should not be necessary if you ran `make install`.

## Installation via Docker

Build project's image as follows:

```bash
docker build -t rl-lightning -f Dockerfile .
# or alternatively
make build
```

The present folder will be mounted at `/workdir`.

### More docker-related features

* To start a Bash session in the container, run: `make bash`
* To start a Jupyter Notebook (+ a Tensorboard), run: `make up`. Then run `make down` when you're done.
* To run PyTest from the container, run: `make pytest-image`
* To start a Python session in the container, run: `make docker-python`
* To start a [development container](https://code.visualstudio.com/docs/remote/create-dev-container), run: `make dev-container` or `make up` (if you go for this one, you can then kill the container with `make down`). Make sure that you're not `root` in the container. To see if a user with your local username and userid exists inside the container, run `cat /etc/passwd`. Then, you can run `su - <your-username>` to change user insinde the dev container.

Please also note that our image's entrypoint is the script [entrypoint.sh](./scripts/entrypoint.sh). This script solves common permission problems for developers who wish to mount their local source code directory on a container, as explained [here](https://denibertovic.com/posts/handling-permissions-with-docker-volumes/). **TL;DR**: you will be logged into your container as your current user, thus avoinding any permission errors.

