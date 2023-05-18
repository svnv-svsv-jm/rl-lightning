# Reproduce experiments

Experiments make use of [Hydra](https://hydra.cc/) for configuration files. Config files cover all classes such as datamodules, models, trainers, etc. and the whole configuration file is then logged / saved to the current run's log directory, for reproducibility.

Configurations are in [configs](../configs/)

## Config files: explained

Using [Hydra](https://hydra.cc/), you can run a script (say `main.py`) and change the configuration file it has to load as follows:

```bash
python main.py --config-name <name>
```

You can create your own configuration file. A configuration file will look something like this (:warning: the example file below may be outdated):

```yaml
# @package _global_

defaults:
  - paths: default.yaml
  - hydra: default.yaml
  - extras: default.yaml # for common params
  - callbacks: default.yaml
  - logger: default.yaml
  - pipeline: gran.yaml
  - model: gran.yaml
  - datamodule: proteins.yaml
  - _self_

ckpt_path: null
tag: "my-cool-tag" # the loggin dir will be named after this tag
stage: fit # must be 'fit', 'validate', 'test' or 'assess'

seed_everything: true
trainer:
  accelerator: auto
  num_sanity_val_steps: 2
  log_every_n_steps: 50
  min_epochs: 10
  max_epochs: 1000
  callbacks:
    - _target_: rl.<some-class-name>
```

The objective of this file is to create a model, a datamodule and a trainer with the desired callbacks.

The callbacks are defined both in the `defaults` and `trainer` section. They will be merged, so both will count.

Everything will be logged to `${paths.log_dir}/${tag}/runs/${now:%Y-%m-%d}/${now:%H-%M-%S}`. By defaults, `${paths.log_dir}` will interpolate to `lightning_logs`. You can override this or change it manually in the file `configs/paths/default.yaml`.

You can also override the configuration via the command line. See [Hydra](https://hydra.cc/).

### Override

To override configuration from the CLI, see the following example:

```bash
python -u experiments/main.py --config-name test.yaml tag="yo" +dp=true
```

Here, we are overriding the configuration file, the `tag` field and the `dp` field (differential privacy).

### Sweeping and HPO

With Hydra, you can seamlessly run multiples experiments with a single configuration file (hydra basic sweeping), instead of having to do so manually. Plus, you can also perform Hyper-Parameter Optimization (HPO).

Let's see an example, where we also use Ray to launch our jobs:

```yaml
# @package _global_

defaults:
  - paths: default.yaml
  - hydra: default.yaml
  - extras: default.yaml
  - callbacks: test.yaml
  - logger: test.yaml
  - model: mnist-classifier.yaml
  - datamodule: mnist-test.yaml
  - trainer: test.yaml
  - _self_
  - override hydra/sweeper: optuna
  - override hydra/sweeper/sampler: tpe
  - override hydra/launcher: ray

hydra:
  mode: MULTIRUN
  sweeper:
    direction: minimize
    study_name: test
    storage: null
    n_trials: 3
    n_jobs: 2
    params:
      model.normalize: choice(True, False)
      model.lr: interval(0.001, 0.1)

optimize_metric: loss/val

ckpt_path: null

tag: _hydra-sweeper
stage: fit

seed_everything: true
```

You will have multiple jobs running (serially or in parallel, depending on the launcher). Plus, if the key `optimize_metric` is set and you're using `hydra/sweeper: optuna`, you will also get information about the best run. For example:

```yaml
# optimization_results.yaml
name: optuna
best_params:
  model.normalize: true
  model.lr: 0.011852952440386172
best_value: 0.6395682096481323 # this is the "loss/val" metric value
```

## Docker: run in container

Make sure you have the project's image (see [here](../README.md)). Once you have, you can run experiments using Docker.

For example:

```bash
docker run --rm -it -d --shm-size 4G --network=host --volume .:/workdir -e LOCAL_USER_ID -e LOCAL_USER \
    --runtime=nvidia --gpus all \
    --name run-4934 \
    -t rl-lightning \
    /rl-lightning/bin/python -u /workdir/experiments/main.py --config-name test.yaml
```

Typing this each time you want to run an experiment would be crazy. By using `make`, you can run available commands more easily.

Examples:

```bash
# specify config file, additional flags such as '-d' (detached mode) and run on gpu
make run-gpu CONFIG=test.yaml DOCKER_FLAGS="-d"
```

```bash
# specify config file, override hydra config and run (on cpu)
make run CONFIG=test.yaml OVERRIDE='+model.cluster_size=50'
```

The general `make run` command looks something like this:

```Makefile
$(DOCKER) run --rm -it $(DOCKER_FLAGS) $(DOCKER_COMMON_FLAGS) \
  $(GPU_FLAGS) \
  --name $(PROJECT_NAME)-run-$(CONTAINER_NAME) \
  -t $(REGISTRY)/$(PROJECT_NAME) \
  $(CMD)
```

According to what command you choose (`make run`, `make run-gpu`, etc.) those environment variables will be interpolated differently. You can also manually override them, of course.

For example, you can run the `make run` command but manually apply GPU support:

```bash
make run GPU_FLAGS='--runtime=nvidia --gpus all' CONFIG=test.yaml
```

The variables `GPU_FLAGS` and `DOCKER_FLAGS` are empty by default. The command `make run-gpu` automatically sets `GPU_FLAGS='--runtime=nvidia --gpus all'`.

> :warning: It is not recommended to change the other variables. Only change `DOCKER_FLAGS`, `CONFIG` and `OVERRIDE`.

## GPU support

You have two ways to control how many and which GPUs your job should use.

**Method 1**: One way is by using the PyTorch Lightning API (`Trainer` class). So you expose all GPUs `GPU_FLAGS='--runtime=nvidia --gpus all'` by default and control which one to use via the config file:

```yaml
trainer:
  enable_checkpointing: true
  accelerator: gpu
  # auto_select_gpus: true # may select any available gpu, ignoring kwarg `gpus`
  gpus: [1] # if you are exposing a single device, no need for this
```

**Method 2**: Another way is to expose only a specific GPU (e.g. `GPU_FLAGS=--runtime=nvidia --gpus '"device=1"'` or any other way) and then let the `Trainer` find it:

```yaml
trainer:
  enable_checkpointing: true
  accelerator: gpu
  auto_select_gpus: true # may select any available gpu, ignoring kwarg `gpus`
  # gpus: [1] # if you are exposing a single device, no need for this
```

Either way should work. The second method, however, is more robust: only the GPU you want to use is exposed to the contaier, no bug in your code can violate that.
